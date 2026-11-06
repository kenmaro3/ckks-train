// Operation micro-benchmark of Microsoft SEAL 4.1.2 (MIT), the SEAL side of hekit's
// crates/ckks-train-bench/benches/ckks_ops.rs (M1 gate, Task 24). Single thread (SEAL has no
// internal threading).
//
//   seal_ops_bench <preset> [reps]     preset: seal_2020 | n16   (reps default 10)
//
// seal_2020: CKKS, N = 2^15, CoeffModulus::Create(N, {60, 40 x 9, 60}), tc128, scale 2^40
//            (exactly hekit's seal_2020; SEAL's last prime is its single special prime).
// n16:       CKKS, N = 2^16, CoeffModulus::Create(N, {58, 42 x 31, 58}), sec_level none,
//            scale 2^42: data primes Q of the same bit sizes as hekit's n16_train (32 primes;
//            the 31 42-bit primes are identical, the 58-bit q_0 differs because hekit draws
//            it together with its four 58-bit special primes), but SEAL's own
//            key switching (one special prime, one digit per data prime: dnum = 32, k = 1)
//            instead of hekit's dnum = 8, k = 4. SEAL has no multi-prime special modulus.
//
// Every operation runs on fresh ciphertexts at the top data level. Per operation: 1 warm-up
// run, then `reps` timed runs (steady_clock); prints median / min / mean in ms.
//
// Operations (hekit name in brackets):
//   encrypt              Encryptor::encrypt with the public key             [encrypt]
//   add                  Evaluator::add                                     [add]
//   mul_relin_rescale    multiply + relinearize_inplace + rescale_to_next   [mul_relin_rescale]
//   rotate1              rotate_vector(ct, 1)                               [rotate1]
//   rotate_x16           16 x rotate_vector of the same ct (seal_2020: steps 1..16 with 16
//                        Galois keys; n16: step 1 sixteen times, to keep key memory at 2
//                        Galois keys of 1.1 GB each; the cost of a rotation does not depend
//                        on the step)                                       [rotate_x16 and
//                        hoisted_rotate_16: SEAL has no public hoisted rotation]
//   matvec_plain_naive   d = 64 diagonal method: 63 rotations (x_i = rotate(x_{i-1}, 1)),
//                        64 multiply_plain, 63 adds                         [matvec_plain_d64]
//   matvec_plain_bsgs    BSGS 8 x 8 like hekit (pre-rotated diagonals): 7 baby rotations
//                        (step 1, incremental), 64 multiply_plain, 7 giant rotations by 8
//                        (Horner), no hoisting                              [matvec_plain_d64]
//   matvec_enc_naive     as matvec_plain_naive with encrypted diagonals: 64 multiply (no
//                        relin), 1 relinearize at the end                   [matvec_enc_d64]
//   matvec_enc_bsgs      as matvec_plain_bsgs with encrypted diagonals: 64 multiply,
//                        8 relinearize (one per giant step)                 [matvec_enc_d64]
// The matvec outputs are not rescaled (as hekit). Each matvec result is decrypted once
// (untimed) and checked against W x in the clear; the max error is printed.
#include "seal/seal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>
using namespace seal;
using namespace std;

static const size_t D = 64;
static const int BABY = 8, GIANT = 8;

struct Preset {
    string name;
    size_t n;
    vector<int> bits;
    sec_level_type sec;
    double scale;
};

static Preset preset(const string &name) {
    if (name == "seal_2020") {
        vector<int> bits{60};
        for (int i = 0; i < 9; i++) bits.push_back(40);
        bits.push_back(60);
        return {name, 32768, bits, sec_level_type::tc128, pow(2.0, 40)};
    }
    if (name == "n16") {
        vector<int> bits{58};
        for (int i = 0; i < 31; i++) bits.push_back(42);
        bits.push_back(58);
        return {name, 65536, bits, sec_level_type::none, pow(2.0, 42)};
    }
    fprintf(stderr, "unknown preset %s (seal_2020 | n16)\n", name.c_str());
    exit(2);
}

static double ms_since(chrono::steady_clock::time_point t0) {
    return chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
}

static void timed(const string &preset, const string &op, int reps, const function<void()> &f) {
    f(); // warm-up
    vector<double> t;
    for (int i = 0; i < reps; i++) {
        auto t0 = chrono::steady_clock::now();
        f();
        t.push_back(ms_since(t0));
    }
    vector<double> s = t;
    sort(s.begin(), s.end());
    double med = s.size() % 2 ? s[s.size() / 2] : 0.5 * (s[s.size() / 2 - 1] + s[s.size() / 2]);
    double mean = 0;
    for (double x : t) mean += x;
    mean /= t.size();
    printf("RESULT %s %-20s median_ms %10.4f  min_ms %10.4f  mean_ms %10.4f  reps %d\n",
           preset.c_str(), op.c_str(), med, s.front(), mean, reps);
    fflush(stdout);
}

static vector<double> rand_vals(mt19937_64 &g, size_t len) {
    uniform_real_distribution<double> d(-1.0, 1.0);
    vector<double> v(len);
    for (auto &x : v) x = d(g);
    return v;
}

// rot(v, s)[j] = v[(j + s) mod len] (SEAL's rotate_vector by s, left rotation).
static vector<double> rot(const vector<double> &v, long s) {
    long n = (long)v.size();
    vector<double> r(n);
    for (long j = 0; j < n; j++) r[j] = v[(((j + s) % n) + n) % n];
    return r;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <seal_2020|n16> [reps]\n", argv[0]);
        return 2;
    }
    Preset P = preset(argv[1]);
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    const size_t slots = P.n / 2;
    auto t0 = chrono::steady_clock::now();

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(P.n);
    parms.set_coeff_modulus(CoeffModulus::Create(P.n, P.bits));
    SEALContext ctx(parms, true, P.sec);
    if (!ctx.parameters_set()) {
        fprintf(stderr, "invalid parameters: %s\n", ctx.parameter_error_message());
        return 1;
    }
    KeyGenerator kg(ctx);
    PublicKey pk;
    kg.create_public_key(pk);
    RelinKeys rlk;
    kg.create_relin_keys(rlk);
    vector<int> steps;
    if (P.name == "seal_2020") {
        for (int s = 1; s <= 16; s++) steps.push_back(s);
        for (int k = 3; k < GIANT; k++) steps.push_back(BABY * k); // 24..56 (8, 16 present)
    } else {
        steps = {1, BABY};
    }
    GaloisKeys gk;
    kg.create_galois_keys(steps, gk);
    Encryptor encryptor(ctx, pk);
    Decryptor decryptor(ctx, kg.secret_key());
    Evaluator ev(ctx);
    CKKSEncoder encoder(ctx);

    mt19937_64 g(24);
    auto x = rand_vals(g, slots), y = rand_vals(g, slots);
    Plaintext px, py;
    encoder.encode(x, P.scale, px);
    encoder.encode(y, P.scale, py);
    Ciphertext cx, cy;
    encryptor.encrypt(px, cx);
    encryptor.encrypt(py, cy);

    // matvec data: W (d x d) uniform, v replicated with period d.
    auto wflat = rand_vals(g, D * D);
    auto v = rand_vals(g, D);
    vector<double> vr(slots), want(slots);
    for (size_t j = 0; j < slots; j++) vr[j] = v[j % D];
    for (size_t j = 0; j < slots; j++) {
        double s = 0;
        for (size_t k = 0; k < D; k++) s += wflat[(j % D) * D + k] * v[k];
        want[j] = s;
    }
    vector<vector<double>> diag(D, vector<double>(slots));
    for (size_t i = 0; i < D; i++)
        for (size_t j = 0; j < slots; j++) diag[i][j] = wflat[(j % D) * D + (j % D + i) % D];
    // pre[k1 * BABY + k2] = rot(diag[BABY k1 + k2], -BABY k1)
    vector<vector<double>> pre(D);
    for (int k1 = 0; k1 < GIANT; k1++)
        for (int k2 = 0; k2 < BABY; k2++)
            pre[k1 * BABY + k2] = rot(diag[BABY * k1 + k2], -(long)(BABY * k1));
    Plaintext pv;
    encoder.encode(vr, P.scale, pv);
    Ciphertext cv;
    encryptor.encrypt(pv, cv);
    vector<Plaintext> pd(D), ppre(D);
    vector<Ciphertext> cd(D), cpre(D);
    for (size_t i = 0; i < D; i++) {
        encoder.encode(diag[i], P.scale, pd[i]);
        encoder.encode(pre[i], P.scale, ppre[i]);
        encryptor.encrypt(pd[i], cd[i]);
        encryptor.encrypt(ppre[i], cpre[i]);
    }
    auto data = ctx.first_context_data();
    printf("SETUP %s: N = %zu, %zu data primes + 1 special, %zu Galois keys, reps %d, setup %.1f s\n",
           P.name.c_str(), P.n, data->parms().coeff_modulus().size(), steps.size(), reps,
           ms_since(t0) / 1000.0);
    fflush(stdout);

    auto check = [&](const string &op, const Ciphertext &ct) {
        Plaintext p;
        decryptor.decrypt(ct, p);
        vector<double> got;
        encoder.decode(p, got);
        double m = 0;
        for (size_t j = 0; j < slots; j++) m = max(m, fabs(got[j] - want[j]));
        printf("CHECK %s %-20s max_err 2^%.2f\n", P.name.c_str(), op.c_str(), log2(m));
        fflush(stdout);
    };

    Ciphertext out;
    vector<Ciphertext> outs(16);
    timed(P.name, "encrypt", reps, [&] { encryptor.encrypt(px, out); });
    timed(P.name, "add", reps, [&] { ev.add(cx, cy, out); });
    timed(P.name, "mul_relin_rescale", reps, [&] {
        ev.multiply(cx, cy, out);
        ev.relinearize_inplace(out, rlk);
        ev.rescale_to_next_inplace(out);
    });
    timed(P.name, "rotate1", reps, [&] { ev.rotate_vector(cx, 1, gk, out); });
    timed(P.name, "rotate_x16", reps, [&] {
        for (int s = 1; s <= 16; s++)
            ev.rotate_vector(cx, P.name == "seal_2020" ? s : 1, gk, outs[s - 1]);
    });

    // Naive diagonal method; `mul` multiplies the rotated x by entry i.
    auto naive = [&](const function<void(const Ciphertext &, size_t, Ciphertext &)> &mul,
                     bool relin, Ciphertext &acc) {
        Ciphertext xr = cv, t;
        mul(xr, 0, acc);
        for (size_t i = 1; i < D; i++) {
            ev.rotate_vector_inplace(xr, 1, gk);
            mul(xr, i, t);
            ev.add_inplace(acc, t);
        }
        if (relin) ev.relinearize_inplace(acc, rlk);
    };
    // BSGS with Horner giant steps: acc = sum_k1 rot(inner_k1, BABY k1).
    auto bsgs = [&](const function<void(const Ciphertext &, size_t, Ciphertext &)> &mul,
                    bool relin, Ciphertext &acc) {
        vector<Ciphertext> xs(BABY);
        xs[0] = cv;
        for (int k = 1; k < BABY; k++) ev.rotate_vector(xs[k - 1], 1, gk, xs[k]);
        vector<Ciphertext> inner(GIANT);
        Ciphertext t;
        for (int k1 = 0; k1 < GIANT; k1++) {
            mul(xs[0], k1 * BABY, inner[k1]);
            for (int k2 = 1; k2 < BABY; k2++) {
                mul(xs[k2], k1 * BABY + k2, t);
                ev.add_inplace(inner[k1], t);
            }
            if (relin) ev.relinearize_inplace(inner[k1], rlk);
        }
        acc = inner[GIANT - 1];
        for (int k1 = GIANT - 2; k1 >= 0; k1--) {
            ev.rotate_vector_inplace(acc, BABY, gk);
            ev.add_inplace(acc, inner[k1]);
        }
    };
    auto mp_naive = [&](const Ciphertext &a, size_t i, Ciphertext &r) { ev.multiply_plain(a, pd[i], r); };
    auto mp_pre = [&](const Ciphertext &a, size_t i, Ciphertext &r) { ev.multiply_plain(a, ppre[i], r); };
    auto mc_naive = [&](const Ciphertext &a, size_t i, Ciphertext &r) { ev.multiply(a, cd[i], r); };
    auto mc_pre = [&](const Ciphertext &a, size_t i, Ciphertext &r) { ev.multiply(a, cpre[i], r); };

    timed(P.name, "matvec_plain_naive", reps, [&] { naive(mp_naive, false, out); });
    check("matvec_plain_naive", out);
    timed(P.name, "matvec_plain_bsgs", reps, [&] { bsgs(mp_pre, false, out); });
    check("matvec_plain_bsgs", out);
    timed(P.name, "matvec_enc_naive", reps, [&] { naive(mc_naive, true, out); });
    check("matvec_enc_naive", out);
    timed(P.name, "matvec_enc_bsgs", reps, [&] { bsgs(mc_pre, true, out); });
    check("matvec_enc_bsgs", out);
    return 0;
}
