// Spike (Task 4): reproduce one training iteration of Mihara et al. 2020
// (arXiv:2012.13552) on Microsoft SEAL 4.1 (CKKS).
//
// Network: d0-d1-d2 MLP (Iris: 4-10-3) with biases, square activation after
// each dense layer, MSE loss (mean over outputs), SGD.
// Packing: generalized diagonals of each weight matrix (paper Alg. 2), with
// optional pre-rotated "stepped" diagonals (paper Fig. 6 / Alg. 5).
// Transposition of the packed weight matrix uses rotations + additions only
// (paper Alg. 4 for plain diagonals, Alg. 5 for stepped diagonals).
//
// One generic implementation of the encrypted iteration (Trainer<BE>) runs on
//   * PlainBE : a slot-exact plaintext simulator (std::vector<double>, cyclic
//               rotations over all N/2 slots) -> validates the packing logic;
//   * SealBE  : SEAL CKKS ciphertexts, with per-phase op counters and timers.
// Both are compared against a dense plaintext reference (sgd_step).
#include "seal/seal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;  // row-major: Mat[row][col]

// ----------------------------------------------------------------------------
// Deterministic RNG (mt19937_64 output is fully specified by the standard; we
// avoid std::normal_distribution / std::shuffle, which are implementation-
// defined).
// ----------------------------------------------------------------------------
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    double uni() { return double(g() >> 11) * 0x1.0p-53; }
    double normal() {
        double u1 = uni(), u2 = uni();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
    size_t below(size_t n) { return size_t(g() % n); }
    void shuffle(std::vector<int> &v) {
        for (size_t i = v.size() - 1; i > 0; --i) std::swap(v[i], v[below(i + 1)]);
    }
};

// ----------------------------------------------------------------------------
// Data
// ----------------------------------------------------------------------------
struct Dataset {
    std::vector<Vec> X;
    std::vector<int> y;
    int classes = 3;
};

static Dataset load_iris(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path + " (run from spikes/seal-2020 or pass --data)");
    Dataset d;
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        Vec x;
        for (int k = 0; k < 4; ++k) {
            std::getline(ss, tok, ',');
            x.push_back(std::stod(tok));
        }
        std::getline(ss, tok, ',');
        d.X.push_back(x);
        d.y.push_back(std::stoi(tok));
    }
    return d;
}

// Synthetic data (used for the paper's Table-2 network 6-3-1).
static Dataset synthetic(size_t n, size_t din, int classes, uint64_t seed) {
    Rng r(seed);
    Dataset d;
    d.classes = classes;
    for (size_t i = 0; i < n; ++i) {
        Vec x(din);
        for (auto &v : x) v = r.normal();
        d.X.push_back(x);
        d.y.push_back(int(r.below(size_t(classes))));
    }
    return d;
}

struct Split {
    std::vector<int> train, test;
};

// 80:20 split (paper Table 3) from one seeded permutation.
static Split split_80_20(size_t n, Rng &r) {
    std::vector<int> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = int(i);
    r.shuffle(p);
    size_t ntr = (n * 8) / 10;
    Split s;
    s.train.assign(p.begin(), p.begin() + long(ntr));
    s.test.assign(p.begin() + long(ntr), p.end());
    return s;
}

// z-score with training-set statistics (population std).
static void standardize(Dataset &d, const std::vector<int> &train) {
    size_t m = d.X[0].size();
    for (size_t k = 0; k < m; ++k) {
        double mu = 0, var = 0;
        for (int i : train) mu += d.X[size_t(i)][k];
        mu /= double(train.size());
        for (int i : train) var += (d.X[size_t(i)][k] - mu) * (d.X[size_t(i)][k] - mu);
        double sd = std::sqrt(var / double(train.size()));
        for (auto &x : d.X) x[k] = (x[k] - mu) / sd;
    }
}

static Vec onehot(int c, int classes) {
    Vec y(size_t(classes), 0.0);
    y[size_t(c)] = 1.0;
    return y;
}

// ----------------------------------------------------------------------------
// Dense plaintext reference
// ----------------------------------------------------------------------------
struct Dims {
    size_t d0 = 4, d1 = 10, d2 = 3;
};

struct Model {
    Mat W1, W2;  // W1: d1 x d0, W2: d2 x d1 (out x in)
    Vec b1, b2;
};

static Model init_model(const Dims &d, double sigma, Rng &r) {
    Model m;
    m.W1.assign(d.d1, Vec(d.d0));
    m.W2.assign(d.d2, Vec(d.d1));
    m.b1.assign(d.d1, 0);
    m.b2.assign(d.d2, 0);
    for (auto &row : m.W1)
        for (auto &v : row) v = sigma * r.normal();
    for (auto &row : m.W2)
        for (auto &v : row) v = sigma * r.normal();
    for (auto &v : m.b1) v = sigma * r.normal();
    for (auto &v : m.b2) v = sigma * r.normal();
    return m;
}

static Vec matvec(const Mat &W, const Vec &x) {
    Vec y(W.size(), 0.0);
    for (size_t i = 0; i < W.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j) y[i] += W[i][j] * x[j];
    return y;
}

struct Fwd {
    Vec z1, a1, z2, a2;
};

static Fwd forward(const Model &m, const Vec &x) {
    Fwd f;
    f.z1 = matvec(m.W1, x);
    for (size_t i = 0; i < f.z1.size(); ++i) f.z1[i] += m.b1[i];
    f.a1 = f.z1;
    for (auto &v : f.a1) v *= v;
    f.z2 = matvec(m.W2, f.a1);
    for (size_t i = 0; i < f.z2.size(); ++i) f.z2[i] += m.b2[i];
    f.a2 = f.z2;
    for (auto &v : f.a2) v *= v;
    return f;
}

// Loss per sample: L = (1/d2) * sum_k (a2_k - y_k)^2.
// One SGD step on a mini-batch (gradient averaged over the batch).
static void sgd_step(Model &m, const std::vector<const Vec *> &xs, const std::vector<Vec> &ys, double lr) {
    size_t d0 = m.W1[0].size(), d1 = m.W1.size(), d2 = m.W2.size();
    Mat gW1(d1, Vec(d0, 0)), gW2(d2, Vec(d1, 0));
    Vec gb1(d1, 0), gb2(d2, 0);
    double inv = 1.0 / double(xs.size());
    for (size_t s = 0; s < xs.size(); ++s) {
        const Vec &x = *xs[s];
        Fwd f = forward(m, x);
        Vec delta2(d2), delta1(d1, 0);
        for (size_t k = 0; k < d2; ++k) delta2[k] = (2.0 / double(d2)) * (f.a2[k] - ys[s][k]) * 2.0 * f.z2[k];
        for (size_t j = 0; j < d1; ++j) {
            for (size_t k = 0; k < d2; ++k) delta1[j] += m.W2[k][j] * delta2[k];
            delta1[j] *= 2.0 * f.z1[j];
        }
        for (size_t k = 0; k < d2; ++k) {
            gb2[k] += inv * delta2[k];
            for (size_t j = 0; j < d1; ++j) gW2[k][j] += inv * delta2[k] * f.a1[j];
        }
        for (size_t j = 0; j < d1; ++j) {
            gb1[j] += inv * delta1[j];
            for (size_t i = 0; i < d0; ++i) gW1[j][i] += inv * delta1[j] * x[i];
        }
    }
    for (size_t k = 0; k < d2; ++k) {
        m.b2[k] -= lr * gb2[k];
        for (size_t j = 0; j < d1; ++j) m.W2[k][j] -= lr * gW2[k][j];
    }
    for (size_t j = 0; j < d1; ++j) {
        m.b1[j] -= lr * gb1[j];
        for (size_t i = 0; i < d0; ++i) m.W1[j][i] -= lr * gW1[j][i];
    }
}

static double max_abs_diff(const Model &a, const Model &b) {
    double e = 0;
    auto upd = [&](const Mat &x, const Mat &y) {
        for (size_t i = 0; i < x.size(); ++i)
            for (size_t j = 0; j < x[i].size(); ++j) e = std::max(e, std::fabs(x[i][j] - y[i][j]));
    };
    upd(a.W1, b.W1);
    upd(a.W2, b.W2);
    for (size_t i = 0; i < a.b1.size(); ++i) e = std::max(e, std::fabs(a.b1[i] - b.b1[i]));
    for (size_t i = 0; i < a.b2.size(); ++i) e = std::max(e, std::fabs(a.b2[i] - b.b2[i]));
    return e;
}

static double max_abs_update(const Model &a, const Model &b) { return max_abs_diff(a, b); }

// ----------------------------------------------------------------------------
// Packing (client side). S = number of CKKS slots (N/2).
// Generalized diagonal i (0 <= i < in) of W (out x in):
//   C_i[j] = W[j][(j+i) mod in],  j = 0..out-1, placed at slot j
//   (stepped / pre-rotated: C'_i = rot(C_i, i), i.e. placed at slot i+j).
// ----------------------------------------------------------------------------
static Vec pack_diag(const Mat &W, size_t i, bool stepped, size_t S) {
    size_t out = W.size(), in = W[0].size();
    Vec v(S, 0.0);
    for (size_t j = 0; j < out; ++j) v[stepped ? i + j : j] = W[j][(j + i) % in];
    return v;
}

static Mat unpack_diags(const std::vector<Vec> &diags, size_t out, size_t in, bool stepped) {
    Mat W(out, Vec(in, 0.0));
    for (size_t i = 0; i < in; ++i)
        for (size_t j = 0; j < out; ++j) W[j][(j + i) % in] = diags[i][stepped ? i + j : j];
    return W;
}

// x replicated with period len(x) over slots [0, len).
static Vec pack_rep(const Vec &x, size_t len, size_t S) {
    Vec v(S, 0.0);
    for (size_t k = 0; k < len; ++k) v[k] = x[k % x.size()];
    return v;
}

static Vec pack_vec(const Vec &x, size_t S) {
    Vec v(S, 0.0);
    std::copy(x.begin(), x.end(), v.begin());
    return v;
}

// ----------------------------------------------------------------------------
// Op counters / phases
// ----------------------------------------------------------------------------
enum Phase { FF = 0, TR = 1, BP = 2, UPD = 3, NPHASE = 4 };
static const char *phase_name[NPHASE] = {"feedforward", "transition", "backprop", "sgd_update"};

struct Counts {
    long mul_cc = 0, mul_cp = 0, relin = 0, rot = 0, rescale = 0, modsw = 0, add = 0;
    Counts &operator+=(const Counts &o) {
        mul_cc += o.mul_cc; mul_cp += o.mul_cp; relin += o.relin; rot += o.rot;
        rescale += o.rescale; modsw += o.modsw; add += o.add;
        return *this;
    }
};

// ----------------------------------------------------------------------------
// Backend 1: slot-exact plaintext simulator.
// ----------------------------------------------------------------------------
struct PlainBE {
    using Ct = Vec;
    size_t S;
    std::set<int> seal_steps;  // SEAL rotate_vector steps needed (left-positive)
    std::array<Counts, NPHASE> cnt{};
    Phase ph = FF;
    explicit PlainBE(size_t slots) : S(slots) {}
    void phase(Phase p) { ph = p; }
    Ct mul(const Ct &a, const Ct &b) {
        cnt[ph].mul_cc++; cnt[ph].relin++; cnt[ph].rescale++;
        Ct r(S);
        for (size_t i = 0; i < S; ++i) r[i] = a[i] * b[i];
        return r;
    }
    Ct mul_scalar(const Ct &a, double c) {
        cnt[ph].mul_cp++; cnt[ph].rescale++;
        Ct r(a);
        for (auto &v : r) v *= c;
        return r;
    }
    // Paper convention: rot(x, k) rotates RIGHT by k slots (k may be negative).
    Ct rot(const Ct &a, long k) {
        if (k == 0) return a;
        cnt[ph].rot++;
        seal_steps.insert(int(-k));
        Ct r(S);
        long s = long(S);
        for (long i = 0; i < s; ++i) r[size_t(((i + k) % s + s) % s)] = a[size_t(i)];
        return r;
    }
    Ct add(const Ct &a, const Ct &b) {
        cnt[ph].add++;
        Ct r(S);
        for (size_t i = 0; i < S; ++i) r[i] = a[i] + b[i];
        return r;
    }
    Ct sub(const Ct &a, const Ct &b) {
        cnt[ph].add++;
        Ct r(S);
        for (size_t i = 0; i < S; ++i) r[i] = a[i] - b[i];
        return r;
    }
    Ct lower_to(const Ct &a, const Ct &) { return a; }
    Ct encrypt(const Vec &v) { return v; }
    Vec decrypt(const Ct &c) { return c; }
};

// ----------------------------------------------------------------------------
// Backend 2: SEAL CKKS.
// ----------------------------------------------------------------------------
struct SealBE {
    using Ct = seal::Ciphertext;
    using clock = std::chrono::steady_clock;
    std::shared_ptr<seal::SEALContext> ctx;
    std::unique_ptr<seal::KeyGenerator> keygen;
    seal::SecretKey sk;
    seal::PublicKey pk;
    seal::RelinKeys rlk;
    seal::GaloisKeys gk;
    std::unique_ptr<seal::CKKSEncoder> enc;
    std::unique_ptr<seal::Encryptor> encryptor;
    std::unique_ptr<seal::Decryptor> decryptor;
    std::unique_ptr<seal::Evaluator> ev;
    double scale;
    size_t S;
    std::array<Counts, NPHASE> cnt{};
    std::array<double, NPHASE> sec{};
    Phase ph = FF;
    clock::time_point t_ph;
    bool timing = false;
    double max_scale_mismatch = 0;  // max |s_a/s_b - 1| forced on add/sub
    double keygen_sec = 0;

    SealBE(size_t logN, const std::vector<int> &bits, int log_scale, const std::set<int> &steps, bool pow2_keys) {
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        size_t N = size_t(1) << logN;
        parms.set_poly_modulus_degree(N);
        parms.set_coeff_modulus(seal::CoeffModulus::Create(N, bits));
        ctx = std::make_shared<seal::SEALContext>(parms, true, seal::sec_level_type::tc128);
        if (!ctx->parameters_set()) throw std::runtime_error(std::string("SEAL params: ") + ctx->parameter_error_message());
        scale = std::pow(2.0, log_scale);
        auto t0 = clock::now();
        keygen = std::make_unique<seal::KeyGenerator>(*ctx);
        sk = keygen->secret_key();
        keygen->create_public_key(pk);
        keygen->create_relin_keys(rlk);
        if (pow2_keys) {
            keygen->create_galois_keys(gk);  // SEAL default: steps +-2^k
        } else {
            std::vector<int> st(steps.begin(), steps.end());
            keygen->create_galois_keys(st, gk);
        }
        keygen_sec = std::chrono::duration<double>(clock::now() - t0).count();
        enc = std::make_unique<seal::CKKSEncoder>(*ctx);
        encryptor = std::make_unique<seal::Encryptor>(*ctx, pk);
        decryptor = std::make_unique<seal::Decryptor>(*ctx, sk);
        ev = std::make_unique<seal::Evaluator>(*ctx);
        S = enc->slot_count();
    }
    size_t level(const Ct &a) const { return ctx->get_context_data(a.parms_id())->chain_index(); }

    void reset() {
        cnt = {};
        sec = {};
    }
    void phase(Phase p) {
        auto now = clock::now();
        if (timing) sec[ph] += std::chrono::duration<double>(now - t_ph).count();
        ph = p;
        t_ph = now;
    }
    void start() {
        timing = true;
        ph = FF;
        t_ph = clock::now();
    }
    void stop() {
        phase(ph);
        timing = false;
    }

    // Bring a and b to the same level (mod-switch the higher one).
    void align(Ct &a, Ct &b) {
        while (level(a) > level(b)) { ev->mod_switch_to_next_inplace(a); cnt[ph].modsw++; }
        while (level(b) > level(a)) { ev->mod_switch_to_next_inplace(b); cnt[ph].modsw++; }
    }
    Ct lower_to(const Ct &a, const Ct &ref) {
        Ct r = a;
        while (level(r) > level(ref)) { ev->mod_switch_to_next_inplace(r); cnt[ph].modsw++; }
        return r;
    }
    Ct mul(const Ct &a0, const Ct &b0) {
        Ct a = a0, b = b0, r;
        align(a, b);
        ev->multiply(a, b, r);
        cnt[ph].mul_cc++;
        ev->relinearize_inplace(r, rlk);
        cnt[ph].relin++;
        ev->rescale_to_next_inplace(r);
        cnt[ph].rescale++;
        return r;
    }
    // Scalar is encoded at scale = the prime about to be dropped, so the
    // ciphertext scale is unchanged after the rescale.
    Ct mul_scalar(const Ct &a, double c) {
        auto cd = ctx->get_context_data(a.parms_id());
        double q = double(cd->parms().coeff_modulus().back().value());
        seal::Plaintext pt;
        enc->encode(c, a.parms_id(), q, pt);
        Ct r;
        ev->multiply_plain(a, pt, r);
        cnt[ph].mul_cp++;
        ev->rescale_to_next_inplace(r);
        cnt[ph].rescale++;
        r.scale() = a.scale();  // exact up to the rounding of q to double
        return r;
    }
    Ct rot(const Ct &a, long k) {
        if (k == 0) return a;
        Ct r;
        ev->rotate_vector(a, int(-k), gk, r);  // SEAL: positive = left
        cnt[ph].rot++;
        return r;
    }
    void match_scale(Ct &a, Ct &b) {
        if (a.scale() != b.scale()) {
            double m = std::fabs(b.scale() / a.scale() - 1.0);
            if (m > 1e-4) throw std::runtime_error("scale mismatch too large");
            max_scale_mismatch = std::max(max_scale_mismatch, m);
            b.scale() = a.scale();
        }
    }
    Ct add(const Ct &a0, const Ct &b0) {
        Ct a = a0, b = b0, r;
        align(a, b);
        match_scale(a, b);
        ev->add(a, b, r);
        cnt[ph].add++;
        return r;
    }
    Ct sub(const Ct &a0, const Ct &b0) {
        Ct a = a0, b = b0, r;
        align(a, b);
        match_scale(a, b);
        ev->sub(a, b, r);
        cnt[ph].add++;
        return r;
    }
    Ct encrypt(const Vec &v) {
        seal::Plaintext pt;
        enc->encode(v, scale, pt);
        Ct c;
        encryptor->encrypt(pt, c);
        return c;
    }
    Vec decrypt(const Ct &c) {
        seal::Plaintext pt;
        decryptor->decrypt(c, pt);
        Vec v;
        enc->decode(pt, v);
        return v;
    }
};

// ----------------------------------------------------------------------------
// Encrypted model / generic iteration
// ----------------------------------------------------------------------------
template <class Ct>
struct EncModel {
    std::vector<Ct> C1, C2;  // diagonals of W1 (d0 of them), W2 (d1 of them)
    Ct b1, b2;
};

template <class BE>
EncModel<typename BE::Ct> encrypt_model(BE &be, const Model &m, bool stepped) {
    EncModel<typename BE::Ct> e;
    size_t S = be.S;
    for (size_t i = 0; i < m.W1[0].size(); ++i) e.C1.push_back(be.encrypt(pack_diag(m.W1, i, stepped, S)));
    for (size_t i = 0; i < m.W2[0].size(); ++i) e.C2.push_back(be.encrypt(pack_diag(m.W2, i, stepped, S)));
    e.b1 = be.encrypt(pack_vec(m.b1, S));
    e.b2 = be.encrypt(pack_vec(m.b2, S));
    return e;
}

template <class BE>
Model decrypt_model(BE &be, const EncModel<typename BE::Ct> &e, const Dims &d, bool stepped) {
    std::vector<Vec> c1, c2;
    for (auto &c : e.C1) c1.push_back(be.decrypt(c));
    for (auto &c : e.C2) c2.push_back(be.decrypt(c));
    Model m;
    m.W1 = unpack_diags(c1, d.d1, d.d0, stepped);
    m.W2 = unpack_diags(c2, d.d2, d.d1, stepped);
    Vec b1 = be.decrypt(e.b1), b2 = be.decrypt(e.b2);
    m.b1.assign(b1.begin(), b1.begin() + long(d.d1));
    m.b2.assign(b2.begin(), b2.begin() + long(d.d2));
    return m;
}

template <class BE>
struct Trainer {
    using Ct = typename BE::Ct;
    BE &be;
    Dims d;
    bool stepped;  // true: Alg.5 layout (pre-rotated diagonals); false: Alg.4 layout
    double lr;

    // Paper Alg. 2: P = sum_i w_i x rot(x, -i). With stepped diagonals the
    // rotation moves from the input to the product: P = sum_i rot(w'_i x x, -i).
    // In plain mode the rotated inputs are cached (reused by the gradient).
    Ct matvec(const std::vector<Ct> &C, const Ct &xrep, size_t in, bool step, std::vector<Ct> *cache) {
        Ct acc;
        for (size_t i = 0; i < in; ++i) {
            Ct p;
            if (step) {
                p = be.rot(be.mul(C[i], xrep), -long(i));
            } else {
                Ct xr = be.rot(xrep, -long(i));
                p = be.mul(C[i], xr);
                if (cache) cache->push_back(xr);
            }
            acc = (i == 0) ? p : be.add(acc, p);
        }
        return acc;
    }
    // Replicate a vector supported on [0, period) so that slots [0, need) hold
    // v[k mod period]; log-doubling with rotations to the right.
    Ct replicate(const Ct &v, size_t period, size_t need) {
        Ct cur = v;
        for (size_t cov = period; cov < need; cov *= 2) cur = be.add(cur, be.rot(cur, long(cov)));
        return cur;
    }
    // Paper Alg. 4 / Alg. 5: diagonals D_i (i < M) of W^T (N x M) from the
    // diagonals C_c (c < N) of W (M x N, in = N, out = M):
    //   D_i = sum_{s=0}^{e_i} rot(C_{(sM-i) mod N}, sM - i),  e_i = floor((N-1+i)/M)
    // (equals the paper's q / q+1 rule with N = qM + r).  For stepped C'_c =
    // rot(C_c, c) the rotation amount becomes (sM - i) - c, which is 0 for
    // s >= 1 and -N for s = 0, i > 0 (Alg. 5: one rotation per i > 0).
    std::vector<Ct> transpose(const std::vector<Ct> &C, size_t N, size_t M) {
        std::vector<Ct> D(M);
        for (size_t i = 0; i < M; ++i) {
            size_t smax = (N - 1 + i) / M;
            for (size_t s = 0; s <= smax; ++s) {
                long t = long(s * M) - long(i);
                size_t c = size_t(((t % long(N)) + long(N)) % long(N));
                long r = t - (stepped ? long(c) : 0);
                Ct term = be.rot(C[c], r);
                D[i] = (s == 0) ? term : be.add(D[i], term);
            }
        }
        return D;
    }
    // Weight gradient in the same packed layout as the weights:
    //   plain:   G_i  = delta x rot(x, -i)          (rotations reused from FF)
    //   stepped: G'_i = rot(delta, i) x x_rep
    std::vector<Ct> outer(const Ct &delta, const Ct &xrep, const std::vector<Ct> &cache, size_t in) {
        std::vector<Ct> G;
        for (size_t i = 0; i < in; ++i)
            G.push_back(stepped ? be.mul(be.rot(delta, long(i)), xrep) : be.mul(delta, cache[i]));
        return G;
    }

    void iteration(EncModel<Ct> &m, const Ct &xrep, const Ct &y) {
        std::vector<Ct> cx, ca;
        be.phase(FF);
        Ct z1 = be.add(matvec(m.C1, xrep, d.d0, stepped, &cx), m.b1);
        Ct a1 = be.mul(z1, z1);
        Ct a1r = replicate(a1, d.d1, d.d1 + d.d2 - 1);
        Ct z2 = be.add(matvec(m.C2, a1r, d.d1, stepped, &ca), m.b2);
        Ct a2 = be.mul(z2, z2);

        be.phase(BP);
        Ct e = be.sub(a2, y);
        Ct t = be.mul(e, z2);
        // delta2 = (2/d2)(a2-y) * 2 z2 ; fold lr in: d2s = lr*delta2
        Ct d2s = be.mul_scalar(t, lr * 4.0 / double(d.d2));
        Ct d2r = replicate(d2s, d.d2, d.d2 + d.d1 - 1);

        be.phase(TR);
        // W2^T is only needed at the level of d2r: mod-switch the W2 diagonals
        // down first so the transposition rotations run on fewer RNS limbs.
        std::vector<Ct> C2low;
        for (auto &c : m.C2) C2low.push_back(be.lower_to(c, d2r));
        std::vector<Ct> D = transpose(C2low, d.d1, d.d2);

        be.phase(BP);
        Ct u = matvec(D, d2r, d.d2, false, nullptr);  // W2^T * d2s (Alg. 2 on D)
        Ct v = be.mul(u, z1);                         // * z1 (kills garbage outside [0,d1))
        Ct d1s = be.add(v, v);                        // * 2  -> lr*delta1
        std::vector<Ct> g2 = outer(d2s, a1r, ca, d.d1);
        std::vector<Ct> g1 = outer(d1s, xrep, cx, d.d0);

        be.phase(UPD);
        for (size_t i = 0; i < g2.size(); ++i) m.C2[i] = be.sub(m.C2[i], g2[i]);
        for (size_t i = 0; i < g1.size(); ++i) m.C1[i] = be.sub(m.C1[i], g1[i]);
        m.b2 = be.sub(m.b2, d2s);
        m.b1 = be.sub(m.b1, d1s);
    }
};

// ----------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------
struct Opts {
    bool check = false, plain_train = false, pow2_keys = false;
    bool stepped = true;  // --transpose alg5 (default) | alg4
    int iters = 10, warmup = 1;
    Dims dims;
    bool synthetic = false;
    uint64_t seed = 2020;
    double lr = 0.1, sigma = 0.2;
    int epochs = 400, batch = 20, seeds = 10;
    std::string data = "data/iris.csv";
};

static Opts parse(int argc, char **argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--check") o.check = true;
        else if (a == "--plain-train") o.plain_train = true;
        else if (a == "--keys") o.pow2_keys = (next() == "pow2");
        else if (a == "--transpose") o.stepped = (next() != "alg4");
        else if (a == "--iters") o.iters = std::stoi(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--lr") o.lr = std::stod(next());
        else if (a == "--sigma") o.sigma = std::stod(next());
        else if (a == "--epochs") o.epochs = std::stoi(next());
        else if (a == "--batch") o.batch = std::stoi(next());
        else if (a == "--seeds") o.seeds = std::stoi(next());
        else if (a == "--data") o.data = next();
        else if (a == "--dims") {
            std::string s = next();
            size_t p1 = s.find(','), p2 = s.find(',', p1 + 1);
            o.dims.d0 = std::stoul(s.substr(0, p1));
            o.dims.d1 = std::stoul(s.substr(p1 + 1, p2 - p1 - 1));
            o.dims.d2 = std::stoul(s.substr(p2 + 1));
            o.synthetic = !(o.dims.d0 == 4 && o.dims.d2 == 3);
        } else if (a == "-h" || a == "--help") {
            std::cout << "usage: he_train [--check] [--plain-train] [--transpose alg5|alg4] [--keys exact|pow2]\n"
                         "                [--iters K] [--warmup W] [--dims d0,d1,d2] [--seed S] [--lr LR]\n"
                         "                [--sigma SIG] [--epochs E] [--batch B] [--seeds K] [--data PATH]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option " + a);
    }
    return o;
}

struct Prepared {
    Dataset ds;
    Split sp;
    Model m0;
};

static Prepared prepare(const Opts &o, uint64_t seed) {
    Prepared p;
    Rng r(seed);
    p.ds = o.synthetic ? synthetic(50, o.dims.d0, int(o.dims.d2), seed ^ 0x5eed) : load_iris(o.data);
    p.sp = split_80_20(p.ds.X.size(), r);
    standardize(p.ds, p.sp.train);
    p.m0 = init_model(o.dims, o.sigma, r);
    return p;
}

static double accuracy(const Model &m, const Dataset &ds, const std::vector<int> &idx, double *loss) {
    size_t ok = 0;
    double L = 0;
    for (int i : idx) {
        Fwd f = forward(m, ds.X[size_t(i)]);
        size_t am = size_t(std::max_element(f.a2.begin(), f.a2.end()) - f.a2.begin());
        ok += (int(am) == ds.y[size_t(i)]);
        Vec y = onehot(ds.y[size_t(i)], ds.classes);
        for (size_t k = 0; k < y.size(); ++k) L += (f.a2[k] - y[k]) * (f.a2[k] - y[k]) / double(y.size());
    }
    if (loss) *loss = L / double(idx.size());
    return double(ok) / double(idx.size());
}

static int plain_train(const Opts &o) {
    std::printf("# plaintext training: dims %zu-%zu-%zu, lr %.3g, batch %d, epochs %d, init N(0,%.3g^2), z-score inputs\n",
                o.dims.d0, o.dims.d1, o.dims.d2, o.lr, o.batch, o.epochs, o.sigma);
    std::printf("seed,status,train_acc,test_acc,train_loss,test_loss\n");
    double sa_tr = 0, sa_te = 0, sl_tr = 0;
    std::vector<double> te_accs;
    int ok = 0;
    for (int k = 0; k < o.seeds; ++k) {
        uint64_t seed = o.seed + uint64_t(k);
        Prepared p = prepare(o, seed);
        Rng r(seed * 7919 + 1);  // epoch shuffling
        Model m = p.m0;
        bool diverged = false;
        for (int ep = 0; ep < o.epochs && !diverged; ++ep) {
            std::vector<int> perm = p.sp.train;
            r.shuffle(perm);
            for (size_t b = 0; b < perm.size(); b += size_t(o.batch)) {
                std::vector<const Vec *> xs;
                std::vector<Vec> ys;
                for (size_t j = b; j < std::min(perm.size(), b + size_t(o.batch)); ++j) {
                    xs.push_back(&p.ds.X[size_t(perm[j])]);
                    ys.push_back(onehot(p.ds.y[size_t(perm[j])], p.ds.classes));
                }
                sgd_step(m, xs, ys, o.lr);
            }
            double l;
            accuracy(m, p.ds, p.sp.train, &l);
            if (!std::isfinite(l) || l > 1e6) diverged = true;
        }
        if (diverged) {
            std::printf("%llu,diverged,,,,\n", (unsigned long long)seed);
            continue;
        }
        double ltr, lte;
        double atr = accuracy(m, p.ds, p.sp.train, &ltr), ate = accuracy(m, p.ds, p.sp.test, &lte);
        std::printf("%llu,ok,%.4f,%.4f,%.4f,%.4f\n", (unsigned long long)seed, atr, ate, ltr, lte);
        sa_tr += atr; sa_te += ate; sl_tr += ltr; te_accs.push_back(ate); ok++;
    }
    if (ok) {
        double mu = sa_te / ok, var = 0;
        for (double a : te_accs) var += (a - mu) * (a - mu);
        std::printf("# summary: %d/%d converged; mean train acc %.4f, mean test acc %.4f (sd %.4f), mean train loss %.4f\n",
                    ok, o.seeds, sa_tr / ok, mu, ok > 1 ? std::sqrt(var / (ok - 1)) : 0.0, sl_tr / ok);
    } else {
        std::printf("# summary: 0/%d converged\n", o.seeds);
    }
    return 0;
}

static void print_counts(const char *title, const std::array<Counts, NPHASE> &c) {
    std::printf("%s\n  %-12s %7s %7s %7s %5s %8s %6s %5s\n", title, "phase", "mul_cc", "mul_cp", "relin", "rot",
                "rescale", "modsw", "add");
    Counts tot;
    for (int p = 0; p < NPHASE; ++p) {
        std::printf("  %-12s %7ld %7ld %7ld %5ld %8ld %6ld %5ld\n", phase_name[p], c[size_t(p)].mul_cc,
                    c[size_t(p)].mul_cp, c[size_t(p)].relin, c[size_t(p)].rot, c[size_t(p)].rescale,
                    c[size_t(p)].modsw, c[size_t(p)].add);
        tot += c[size_t(p)];
    }
    std::printf("  %-12s %7ld %7ld %7ld %5ld %8ld %6ld %5ld\n", "total", tot.mul_cc, tot.mul_cp, tot.relin, tot.rot,
                tot.rescale, tot.modsw, tot.add);
}

static void mean_sd(const std::vector<double> &v, double &mu, double &sd) {
    mu = 0;
    for (double x : v) mu += x;
    mu /= double(v.size());
    double var = 0;
    for (double x : v) var += (x - mu) * (x - mu);
    sd = v.size() > 1 ? std::sqrt(var / double(v.size() - 1)) : 0.0;
}

static int run_he(const Opts &o) {
    const size_t logN = 15;
    const std::vector<int> bits = {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60};
    const int log_scale = 40;
    const size_t S = (size_t(1) << logN) / 2;
    const Dims &d = o.dims;
    Prepared p = prepare(o, o.seed);
    Rng rs(o.seed * 7919 + 1);
    std::vector<int> order = p.sp.train;
    rs.shuffle(order);
    int n_it = o.check ? 1 : o.warmup + o.iters;
    std::printf("# he_train: dims %zu-%zu-%zu, %s, layout %s, keys %s, N=2^%zu, bits {60,40x9,60}, scale 2^%d, lr %.3g, seed %llu\n",
                d.d0, d.d1, d.d2, o.synthetic ? "synthetic data" : "Iris", o.stepped ? "stepped/Alg5" : "plain-diag/Alg4",
                o.pow2_keys ? "pow2 (SEAL default)" : "exact steps", logN, log_scale, o.lr, (unsigned long long)o.seed);
    std::printf("# samples (train-order idx): ");
    for (int k = 0; k < n_it; ++k) std::printf("%d ", order[size_t(k)]);
    std::printf("\n");

    auto inputs = [&](int k, Vec &xrep, Vec &y) {
        const Vec &x = p.ds.X[size_t(order[size_t(k)])];
        xrep = pack_rep(x, d.d0 + d.d1 - 1, S);
        y = pack_vec(onehot(p.ds.y[size_t(order[size_t(k)])], p.ds.classes), S);
    };

    // 1) Slot-exact plaintext simulation of the packed algorithm vs dense reference.
    std::set<int> steps;
    {
        PlainBE pb(S);
        Trainer<PlainBE> tr{pb, d, o.stepped, o.lr};
        auto em = encrypt_model(pb, p.m0, o.stepped);
        Model ref = p.m0;
        double worst = 0;
        for (int k = 0; k < n_it; ++k) {
            Vec xrep, y;
            inputs(k, xrep, y);
            tr.iteration(em, xrep, y);
            const Vec &x = p.ds.X[size_t(order[size_t(k)])];
            sgd_step(ref, {&x}, {onehot(p.ds.y[size_t(order[size_t(k)])], p.ds.classes)}, o.lr);
            Model got = decrypt_model(pb, em, d, o.stepped);
            worst = std::max(worst, max_abs_diff(got, ref));
            // re-pack (client) to drop any off-support residue, as in the HE path
            em = encrypt_model(pb, got, o.stepped);
        }
        steps = pb.seal_steps;
        std::printf("[plain-sim] packed algorithm vs dense reference, %d iteration(s): max |err| = %.3e\n", n_it, worst);
        if (worst > 1e-9) {
            std::printf("FAIL: packing logic mismatch\n");
            return 1;
        }
        std::printf("[plain-sim] SEAL rotation steps needed (%zu):", steps.size());
        for (int s : steps) std::printf(" %d", s);
        std::printf("\n");
    }

    // 2) SEAL.
    SealBE be(logN, bits, log_scale, steps, o.pow2_keys);
    std::printf("[seal] SEAL %d.%d.%d, slots %zu, keygen (sk,pk,rlk,%s galois) %.2f s\n", SEAL_VERSION_MAJOR,
                SEAL_VERSION_MINOR, SEAL_VERSION_PATCH, be.S, o.pow2_keys ? "pow2" : "exact", be.keygen_sec);
    Trainer<SealBE> tr{be, d, o.stepped, o.lr};
    auto em = encrypt_model(be, p.m0, o.stepped);
    std::printf("[seal] fresh ciphertext level (chain index) = %zu\n", be.level(em.C1[0]));
    Model ref = p.m0;
    std::vector<double> t_server, t_repack, t_enc_in;
    std::array<std::vector<double>, NPHASE> t_phase;
    std::array<Counts, NPHASE> counts_first{};
    double worst = 0, worst_update = 0;
    size_t final_level_W1 = 0, final_level_W2 = 0;
    using clock = std::chrono::steady_clock;
    for (int k = 0; k < n_it; ++k) {
        Vec xrep, y;
        inputs(k, xrep, y);
        auto t0 = clock::now();
        auto cx = be.encrypt(xrep);
        auto cy = be.encrypt(y);
        double te = std::chrono::duration<double>(clock::now() - t0).count();

        be.reset();
        auto t1 = clock::now();
        be.start();
        tr.iteration(em, cx, cy);
        be.stop();
        double ts = std::chrono::duration<double>(clock::now() - t1).count();
        if (k == 0) counts_first = be.cnt;
        final_level_W1 = be.level(em.C1[0]);
        final_level_W2 = be.level(em.C2[0]);

        // reference
        Model before = ref;
        const Vec &x = p.ds.X[size_t(order[size_t(k)])];
        sgd_step(ref, {&x}, {onehot(p.ds.y[size_t(order[size_t(k)])], p.ds.classes)}, o.lr);

        // client: repacking = decrypt + re-encode + re-encrypt the weights
        auto t2 = clock::now();
        Model got = decrypt_model(be, em, d, o.stepped);
        em = encrypt_model(be, got, o.stepped);
        double tp = std::chrono::duration<double>(clock::now() - t2).count();

        double err = max_abs_diff(got, ref);
        worst = std::max(worst, err);
        worst_update = std::max(worst_update, max_abs_update(before, ref));
        std::printf("[seal] iter %d: server %.3f s (ff %.3f, tr %.3f, bp %.3f, upd %.3f) | repack %.3f s | enc x,y %.3f s | max|W_he - W_ref| = %.3e\n",
                    k, ts, be.sec[FF], be.sec[TR], be.sec[BP], be.sec[UPD], tp, te, err);
        if (o.check || k >= o.warmup) {
            t_server.push_back(ts);
            t_repack.push_back(tp);
            t_enc_in.push_back(te);
            for (int q = 0; q < NPHASE; ++q) t_phase[size_t(q)].push_back(be.sec[size_t(q)]);
        }
        // The reference is trained on its own (it is NOT reset to the decrypted HE model),
        // so `worst` is the drift compounded over all chained iterations.
    }
    std::printf("[seal] updated-weight ciphertext levels after one iteration: W1 %zu, W2 %zu (0 = last prime)\n",
                final_level_W1, final_level_W2);
    std::printf("[seal] max forced scale mismatch on add/sub: %.3e (relative)\n", be.max_scale_mismatch);
    print_counts("[counts] one iteration (server side), per phase:", counts_first);
    double mu, sd;
    mean_sd(t_server, mu, sd);
    std::printf("[timing] server iteration over %zu iters: mean %.3f s, sd %.3f s\n", t_server.size(), mu, sd);
    for (int q = 0; q < NPHASE; ++q) {
        double a, b;
        mean_sd(t_phase[size_t(q)], a, b);
        std::printf("[timing]   %-12s mean %.3f s, sd %.3f s\n", phase_name[q], a, b);
    }
    double rmu, rsd, emu, esd;
    mean_sd(t_repack, rmu, rsd);
    mean_sd(t_enc_in, emu, esd);
    std::printf("[timing] repacking (client decrypt+re-encrypt of %zu weight cts): mean %.3f s, sd %.3f s\n",
                d.d0 + d.d1 + 2, rmu, rsd);
    std::printf("[timing] client encrypt x,y: mean %.3f s\n", emu);
    std::printf("[timing] server + repacking: mean %.3f s\n", mu + rmu);
    std::printf("[check] max |W_he - W_ref| over %d iteration(s): %.3e (largest single-step update magnitude %.3e)\n",
                n_it, worst, worst_update);
    if (o.check) {
        bool pass = worst < 1e-3;
        std::printf("%s: encrypted-vs-plaintext weight error %.3e %s 1e-3\n", pass ? "PASS" : "FAIL", worst,
                    pass ? "<" : ">=");
        return pass ? 0 : 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    try {
        Opts o = parse(argc, argv);
        if (o.plain_train) return plain_train(o);
        return run_he(o);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
