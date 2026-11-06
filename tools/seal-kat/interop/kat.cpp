// Known-answer interop between ckks-train-ckks and Microsoft SEAL 4.1.2 (MIT), seal_2020 preset.
//
//   seal_interop_kat generate <dir>   SEAL keys + ciphertexts + decryptions -> <dir>
//   seal_interop_kat verify <dir>     SEAL decrypts hekit's <dir>/hekit_*.bin; exit 0 iff
//                                     every object loads and decodes within its bound
//
// Parameters: CKKS, N = 2^15, CoeffModulus::Create(N, {60, 40 x 9, 60}) (the last prime is
// the special prime), sec_level_type::tc128, scale 2^40. Everything is saved with
// compr_mode_type::none. See README.md for the file list and the error bounds.
#include "seal/seal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using namespace seal;
using namespace std;

static const size_t N = 32768;
static const size_t SLOTS = N / 2;
static const double SCALE = 1099511627776.0; // 2^40

// Error bounds (log2), see README.md "Bounds".
static const double FRESH_BOUND = -25.0;    // symmetric encryption, add, tensor product
static const double ROUNDING_BOUND = -23.0; // rescale / public-key encryption (rounding)

static SEALContext make_context() {
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    vector<int> bits{60};
    for (int i = 0; i < 9; i++) bits.push_back(40);
    bits.push_back(60);
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
    return SEALContext(parms, true, sec_level_type::tc128);
}

template <class T> static void save(const T &obj, const string &path) {
    ofstream f(path, ios::binary);
    if (!f) throw runtime_error("cannot open " + path);
    obj.save(f, compr_mode_type::none);
}

static void write_array(FILE *f, const char *name, const vector<double> &v, bool comma = true) {
    fprintf(f, "  \"%s\": [", name);
    for (size_t i = 0; i < v.size(); i++) fprintf(f, "%.17g%s", v[i], i + 1 < v.size() ? ", " : "");
    fprintf(f, "]%s\n", comma ? "," : "");
}

static void write_parms_id(FILE *f, const parms_id_type &id) {
    fprintf(f, "[%llu, %llu, %llu, %llu]", (unsigned long long)id[0], (unsigned long long)id[1],
            (unsigned long long)id[2], (unsigned long long)id[3]);
}

static double max_err(const vector<double> &a, const vector<double> &b) {
    double m = 0;
    for (size_t i = 0; i < SLOTS; i++) m = max(m, fabs(a[i] - b[i]));
    return m;
}

static void inputs(vector<double> &x, vector<double> &y) {
    // Uniform in [-1, 1), fixed seed (values are also written to values.json / inputs.f64).
    mt19937_64 g(20201225);
    uniform_real_distribution<double> d(-1.0, 1.0);
    x.resize(SLOTS);
    y.resize(SLOTS);
    for (size_t i = 0; i < SLOTS; i++) x[i] = d(g);
    for (size_t i = 0; i < SLOTS; i++) y[i] = d(g);
}

struct Expected {
    vector<double> x, y, x_plus_y, x_times_y, x_rot1;
};

static Expected expected() {
    Expected e;
    inputs(e.x, e.y);
    e.x_plus_y.resize(SLOTS);
    e.x_times_y.resize(SLOTS);
    e.x_rot1.resize(SLOTS);
    for (size_t i = 0; i < SLOTS; i++) {
        e.x_plus_y[i] = e.x[i] + e.y[i];
        e.x_times_y[i] = e.x[i] * e.y[i];
        e.x_rot1[i] = e.x[(i + 1) % SLOTS];
    }
    return e;
}

static int generate(const string &dir) {
    SEALContext ctx = make_context();
    Expected ex = expected();
    KeyGenerator keygen(ctx);
    const SecretKey &sk = keygen.secret_key();
    PublicKey pk;
    keygen.create_public_key(pk);
    RelinKeys rlk;
    keygen.create_relin_keys(rlk);
    GaloisKeys gk;
    keygen.create_galois_keys(vector<int>{1}, gk);

    CKKSEncoder enc(ctx);
    Encryptor sym(ctx, sk);
    Encryptor asym(ctx, pk);
    Decryptor dec(ctx, sk);
    Evaluator ev(ctx);

    Plaintext px, py;
    enc.encode(ex.x, ctx.first_parms_id(), SCALE, px);
    enc.encode(ex.y, ctx.first_parms_id(), SCALE, py);

    Ciphertext ct_x, ct_y, ct_x_pk;
    sym.encrypt_symmetric(px, ct_x);
    sym.encrypt_symmetric(py, ct_y);
    asym.encrypt(px, ct_x_pk);
    Ciphertext ct_x_plus_y;
    ev.add(ct_x, ct_y, ct_x_plus_y);
    Ciphertext ct_x_times_y_3;
    ev.multiply(ct_x, ct_y, ct_x_times_y_3); // 3 components, scale 2^80
    Ciphertext ct_x_times_y;
    ev.relinearize(ct_x_times_y_3, rlk, ct_x_times_y); // 2 components, scale 2^80, level 9
    Ciphertext ct_x_times_y_rescaled;
    ev.rescale_to_next(ct_x_times_y, ct_x_times_y_rescaled); // level 8, scale 2^80 / q_9
    Ciphertext ct_rot1;
    ev.rotate_vector(ct_x, 1, gk, ct_rot1);

    save(sk, dir + "/secret_key.bin");
    save(pk, dir + "/public_key.bin");
    {
        ofstream f(dir + "/inputs.f64", ios::binary);
        f.write(reinterpret_cast<const char *>(ex.x.data()), SLOTS * sizeof(double));
        f.write(reinterpret_cast<const char *>(ex.y.data()), SLOTS * sizeof(double));
    }

    struct Obj {
        const char *name;
        const Ciphertext *ct;
        const char *encryption;
        const char *expected;
        const vector<double> *want;
    };
    vector<Obj> objs{
        {"ct_x", &ct_x, "symmetric", "x", &ex.x},
        {"ct_y", &ct_y, "symmetric", "y", &ex.y},
        {"ct_x_pk", &ct_x_pk, "public", "x", &ex.x},
        {"ct_x_plus_y", &ct_x_plus_y, "add", "x_plus_y", &ex.x_plus_y},
        {"ct_x_times_y_3", &ct_x_times_y_3, "multiply", "x_times_y", &ex.x_times_y},
        {"ct_x_times_y", &ct_x_times_y, "multiply+relinearize", "x_times_y", &ex.x_times_y},
        {"ct_x_times_y_rescaled", &ct_x_times_y_rescaled, "multiply+relinearize+rescale_to_next",
         "x_times_y", &ex.x_times_y},
        {"ct_rot1", &ct_rot1, "rotate_vector(ct_x, 1)", "x_rot1", &ex.x_rot1},
    };

    FILE *f = fopen((dir + "/values.json").c_str(), "w");
    if (!f) throw runtime_error("cannot open values.json");
    fprintf(f, "{\n  \"seal_version\": \"%d.%d.%d\",\n", SEAL_VERSION_MAJOR, SEAL_VERSION_MINOR,
            SEAL_VERSION_PATCH);
    fprintf(f, "  \"poly_modulus_degree\": %zu,\n  \"scale\": %.17g,\n", N, SCALE);
    fprintf(f, "  \"coeff_modulus\": [");
    auto &cm = ctx.key_context_data()->parms().coeff_modulus();
    for (size_t i = 0; i < cm.size(); i++)
        fprintf(f, "%llu%s", (unsigned long long)cm[i].value(), i + 1 < cm.size() ? ", " : "");
    fprintf(f, "],\n  \"key_parms_id\": ");
    write_parms_id(f, ctx.key_parms_id());
    // data_parms_ids[l] = parms_id of the level with l + 1 primes.
    vector<parms_id_type> data_ids;
    for (auto cd = ctx.last_context_data(); cd; cd = cd->prev_context_data()) {
        if (cd->parms_id() == ctx.key_parms_id()) break;
        data_ids.push_back(cd->parms_id());
    }
    fprintf(f, ",\n  \"data_parms_ids\": [");
    for (size_t i = 0; i < data_ids.size(); i++) {
        write_parms_id(f, data_ids[i]);
        fprintf(f, "%s", i + 1 < data_ids.size() ? ", " : "");
    }
    fprintf(f, "],\n  \"objects\": {\n");
    for (size_t k = 0; k < objs.size(); k++) {
        const Obj &o = objs[k];
        save(*o.ct, dir + "/" + o.name + ".bin");
        Plaintext pt;
        dec.decrypt(*o.ct, pt);
        string pt_name = string("pt") + (o.name + 2); // ct_x -> pt_x
        save(pt, dir + "/" + pt_name + ".bin");
        vector<double> got;
        enc.decode(pt, got);
        double err = max_err(got, *o.want);
        size_t level = ctx.get_context_data(o.ct->parms_id())->chain_index();
        printf("SEAL %-22s level %zu size %zu scale 2^%.6f  SEAL decrypt-decode max error 2^%.2f\n", o.name,
               level, o.ct->size(), log2(o.ct->scale()), log2(err));
        fprintf(f, "    \"%s\": {\"file\": \"%s.bin\", \"plain\": \"%s.bin\", \"op\": \"%s\", \"expected\": \"%s\", ",
                o.name, o.name, pt_name.c_str(), o.encryption, o.expected);
        fprintf(f, "\"level\": %zu, \"size\": %zu, \"scale\": %.17g, \"seal_max_err\": %.17g,\n", level,
                o.ct->size(), o.ct->scale(), err);
        fprintf(f, "      \"seal_decoded\": [");
        for (size_t i = 0; i < SLOTS; i++) fprintf(f, "%.17g%s", got[i], i + 1 < SLOTS ? ", " : "");
        fprintf(f, "]}%s\n", k + 1 < objs.size() ? "," : "");
    }
    fprintf(f, "  },\n");
    write_array(f, "x", ex.x);
    write_array(f, "y", ex.y);
    write_array(f, "x_plus_y", ex.x_plus_y);
    write_array(f, "x_times_y", ex.x_times_y);
    write_array(f, "x_rot1", ex.x_rot1, false);
    fprintf(f, "}\n");
    fclose(f);
    printf("wrote %s\n", dir.c_str());
    return 0;
}

static int verify(const string &dir) {
    SEALContext ctx = make_context();
    Expected ex = expected();
    {
        // The Rust side encrypted the values of values.json; make sure they are these.
        vector<double> xy(2 * SLOTS);
        ifstream f(dir + "/inputs.f64", ios::binary);
        f.read(reinterpret_cast<char *>(xy.data()), 2 * SLOTS * sizeof(double));
        if (!f || !equal(ex.x.begin(), ex.x.end(), xy.begin()) || !equal(ex.y.begin(), ex.y.end(), xy.begin() + SLOTS)) {
            fprintf(stderr, "inputs.f64 does not match the generator's inputs\n");
            return 1;
        }
    }
    SecretKey sk;
    {
        ifstream f(dir + "/secret_key.bin", ios::binary);
        sk.load(ctx, f);
    }
    CKKSEncoder enc(ctx);
    Decryptor dec(ctx, sk);
    struct Obj {
        const char *name;
        const vector<double> *want;
        size_t level, size;
        double bound_log2;
    };
    vector<Obj> objs{
        {"hekit_ct_x_sym", &ex.x, 9, 2, FRESH_BOUND},
        {"hekit_ct_x_sym_l3", &ex.x, 3, 2, FRESH_BOUND},
        {"hekit_ct_x_pk", &ex.x, 9, 2, ROUNDING_BOUND},
        {"hekit_ct_x_plus_y", &ex.x_plus_y, 9, 2, FRESH_BOUND},
        {"hekit_ct_x_times_y_3", &ex.x_times_y, 9, 3, FRESH_BOUND},
        {"hekit_ct_x_times_y_rescaled", &ex.x_times_y, 8, 2, ROUNDING_BOUND},
    };
    int failures = 0;
    for (const Obj &o : objs) {
        string path = dir + "/" + o.name + ".bin";
        try {
            Ciphertext ct;
            {
                ifstream f(path, ios::binary);
                if (!f) throw runtime_error("missing");
                ct.load(ctx, f); // checks header, parms_id, metadata and that every residue < q_i
            }
            size_t level = ctx.get_context_data(ct.parms_id())->chain_index();
            if (level != o.level || ct.size() != o.size || !ct.is_ntt_form())
                throw runtime_error("unexpected level/size/form");
            Plaintext pt;
            dec.decrypt(ct, pt);
            vector<double> got;
            enc.decode(pt, got);
            double err = max_err(got, *o.want);
            bool ok = err < exp2(o.bound_log2);
            printf("SEAL decrypts %-28s level %zu size %zu: max error 2^%.2f (bound 2^%.0f) %s\n", o.name, level,
                   ct.size(), log2(err), o.bound_log2, ok ? "ok" : "FAIL");
            failures += !ok;
        } catch (const exception &e) {
            printf("SEAL decrypts %-28s: FAIL (%s)\n", o.name, e.what());
            failures++;
        }
    }
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc != 3 || (string(argv[1]) != "generate" && string(argv[1]) != "verify")) {
        fprintf(stderr, "usage: %s generate|verify <dir>\n", argv[0]);
        return 2;
    }
    try {
        return string(argv[1]) == "generate" ? generate(argv[2]) : verify(argv[2]);
    } catch (const exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
