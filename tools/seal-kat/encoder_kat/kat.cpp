// Known-answer vectors for ckks-train-ckks `Encoder` (test `matches_seal_4_1_encoder` in
// crates/ckks-train-ckks/src/encoder.rs), generated with Microsoft SEAL 4.1.2 (MIT).
//
// For N in {16, 32}: CKKS, CoeffModulus::Create(N, {60, 40, 60}), sec_level_type::none,
// scale 2^30, encode at context.first_parms_id() (data primes {60, 40}). Prints the
// inputs and the centered coefficients of limb 0 after inverse_ntt_negacyclic_harvey
// (checked equal to limb 1's centered coefficients). See README.md.
#include "seal/seal.h"
#include "seal/util/ntt.h"
#include <complex>
#include <cstdio>
#include <vector>
using namespace seal;
using namespace std;

static void dump(const SEALContext &ctx, const Plaintext &pt, const char *name) {
    auto cd = ctx.get_context_data(pt.parms_id());
    size_t n = cd->parms().poly_modulus_degree();
    auto &qs = cd->parms().coeff_modulus();
    vector<uint64_t> buf(pt.data(), pt.data() + n * qs.size());
    for (size_t j = 0; j < qs.size(); j++) util::inverse_ntt_negacyclic_harvey(buf.data() + j * n, cd->small_ntt_tables()[j]);
    printf("// %s: N=%zu limbs=%zu q0=%llu\n", name, n, qs.size(), (unsigned long long)qs[0].value());
    printf("[");
    for (size_t i = 0; i < n; i++) {
        uint64_t q = qs[0].value(), c = buf[i];
        long long s = c > q / 2 ? -(long long)(q - c) : (long long)c;
        // consistency with limb 1
        uint64_t q1 = qs[1].value(), c1 = buf[n + i];
        long long s1 = c1 > q1 / 2 ? -(long long)(q1 - c1) : (long long)c1;
        if (s != s1) { fprintf(stderr, "limb mismatch\n"); return; }
        printf("%lld%s", s, i + 1 < n ? ", " : "");
    }
    printf("]\n");
}

int main() {
    for (size_t n : {16, 32}) {
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, {60, 40, 60}));
        SEALContext ctx(parms, true, sec_level_type::none);
        CKKSEncoder enc(ctx);
        size_t slots = n / 2;
        double scale = pow(2.0, 30);
        vector<double> v(slots);
        for (size_t i = 0; i < slots; i++) v[i] = 0.37 * double(i) - 1.1 + (i % 3 == 0 ? 0.25 : -0.125);
        printf("// real values: ");
        for (auto x : v) printf("%.17g, ", x);
        printf("\n");
        Plaintext pt;
        enc.encode(v, ctx.first_parms_id(), scale, pt);
        dump(ctx, pt, "real");
        vector<complex<double>> w(slots);
        for (size_t i = 0; i < slots; i++) w[i] = complex<double>(0.5 - 0.21 * double(i), 0.13 * double(i) - 0.4);
        printf("// complex values: ");
        for (auto x : w) printf("(%.17g, %.17g), ", x.real(), x.imag());
        printf("\n");
        enc.encode(w, ctx.first_parms_id(), scale, pt);
        dump(ctx, pt, "complex");
        // short input (zero padded)
        vector<double> s3 = {0.75, -0.5, 0.125};
        enc.encode(s3, ctx.first_parms_id(), scale, pt);
        dump(ctx, pt, "short [0.75,-0.5,0.125]");
    }
}
