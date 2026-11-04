// SEAL 4.1 negacyclic Harvey NTT micro-benchmark (for Task 2's Rust/SEAL ratio).
// N = 2^16, q = 1152921504606584833 (60-bit, q = 1 mod 2N; the prime Task 2 used).
// Times one forward NTT per rep (input copy excluded), reports median/mean/min.
#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/util/ntt.h"
#include "seal/version.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

template <class F>
static std::vector<double> bench(std::vector<uint64_t> &buf, const std::vector<uint64_t> &src, int reps, F f) {
    std::vector<double> t;
    t.reserve(size_t(reps));
    for (int r = 0; r < reps; ++r) {
        std::copy(src.begin(), src.end(), buf.begin());
        auto t0 = clk::now();
        f();
        auto t1 = clk::now();
        t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    return t;
}

static void report(const char *name, std::vector<double> t) {
    std::sort(t.begin(), t.end());
    double mean = 0;
    for (double x : t) mean += x;
    mean /= double(t.size());
    std::printf("%-34s reps %zu  median %.1f us  mean %.1f us  min %.1f us  p90 %.1f us\n", name, t.size(),
                t[t.size() / 2], mean, t.front(), t[(t.size() * 9) / 10]);
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? std::atoi(argv[1]) : 1000;
    const int logn = 16;
    const size_t n = size_t(1) << logn;
    const uint64_t qv = 1152921504606584833ULL;
    seal::Modulus q(qv);
    if (!q.is_prime() || (qv - 1) % (2 * n) != 0) {
        std::fprintf(stderr, "q is not an NTT prime for N=2^16\n");
        return 1;
    }
    seal::util::NTTTables tables(logn, q, seal::MemoryManager::GetPool());
    std::mt19937_64 g(42);
    std::vector<uint64_t> src(n), buf(n);
    for (auto &v : src) v = g() % qv;

    // correctness sanity: forward then inverse returns the input
    std::copy(src.begin(), src.end(), buf.begin());
    seal::util::ntt_negacyclic_harvey(seal::util::CoeffIter(buf.data()), tables);
    seal::util::inverse_ntt_negacyclic_harvey(seal::util::CoeffIter(buf.data()), tables);
    if (buf != src) {
        std::fprintf(stderr, "round-trip mismatch\n");
        return 1;
    }
    std::printf("# SEAL %d.%d.%d NTT bench: N = 2^%d, q = %llu (%d bits), single thread\n", SEAL_VERSION_MAJOR,
                SEAL_VERSION_MINOR, SEAL_VERSION_PATCH, logn, (unsigned long long)qv, q.bit_count());

    // warm-up
    bench(buf, src, 50, [&] { seal::util::ntt_negacyclic_harvey(seal::util::CoeffIter(buf.data()), tables); });

    report("forward ntt_negacyclic_harvey", bench(buf, src, reps, [&] {
               seal::util::ntt_negacyclic_harvey(seal::util::CoeffIter(buf.data()), tables);
           }));
    report("forward ntt_negacyclic_harvey_lazy", bench(buf, src, reps, [&] {
               seal::util::ntt_negacyclic_harvey_lazy(seal::util::CoeffIter(buf.data()), tables);
           }));
    report("inverse_ntt_negacyclic_harvey", bench(buf, src, reps, [&] {
               seal::util::inverse_ntt_negacyclic_harvey(seal::util::CoeffIter(buf.data()), tables);
           }));
    return 0;
}
