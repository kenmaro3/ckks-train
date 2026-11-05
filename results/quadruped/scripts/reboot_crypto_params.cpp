// R-REB-4 helper (my code; links OpenFHE 1.2.1, BSD-2). Rebuilds the CKKS context from the CCParams values ReBoot
// passes (read from the Python dump) and prints the modulus chain, P moduli, dnum, sigma and the Hamming weight of a
// freshly generated secret key, as JSON. Usage: reboot_crypto_params RINGDIM DEPTH SCALEBITS FIRSTBITS NUMLARGEDIGITS
#include "openfhe.h"
#include <cmath>
#include <iostream>
using namespace lbcrypto;
int main(int argc, char** argv) {
    if (argc < 6) return 2;
    CCParams<CryptoContextCKKSRNS> p;
    p.SetRingDim(std::stoul(argv[1]));
    p.SetMultiplicativeDepth(std::stoul(argv[2]));
    p.SetScalingModSize(std::stoul(argv[3]));
    p.SetFirstModSize(std::stoul(argv[4]));
    p.SetNumLargeDigits(std::stoul(argv[5]));
    p.SetSecurityLevel(HEStd_128_classic);
    p.SetSecretKeyDist(SPARSE_TERNARY);
    p.SetKeySwitchTechnique(HYBRID);
    p.SetScalingTechnique(FLEXIBLEAUTOEXT);
    p.SetBatchSize(0);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE); cc->Enable(FHE);
    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto q = cp->GetElementParams()->GetParams();
    auto pp = cp->GetParamsP()->GetParams();
    double logQ = 0, logP = 0;
    std::cout << "{\n \"ring_dim\": " << cc->GetRingDimension() << ",\n \"q_moduli\": [";
    for (size_t i = 0; i < q.size(); i++) {
        double b = std::log2(q[i]->GetModulus().ConvertToDouble()); logQ += b;
        std::cout << (i ? ", " : "") << "{\"value\": \"" << q[i]->GetModulus() << "\", \"bits\": " << b << "}";
    }
    std::cout << "],\n \"p_moduli\": [";
    for (size_t i = 0; i < pp.size(); i++) {
        double b = std::log2(pp[i]->GetModulus().ConvertToDouble()); logP += b;
        std::cout << (i ? ", " : "") << "{\"value\": \"" << pp[i]->GetModulus() << "\", \"bits\": " << b << "}";
    }
    std::cout.precision(10);
    std::cout << "],\n \"num_q\": " << q.size() << ", \"num_p\": " << pp.size() << ",\n \"log2_Q\": " << logQ
              << ", \"log2_P\": " << logP << ", \"log2_PQ\": " << logQ + logP
              << ",\n \"dnum_num_part_q\": " << cp->GetNumPartQ() << ", \"alpha_towers_per_digit\": " << cp->GetNumPerPartQ()
              << ",\n \"key_switch_technique\": \"" << cp->GetKeySwitchTechnique() << "\", \"scaling_technique\": \""
              << cp->GetScalingTechnique() << "\",\n \"secret_key_dist\": \"" << cp->GetSecretKeyDist()
              << "\", \"standard_deviation\": " << cp->GetDistributionParameter()
              << ", \"security_level\": \"" << cp->GetStdLevel() << "\"";
    auto kp = cc->KeyGen();
    DCRTPoly s = kp.secretKey->GetPrivateElement();
    s.SetFormat(Format::COEFFICIENT);
    auto t0 = s.GetElementAtIndex(0);
    auto m0 = t0.GetModulus();
    size_t nz = 0, plus = 0, minus = 0, other = 0;
    for (size_t i = 0; i < t0.GetLength(); i++) {
        auto v = t0[i];
        if (v == 0) continue;
        nz++;
        if (v == 1) plus++; else if (v == m0 - 1) minus++; else other++;
    }
    std::cout << ",\n \"secret_key_hamming_weight_measured\": " << nz << ", \"sk_plus_one\": " << plus
              << ", \"sk_minus_one\": " << minus << ", \"sk_other\": " << other << "\n}\n";
    return 0;
}
