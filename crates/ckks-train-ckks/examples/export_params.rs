//! Exports preset parameters as JSON for `tools/estimate/estimate.py`.
//!
//! Usage: `cargo run -p ckks-train-ckks --example export_params > tools/estimate/params.json`
//!
//! Besides the presets themselves (RLWE under the main key at modulus Q*P), it exports the
//! sparse-secret-encapsulation switching key s -> s' of `n16_boot` (`SseKeys::to_sparse`): RLWE
//! samples under the ephemeral key s' (Hamming weight `BootParams::default_n16().ephemeral_hw`)
//! at modulus q_0 * P only.

use ckks_train_ckks::bootstrap::BootParams;
use ckks_train_ckks::keys::ERROR_STD_DEV;
use ckks_train_ckks::params::CkksParams;

fn log2_sum(m: &[u64]) -> f64 {
    m.iter().map(|&m| (m as f64).log2()).sum()
}

fn entry_raw(name: &str, n: usize, log_q: f64, log_p: f64, secret: Option<usize>) -> String {
    let secret = match secret {
        Some(h) => format!("{h}"),
        None => "\"ternary_default\"".to_string(),
    };
    format!(
        "    {{\n      \"name\": \"{name}\",\n      \"n\": {n},\n      \"log_q\": {log_q:.4},\n      \"log_p\": {log_p:.4},\n      \"log_pq\": {:.4},\n      \"secret\": {secret},\n      \"sigma\": {ERROR_STD_DEV}\n    }}",
        log_q + log_p
    )
}

fn entry(name: &str, p: &CkksParams) -> String {
    entry_raw(
        name,
        p.n(),
        log2_sum(p.q()),
        log2_sum(p.p()),
        p.hamming_weight(),
    )
}

fn main() {
    let boot = CkksParams::n16_boot();
    let h_eph = BootParams::default_n16()
        .ephemeral_hw
        .expect("n16_boot uses SSE");
    let presets = [
        entry("seal_2020", &CkksParams::seal_2020()),
        entry("n16_train", &CkksParams::n16_train()),
        entry("n16_boot", &boot),
        entry("n16_boot_sparse", &CkksParams::n16_boot_sparse()),
        entry_raw(
            "n16_boot_sse_to_sparse",
            boot.n(),
            log2_sum(&boot.q()[..1]),
            log2_sum(boot.p()),
            Some(h_eph),
        ),
    ];
    println!("{{\n  \"presets\": [\n{}\n  ]\n}}", presets.join(",\n"));
}
