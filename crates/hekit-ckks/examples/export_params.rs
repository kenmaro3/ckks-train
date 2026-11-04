//! Exports preset parameters as JSON for `tools/estimate/estimate.py`.
//!
//! Usage: `cargo run -p hekit-ckks --example export_params > tools/estimate/params.json`

use hekit_ckks::params::CkksParams;

fn entry(name: &str, p: &CkksParams) -> String {
    let log_q: f64 = p.q().iter().map(|&m| (m as f64).log2()).sum();
    let log_p: f64 = p.p().iter().map(|&m| (m as f64).log2()).sum();
    let secret = match p.hamming_weight() {
        Some(h) => format!("{h}"),
        None => "\"ternary_uniform\"".to_string(),
    };
    format!(
        "    {{\n      \"name\": \"{name}\",\n      \"n\": {},\n      \"log_q\": {log_q:.4},\n      \"log_p\": {log_p:.4},\n      \"log_pq\": {:.4},\n      \"secret\": {secret},\n      \"sigma\": 3.2\n    }}",
        p.n(),
        p.log_pq()
    )
}

fn main() {
    let presets = [
        entry("seal_2020", &CkksParams::seal_2020()),
        entry("n16_train", &CkksParams::n16_train()),
    ];
    println!("{{\n  \"presets\": [\n{}\n  ]\n}}", presets.join(",\n"));
}
