//! Prints the precision table (Markdown) of every basic operation for both presets.
//!
//! Usage: `cargo run -p hekit-ckks --release --example precision`

use hekit_ckks::params::CkksParams;
use hekit_ckks::precision::run_precision_suite;

fn main() {
    for (name, params) in [
        ("seal_2020", CkksParams::seal_2020()),
        ("n16_train", CkksParams::n16_train()),
    ] {
        println!("### {name}\n");
        println!("| op | level | bits |");
        println!("|----|-------|------|");
        for r in run_precision_suite(params) {
            println!("| {} | {} | {:.2} |", r.op, r.level, r.bits);
        }
        println!();
    }
}
