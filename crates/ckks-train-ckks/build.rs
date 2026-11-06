// Configured-compiler diagnostics only: not source/build/executable attestation.
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    for name in [
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "TARGET",
        "HOST",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    let compiler = env::var_os("RUSTC")
        .and_then(|rustc| Command::new(rustc).arg("-vV").output().ok())
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok());
    let diagnostic = |value: Option<String>| match value {
        Some(value) if !value.is_empty() => format!("Some({value:?})"),
        _ => "None".to_owned(),
    };
    let generated = format!(
        "pub const CONFIGURED_COMPILER: Option<&str> = {};\npub const TARGET: Option<&str> = {};\npub const HOST: Option<&str> = {};\n",
        diagnostic(compiler),
        diagnostic(env::var("TARGET").ok()),
        diagnostic(env::var("HOST").ok()),
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    fs::write(output.join("bootstrap_diagnostics.rs"), generated)
        .expect("write compiler diagnostics");
}
