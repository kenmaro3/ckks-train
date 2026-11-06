//! Only an explicit fixed-recipe request produces a compile capsule.
use std::{
    env, fs,
    io::Read,
    path::PathBuf,
    process::{Command, Stdio},
};

const ENV: &[&str] = &[
    "HOME",
    "CARGO_HOME",
    "CARGO_TARGET_DIR",
    "RUSTC",
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_CONFIG_GLOBAL",
    "GIT_OPTIONAL_LOCKS",
    "TMPDIR",
    "CARGO_BUILD_JOBS",
    "CARGO_INCREMENTAL",
    "CARGO_NET_OFFLINE",
    "CARGO_ENCODED_RUSTFLAGS",
    "HEKIT_STUDY_CAPSULE_REQUEST",
    "HEKIT_STUDY_PYTHON",
    "TARGET",
    "HOST",
    "PROFILE",
    "OPT_LEVEL",
    "DEBUG",
    "CARGO_PKG_VERSION",
    "CARGO_MANIFEST_DIR",
    "OUT_DIR",
    "CARGO_FEATURE_PARALLEL",
    "CARGO_FEATURE_CUDA",
    "CARGO_FEATURE_RAYON",
    "CARGO_FEATURE_CKKS_TRAIN_CUDA",
];
fn main() {
    for key in ENV {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let Some(request) = env::var_os("HEKIT_STUDY_CAPSULE_REQUEST") else {
        fs::write(
            out.join("study_capsule.rs"),
            "pub(super) const EMBEDDED_CAPSULE: Option<&[u8]> = None;\n",
        )
        .expect("write absent capsule");
        return;
    };
    for (key, _) in env::vars_os() {
        assert!(
            !key.to_string_lossy().starts_with("CARGO_FEATURE_"),
            "production bench features unsupported"
        );
    }
    for key in [
        "RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTC_BOOTSTRAP",
    ] {
        assert!(
            env::var_os(key).is_none(),
            "unsupported production override: {key}"
        );
        println!("cargo:rerun-if-env-changed={key}");
    }
    let rustc = PathBuf::from(env::var_os("RUSTC").expect("selected compiler"));
    assert!(rustc.is_absolute(), "absolute selected compiler required");
    let mut child = Command::new(&rustc)
        .arg("-vV")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
        .expect("selected rustc -vV");
    let mut version = Vec::new();
    let read = child
        .stdout
        .take()
        .expect("compiler stdout")
        .take(16_385)
        .read_to_end(&mut version);
    if read.is_err() || version.len() > 16_384 {
        let _ = child.kill();
        let _ = child.wait();
        panic!("compiler version exceeds bound");
    }
    assert!(
        child.wait().expect("compiler wait").success(),
        "compiler version unavailable"
    );
    let compiler = String::from_utf8(version).expect("compiler UTF8");
    let python = PathBuf::from(env::var_os("HEKIT_STUDY_PYTHON").expect("selected Python"));
    assert!(python.is_absolute(), "absolute selected Python required");
    let helper = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"))
        .join("../../scripts/study_reference_build.py");
    println!("cargo:rerun-if-changed={}", helper.display());
    println!(
        "cargo:rerun-if-changed={}",
        helper
            .with_file_name("study_reference_build_support.py")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        PathBuf::from(&request).display()
    );
    let mut cmd = Command::new(python);
    cmd.arg("-B")
        .arg(helper)
        .arg("capture")
        .arg("--request")
        .arg(request)
        .arg("--outdir")
        .arg(out)
        .arg("--compiler")
        .arg(compiler);
    for (option, key) in [
        ("target", "TARGET"),
        ("host", "HOST"),
        ("profile", "PROFILE"),
        ("opt", "OPT_LEVEL"),
        ("debug", "DEBUG"),
        ("version", "CARGO_PKG_VERSION"),
    ] {
        cmd.arg(format!("--{option}"))
            .arg(env::var_os(key).expect("Cargo compile fact"));
    }
    assert!(
        cmd.status().expect("capsule capture helper").success(),
        "compile capsule rejected"
    );
}
