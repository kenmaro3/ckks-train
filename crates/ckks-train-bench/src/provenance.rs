//! Where a result comes from, written by the binary itself (moved here from the `mnist` bin in
//! Task 48 fix round 1, so that `mnist` and `ablation` write the same object): the git commit
//! and the tracked files that differ from it, the command line, the build, the machine, the
//! threads, the load before and after, the peak RSS and the start and finish times.

use std::process::Command;

use serde_json::{Value, json};

/// The stdout of `cmd args` (trimmed), or `"unknown"`.
pub fn cmd_out(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Seconds since the Unix epoch.
pub fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(f64::NAN, |d| d.as_secs_f64())
}

/// The load averages: `sysctl -n vm.loadavg` (macOS), else `/proc/loadavg` (Linux), else
/// `"unknown"`.
pub fn loadavg() -> String {
    Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .or_else(|| std::fs::read_to_string("/proc/loadavg").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// This process's peak resident set size in bytes (`getrusage(RUSAGE_SELF)`; `ru_maxrss` is
/// in bytes on macOS and in KiB on Linux), or `None` if the call fails.
pub fn peak_rss_bytes() -> Option<u64> {
    // SAFETY: `getrusage` only writes the `rusage` struct it is given, which is plain data
    // and fully initialised by `zeroed`.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    if rc != 0 {
        return None;
    }
    let raw = u64::try_from(ru.ru_maxrss).ok()?;
    Some(if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    })
}

/// The CPU model: `sysctl machdep.cpu.brand_string` (macOS), else the first `model name` of
/// `/proc/cpuinfo` (Linux), else `"unknown"`.
pub fn cpu_name() -> String {
    let mac = cmd_out("sysctl", &["-n", "machdep.cpu.brand_string"]);
    if mac != "unknown" {
        return mac;
    }
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

/// Physical memory in bytes: `sysctl hw.memsize` (macOS), else `MemTotal` of `/proc/meminfo`
/// (Linux, converted from kB), else `"unknown"`.
pub fn memory_bytes() -> String {
    let mac = cmd_out("sysctl", &["-n", "hw.memsize"]);
    if mac != "unknown" {
        return mac;
    }
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or_else(|| "unknown".into(), |kb| (kb * 1024).to_string())
}

/// How this binary was built (compile-time facts of the `ckks-train-bench` crate; the binaries
/// share its features): debug assertions, version, target, and the `parallel` / `cuda`
/// features.
pub fn build_info() -> Value {
    json!({
        "debug_assertions": cfg!(debug_assertions),
        "package_version": env!("CARGO_PKG_VERSION"),
        "target_os": std::env::consts::OS,
        "target_arch": std::env::consts::ARCH,
        "features": {
            "parallel": cfg!(feature = "parallel"),
            "cuda": cfg!(feature = "cuda"),
        },
    })
}

/// The provenance object: `written_by`, git commit and modified tracked files, command line,
/// build, rustc, machine (CPU, logical CPUs, memory, OS, host), threads, load at `load_start`
/// and now, peak RSS so far, and the start and finish times. Call it when the result is
/// written (the finish time and the end load are taken then).
pub fn provenance(written_by: &str, load_start: &str, started_unix: f64) -> Value {
    let threads = std::thread::available_parallelism().map_or(0, |n| n.get());
    let rayon = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    let dirty = cmd_out("git", &["status", "--porcelain", "--untracked-files=no"]);
    json!({
        "written_by": written_by,
        "git_commit": cmd_out("git", &["rev-parse", "HEAD"]),
        "git_modified_tracked_files": if dirty == "unknown" { "none".to_string() } else { dirty },
        "command_line": std::env::args().collect::<Vec<_>>().join(" "),
        "build": build_info(),
        "parallel_feature": cfg!(feature = "parallel"),
        "rustc_on_path": cmd_out("rustc", &["-V"]),
        "machine": {
            "cpu": cpu_name(),
            "logical_cpus": threads,
            "memory_bytes": memory_bytes(),
            "os": cmd_out("uname", &["-srm"]),
            "host": cmd_out("hostname", &["-s"]),
        },
        "threads": if cfg!(feature = "parallel") {
            format!("rayon (feature parallel), RAYON_NUM_THREADS = {rayon}, {threads} logical CPUs")
        } else {
            "1 (no parallel feature)".to_string()
        },
        "load_start": load_start,
        "load_end": loadavg(),
        "peak_rss_bytes": peak_rss_bytes(),
        "started_unix": started_unix,
        "finished_unix": unix_now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_has_every_field() {
        let p = provenance("test", "1.0 1.0 1.0", unix_now());
        for k in [
            "written_by",
            "git_commit",
            "git_modified_tracked_files",
            "command_line",
            "build",
            "parallel_feature",
            "rustc_on_path",
            "machine",
            "threads",
            "load_start",
            "load_end",
            "peak_rss_bytes",
            "started_unix",
            "finished_unix",
        ] {
            assert!(p.get(k).is_some(), "{k} missing");
        }
        for k in ["cpu", "logical_cpus", "memory_bytes", "os", "host"] {
            assert!(p["machine"].get(k).is_some(), "machine.{k} missing");
        }
        assert!(p["build"]["features"].get("cuda").is_some());
        assert!(p["peak_rss_bytes"].as_u64().is_some_and(|b| b > 0));
    }
}
