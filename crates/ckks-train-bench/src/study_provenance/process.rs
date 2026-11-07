//! Internal Linux process identity; no receipt or reconstruction API.
use super::{ProvenanceError, Result, ensure};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ProcessStamp {
    pid: libc::pid_t,
    start_ticks: u64,
    boot: [u8; 36],
}
impl ProcessStamp {
    pub(crate) fn capture() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            use super::publication::read_proc;
            use std::path::Path;
            let (pid, start_ticks) = parse_stat(&read_proc(Path::new("/proc/self/stat"), 16384)?)?;
            // SAFETY: getpid has no pointer or allocation preconditions.
            check_pid(pid, unsafe { libc::getpid() })?;
            let boot = parse_boot(&read_proc(
                Path::new("/proc/sys/kernel/random/boot_id"),
                64,
            )?)?;
            Ok(Self {
                pid,
                start_ticks,
                boot,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(ProvenanceError("process stamp requires native Linux"))
        }
    }
    pub(crate) fn recheck(&self) -> Result<()> {
        self.require_same(&Self::capture()?)
    }
    fn require_same(&self, current: &Self) -> Result<()> {
        ensure(self == current, "process identity changed")
    }
}
fn check_pid(parsed: libc::pid_t, observed: libc::pid_t) -> Result<()> {
    ensure(parsed > 0 && parsed == observed, "process PID mismatch")
}
fn parse_stat(bytes: &[u8]) -> Result<(libc::pid_t, u64)> {
    ensure(bytes.len() <= 16384, "process stat byte cap")?;
    let open = bytes
        .iter()
        .position(|b| *b == b'(')
        .ok_or(ProvenanceError("stat comm missing"))?;
    let close = bytes
        .iter()
        .rposition(|b| *b == b')')
        .ok_or(ProvenanceError("stat comm missing"))?;
    ensure(
        open > 0 && close > open && bytes[open - 1].is_ascii_whitespace(),
        "stat comm framing",
    )?;
    let pid = std::str::from_utf8(&bytes[..open])
        .map_err(|_| ProvenanceError("stat PID encoding"))?
        .trim()
        .parse::<libc::pid_t>()
        .map_err(|_| ProvenanceError("stat PID invalid"))?;
    ensure(pid > 0, "stat PID positive")?;
    let tail = bytes
        .get(close + 1..)
        .ok_or(ProvenanceError("stat tail missing"))?;
    ensure(
        tail.first().is_some_and(u8::is_ascii_whitespace),
        "stat tail separator",
    )?;
    let tail = std::str::from_utf8(tail).map_err(|_| ProvenanceError("stat tail encoding"))?;
    let fields = tail.split_ascii_whitespace().collect::<Vec<_>>();
    ensure(
        fields.len() >= 20 && fields[0].len() == 1 && fields[0].as_bytes()[0].is_ascii_alphabetic(),
        "stat fields missing",
    )?;
    for field in &fields[1..] {
        let digits = field
            .strip_prefix('-')
            .or_else(|| field.strip_prefix('+'))
            .unwrap_or(field);
        ensure(
            !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()),
            "stat numeric field malformed",
        )?;
    }
    let ticks = fields[19]
        .parse::<u64>()
        .map_err(|_| ProvenanceError("stat start ticks invalid"))?;
    Ok((pid, ticks))
}
fn parse_boot(bytes: &[u8]) -> Result<[u8; 36]> {
    ensure(
        bytes.len() == 37 && bytes[36] == b'\n',
        "boot UUID length/newline",
    )?;
    for (i, b) in bytes[..36].iter().enumerate() {
        ensure(
            if [8, 13, 18, 23].contains(&i) {
                *b == b'-'
            } else {
                b.is_ascii_digit() || (b'a'..=b'f').contains(b)
            },
            "boot UUID grammar",
        )?;
    }
    Ok(bytes[..36].try_into().expect("checked UUID length"))
}
#[cfg(test)]
mod tests {
    use super::*;
    const STAT: &str = "4242 (worker ) name)) S 1 4242 4242 0 -1 4194304 10 0 0 0 12 3 0 0 20 0 2 0 123456 1048576 64 18446744073709551615 4194304 4198400 140737488347136 0 0 0 0 0 0 0 0 0 17 3 0 0 0 0 0 4198400 4202496 6291456 140737488348000 140737488348020 140737488348020 140737488349000 0\n";
    const BOOT: &[u8] = b"01234567-89ab-cdef-0123-456789abcdef\n";
    #[test]
    fn process_exact_match_checks_every_field_and_observed_pid() {
        check_pid(4242, 4242).unwrap();
        assert!(check_pid(4243, 4242).is_err());
        let make = || ProcessStamp {
            pid: 4242,
            start_ticks: 123456,
            boot: *b"01234567-89ab-cdef-0123-456789abcdef",
        };
        let held = make();
        held.require_same(&make()).unwrap();
        let mut changed = make();
        changed.pid = 4243;
        assert!(held.require_same(&changed).is_err());
        let mut changed = make();
        changed.start_ticks += 1;
        assert!(held.require_same(&changed).is_err());
        let mut changed = make();
        changed.boot[35] = b'e';
        assert!(held.require_same(&changed).is_err());
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn process_native_linux_capture_rechecks_same_process() {
        ProcessStamp::capture().unwrap().recheck().unwrap();
    }
    #[test]
    fn process_stat_comm_delimiter_and_u64_boundaries() {
        assert_eq!(parse_stat(STAT.as_bytes()).unwrap(), (4242, 123456));
        for (token, value) in [("0", 0), ("1", 1), ("18446744073709551615", u64::MAX)] {
            assert_eq!(
                parse_stat(STAT.replace("123456", token).as_bytes()).unwrap(),
                (4242, value)
            );
        }
    }
    #[test]
    fn process_stat_malformed_missing_overflow_and_trailing_reject() {
        for token in ["-1", "123x", "18446744073709551616"] {
            assert!(parse_stat(STAT.replace("123456", token).as_bytes()).is_err());
        }
        for pid in [
            "0",
            "-4242",
            "4242x",
            "99999999999999999999999999999999999999999999",
        ] {
            assert!(parse_stat(STAT.replacen("4242", pid, 1).as_bytes()).is_err());
        }
        for value in [
            STAT.split("123456").next().unwrap().to_owned(),
            STAT.replace(')', ""),
            STAT.replacen('(', "", 1),
            format!("{STAT}\0junk"),
        ] {
            assert!(parse_stat(value.as_bytes()).is_err());
        }
        assert!(parse_stat(&vec![b'x'; 16385]).is_err());
    }
    #[test]
    fn process_boot_exact_grammar_not_uuid_version_policy() {
        assert_eq!(&parse_boot(BOOT).unwrap(), &BOOT[..36]);
        assert!(parse_boot(b"00000000-0000-0000-0000-000000000000\n").is_ok());
        for value in [
            b"".to_vec(),
            BOOT[..36].to_vec(),
            [BOOT, b"\n"].concat(),
            [BOOT, b"x"].concat(),
            [b" ", BOOT].concat(),
            [&BOOT[..36], b"\r\n"].concat(),
            [BOOT, &[b' '; 27]].concat(),
            [BOOT, &[b' '; 28]].concat(),
        ] {
            assert!(parse_boot(&value).is_err());
        }
        for (index, byte) in [(10, b'A'), (10, b'g'), (8, b'_')] {
            let mut value = BOOT.to_vec();
            value[index] = byte;
            assert!(parse_boot(&value).is_err());
        }
    }
}
