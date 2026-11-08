use super::protocol::{Decoder, Frame, Tag, checked_time, elapsed, encode};
#[test]
fn d7_supervisor_frames_literal_roundtrip() {
    // Frame length11: tag0, sequence0, payload aa bb. Independent literal.
    let bytes = [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xaa, 0xbb];
    let frame = Frame {
        tag: Tag::Hello,
        sequence: 0,
        payload: vec![0xaa, 0xbb],
    };
    assert_eq!(encode(&frame).unwrap(), bytes);
    let mut d = Decoder::new();
    for b in &bytes[..14] {
        d.feed(&[*b]).unwrap();
        assert!(d.next().unwrap().is_none());
    }
    d.feed(&bytes[14..]).unwrap();
    assert_eq!(d.next().unwrap(), Some(frame));
    assert!(d.next().unwrap().is_none());
}
#[test]
fn d7_supervisor_frames_reject_bounds_order_eof() {
    for length in [0u32, 8, 8 * 1024 * 1024 + 10, u32::MAX] {
        let mut d = Decoder::new();
        assert!(d.feed(&length.to_le_bytes()).is_err());
    }
    let mut d = Decoder::new();
    let f = Frame {
        tag: Tag::Hello,
        sequence: 0,
        payload: Vec::new(),
    };
    let b = encode(&f).unwrap();
    d.feed(&b).unwrap();
    d.next().unwrap();
    d.feed(&b).unwrap();
    assert!(d.next().is_err());
    let mut bad = b.clone();
    bad[4] = 9;
    let mut d = Decoder::new();
    d.feed(&bad).unwrap();
    assert!(d.next().is_err());
    let mut d = Decoder::new();
    d.feed(&b[..6]).unwrap();
    assert!(d.eof().is_err());
    let mut d = Decoder::new();
    let mut skipped = b.clone();
    skipped[5] = 1;
    d.feed(&skipped).unwrap();
    assert!(d.next().is_err());
}
#[test]
fn d7_supervisor_clock_checked_common_origin() {
    assert_eq!(checked_time(3, 4).unwrap(), 3_000_000_004);
    for (s, n) in [(-1, 0), (0, -1), (0, 1_000_000_000), (i64::MAX, 0)] {
        assert!(checked_time(s, n).is_err());
    }
    assert_eq!(elapsed(5, 7).unwrap(), 2);
    assert_eq!(elapsed(5, 9).unwrap(), 4);
    assert!(elapsed(5, 4).is_err());
}
#[test]
fn d7_supervisor_broker_exact_git_argv() {
    use super::protocol::git_argv;
    let allowed: &[&[&str]] = &[
        &["rev-parse", "--show-toplevel"],
        &["rev-parse", "HEAD"],
        &[
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "HEAD",
            "--",
        ],
        &["ls-files", "--others", "--exclude-standard", "-z"],
        &["ls-files", "--stage", "-z"],
    ];
    for suffix in allowed {
        let args = git_argv(b"/repo", suffix).unwrap();
        let mut expected = vec![
            b"--no-pager".to_vec(),
            b"-c".to_vec(),
            b"core.fsmonitor=false".to_vec(),
            b"-c".to_vec(),
            b"core.untrackedCache=false".to_vec(),
            b"-C".to_vec(),
            b"/repo".to_vec(),
        ];
        expected.extend(suffix.iter().map(|s| s.as_bytes().to_vec()));
        assert_eq!(args, expected);
    }
    assert!(git_argv(b"/repo", &["status"]).is_err());
    assert!(git_argv(b"/repo", &["diff", "--quiet", "HEAD", "--"]).is_err());
    assert!(git_argv(b"/repo\0x", allowed[0]).is_err());
}
#[test]
fn d7_supervisor_proc_rss_identity() {
    use super::observation::{rss_kib, start_ticks};
    assert_eq!(rss_kib(b"Name:\tunit\nVmRSS:\t42 kB\n").unwrap(), 42);
    for raw in [
        b"VmRSS: 42 MB\n".as_slice(),
        b"VmRSS: 1 kB\nVmRSS: 2 kB\n",
        b"VmRSS: -1 kB\n",
        b"Name: unit\n",
    ] {
        assert!(rss_kib(raw).is_err());
    }
    let stat = b"11 (name ) spaces) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 123 20";
    assert_eq!(start_ticks(stat, 11).unwrap(), 123);
    assert!(start_ticks(stat, 12).is_err());
}
#[test]
fn d7_supervisor_storage_aggregate_caps() {
    use super::observation::Storage;
    let mut s = Storage::new();
    s.add_raw(8 * 1024 * 1024).unwrap();
    for _ in 1..8 {
        s.add_raw(8 * 1024 * 1024).unwrap();
    }
    assert!(s.add_raw(1).is_err());
    let mut s = Storage::new();
    for _ in 0..4096 {
        s.add_raw(0).unwrap();
    }
    assert!(s.add_raw(0).is_err());
    assert!(Storage::new().add_raw(8 * 1024 * 1024 + 1).is_err());
}
#[test]
fn operational_handshake_projection_binds_all_fields() {
    use super::handshake::Projection;
    // Independent ordered literal, not a production encoder round-trip oracle.
    let mut raw = b"HKCPU1".to_vec();
    for n in [11u64, 101, 12, 102, 2051, 8193, 1000000] {
        raw.extend_from_slice(&n.to_le_bytes());
    }
    raw.extend_from_slice(b"01234567-89ab-cdef-0123-456789abcdef");
    raw.extend_from_slice(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    raw.extend_from_slice(b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    let p = Projection::decode(&raw).unwrap();
    assert_eq!(
        (p.parent.pid, p.parent.start, p.child.pid, p.child.start),
        (11, 101, 12, 102)
    );
    assert_eq!(p.lock, (2051, 8193));
    assert_eq!(p.origin, 1000000);
    assert_eq!(p.encode().as_slice(), raw);
    for position in [6, 14, 22, 30, 38, 46, 54, 62, 98, 162] {
        let mut changed = raw.clone();
        changed[position] = if position >= 62 {
            b'c'
        } else {
            changed[position] + 3
        };
        assert_ne!(Projection::decode(&changed).unwrap(), p);
    }
    for bad in [
        &raw[..raw.len() - 1],
        [raw.as_slice(), b"x"].concat().as_slice(),
    ] {
        assert!(Projection::decode(bad).is_err());
    }
}
#[test]
fn operational_peer_credentials_reject_wrong_creator_uid_and_fd() {
    use super::handshake::peer_matches;
    assert!(peer_matches(true, 11, 1000, 11, 1000, 1000).is_ok());
    for (socket, pid, uid, parent, real, effective) in [
        (false, 11, 1000, 11, 1000, 1000),
        (true, 12, 1000, 11, 1000, 1000),
        (true, 11, 0, 11, 1000, 1000),
        (true, 11, 1000, 11, 0, 1000),
        (true, 0, 1000, 0, 1000, 1000),
    ] {
        assert!(peer_matches(socket, pid, uid, parent, real, effective).is_err());
    }
}
#[test]
fn operational_broker_request_payload_exact_eof_and_bounds() {
    use super::protocol::HelperRequest;
    let mut raw = vec![1];
    raw.extend_from_slice(&2u64.to_le_bytes());
    raw.extend_from_slice(b"/r");
    raw.extend_from_slice(&1u64.to_le_bytes());
    raw.extend_from_slice(&4u64.to_le_bytes());
    raw.extend_from_slice(b"/git");
    let r = HelperRequest::decode(&raw).unwrap();
    assert_eq!(r.kind, 1);
    assert_eq!(r.cwd, b"/r");
    assert_eq!(r.argv, [b"/git"]);
    for i in 0..raw.len() {
        assert!(HelperRequest::decode(&raw[..i]).is_err());
    }
    let mut bad = raw.clone();
    bad.push(0);
    assert!(HelperRequest::decode(&bad).is_err());
    let mut bad = raw.clone();
    bad[0] = 2;
    assert!(HelperRequest::decode(&bad).is_err());
    let mut bad = raw.clone();
    bad[11..19].copy_from_slice(&257u64.to_le_bytes());
    assert!(HelperRequest::decode(&bad).is_err());
    let mut bad = raw.clone();
    bad[19..27].copy_from_slice(&4097u64.to_le_bytes());
    assert!(HelperRequest::decode(&bad).is_err());
    let mut bad = raw;
    bad[27] = 0;
    assert!(HelperRequest::decode(&bad).is_err());
}
#[test]
fn operational_capacity_strict_kernel_units_and_headroom() {
    use super::capacity::*;
    assert_eq!(
        mem_available(b"MemTotal: 100 kB\nMemAvailable: 12 kB\n").unwrap(),
        12288
    );
    for raw in [
        b"MemAvailable: 12 KB\n".as_slice(),
        b"MemAvailable: 12 kB\nMemAvailable: 12 kB\n",
        b"MemAvailable: 18446744073709551615 kB\n",
        b"MemTotal: 100 kB\n",
    ] {
        assert!(mem_available(raw).is_err());
    }
    assert_eq!(memory_limit(b"max\n").unwrap(), None);
    assert_eq!(memory_limit(b"123\n").unwrap(), Some(123));
    for raw in [
        b"001\n".as_slice(),
        b"1\n2\n",
        b"-1\n",
        b"18446744073709551616\n",
    ] {
        assert!(memory_limit(raw).is_err());
    }
    assert!(fits(100, &[(Some(80), 30), (None, 400)], 50).is_ok());
    assert!(fits(100, &[(Some(80), 30)], 51).is_err());
    assert!(fits(49, &[(None, 0)], 50).is_err());
    assert_eq!(membership(b"0::/a/b\n").unwrap(), "/a/b");
    assert!(membership(b"0::/a\n1:memory:/a\n").is_err());
    assert!(membership(b"0::/a/../b\n").is_err());
    let mount =
        b"29 25 0:28 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n";
    assert!(global_mount(mount, 29, b"/sys/fs/cgroup").is_ok());
    assert!(global_mount(mount, 30, b"/sys/fs/cgroup").is_err());
    assert!(
        global_mount(
            &String::from_utf8(mount.to_vec())
                .unwrap()
                .replace("0:28 / ", "0:28 /hidden ")
                .into_bytes(),
            29,
            b"/sys/fs/cgroup"
        )
        .is_err()
    );
}
#[test]
fn operational_exited_helper_waits_for_both_eofs() {
    use super::protocol::DrainState;
    let mut d = DrainState::new();
    d.status = Some(0);
    assert!(!d.ready());
    d.stdout_bytes(65536).unwrap();
    d.stdout_bytes(65537).unwrap();
    assert!(!d.ready());
    d.stdout_eof = true;
    assert!(!d.ready());
    d.stderr_eof = true;
    assert!(d.ready());
    let mut d = DrainState::new();
    d.stdout_eof = true;
    d.stderr_eof = true;
    assert!(!d.ready());
    d.status = Some(2);
    assert!(d.ready());
    let mut d = DrainState::new();
    assert!(d.stdout_bytes(8 * 1024 * 1024).is_err());
    assert!(!d.ready());
}
#[test]
fn operational_ready_frame_preserves_following_raw_socket_bytes() {
    use super::protocol::{Channel, Frame, Tag, encode};
    use std::{
        io::{Read, Write},
        os::unix::net::UnixStream,
    };
    let raw = (0..65537).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    let frame = Frame {
        tag: Tag::Ready,
        sequence: 0,
        payload: b"ready".to_vec(),
    };
    let framed = encode(&frame).unwrap();
    let complete = [framed.as_slice(), raw.as_slice()].concat();
    for chunk in [1, 65536] {
        let (mut writer, reader) = UnixStream::pair().unwrap();
        writer.set_nonblocking(true).unwrap();
        let mut channel = Channel::new(reader).unwrap();
        let mut offset = 0;
        let mut received = None;
        let mut trailing = Vec::new();
        for _ in 0..200000 {
            if offset < complete.len() {
                let end = (offset + chunk).min(complete.len());
                match writer.write(&complete[offset..end]) {
                    Ok(n) => offset += n,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(e) => panic!("local transport write: {e}"),
                }
            }
            if received.is_none() {
                let frames = channel.turn().unwrap();
                assert!(frames.len() <= 1);
                if let Some(f) = frames.into_iter().next() {
                    received = Some(f);
                }
            } else {
                let mut buf = [0; 65536];
                match channel.stream.read(&mut buf) {
                    Ok(n) => trailing.extend_from_slice(&buf[..n]),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(e) => panic!("local raw read: {e}"),
                }
            }
            if offset == complete.len() && trailing.len() == raw.len() {
                break;
            }
        }
        assert_eq!(received, Some(frame.clone()));
        assert_eq!(trailing, raw);
    }
}

#[test]
fn operational_live_broker_has_only_worker_python_git_edges() {
    for kind in 0..4 {
        for request in 0..3 {
            for active in 0..4 {
                assert_eq!(
                    super::protocol::broker_edge(kind, request, active).is_ok(),
                    (kind == 0 && request == 0 && active == 0)
                        || (kind == 1 && request == 1 && active == 1)
                );
            }
        }
    }
}

#[test]
fn operational_probe_diagnostics_are_bounded_event_chunks() {
    use super::{protocol::DiagnosticCapture, schedule::Token};
    let token = Token {
        pid: 11,
        start: 100,
    };
    let mut capture = DiagnosticCapture::new();
    let raw = vec![b'x'; 8 * 1024 * 1024];
    capture.push(token, &raw).unwrap();
    assert_eq!(capture.bytes, raw.len());
    assert_eq!(capture.chunks.len(), 2048);
    assert!(
        capture
            .chunks
            .iter()
            .all(|(t, b)| *t == token && b.len() == 4096)
    );
    assert!(capture.push(token, b"x").is_err());
    assert_eq!(capture.bytes, raw.len());
    assert_eq!(capture.chunks.len(), 2048);
}

#[test]
fn operational_native_stopped_state_is_actual_and_unique() {
    use super::observation::stopped;
    assert!(
        stopped(
            b"Name:	x\n
State:	T (stopped)
Pid:	42
"
        )
        .unwrap()
    );
    assert!(
        !stopped(
            b"State:	S (sleeping)
"
        )
        .unwrap()
    );
    for raw in [
        b"State:	Z (zombie)
State:	T (stopped)
"
        .as_slice(),
        b"Pid:	42
",
        b"State:	X
",
        b"State:	TT (stopped)
",
    ] {
        assert!(stopped(raw).is_err());
    }
}
#[test]
fn operational_native_rescue_projection_binds_direction_and_session() {
    use super::{
        handshake::Projection,
        probe_transfer::{Boundary, rescue_record, validate_rescue},
        schedule::Token,
    };
    let outer = Projection {
        parent: Token {
            pid: 10,
            start: 100,
        },
        child: Token {
            pid: 11,
            start: 101,
        },
        lock: (1, 2),
        origin: 9,
        boot: *b"11111111-1111-1111-1111-111111111111",
        executable: [b'a'; 64],
        invocation: [b'b'; 64],
    };
    let mut inner = outer.clone();
    inner.parent = outer.child;
    inner.child = Token {
        pid: 12,
        start: 102,
    };
    inner.invocation = [b'c'; 64];
    let a = rescue_record(Boundary::RescueToMiddle, &outer, None).unwrap();
    let b = rescue_record(Boundary::RescueToLeaf, &outer, Some(&inner)).unwrap();
    assert!(validate_rescue(&a, Boundary::RescueToMiddle, &outer, None).is_ok());
    assert!(validate_rescue(&b, Boundary::RescueToLeaf, &outer, Some(&inner)).is_ok());
    assert!(validate_rescue(&a, Boundary::RescueToLeaf, &outer, Some(&inner)).is_err());
    assert!(rescue_record(Boundary::CustodyToOuter, &outer, None).is_err());
    let mut changed = outer.clone();
    changed.invocation[0] = b'd';
    assert!(validate_rescue(&a, Boundary::RescueToMiddle, &changed, None).is_err());
    for field in 0..3 {
        let mut bad = inner.clone();
        match field {
            0 => bad.parent.start += 1,
            1 => bad.origin += 1,
            _ => bad.lock.0 += 1,
        };
        assert!(rescue_record(Boundary::RescueToLeaf, &outer, Some(&bad)).is_err());
    }
}

#[test]
fn operational_fix1_history_stream_bounds() {
    use super::history::{Names, record_length, visit};
    let mut seen = 0usize;
    visit(
        (0..5000).map(|i| Ok(format!("probe_{i}").into_bytes())),
        Names::Probes,
        |name| {
            assert_eq!(name, format!("probe_{seen}"));
            record_length(8 * 1024 * 1024, 8 * 1024 * 1024)?;
            record_length(8 * 1024 * 1024, 8 * 1024 * 1024)?;
            seen += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, 5000); // 80,000MiB replay lengths, never allocated/retained.
    assert!(record_length(8 * 1024 * 1024 + 1, 8 * 1024 * 1024).is_err());
    let run = |rows: &[&[u8]], mode| visit(rows.iter().map(|s| Ok(s.to_vec())), mode, |_| Ok(()));
    assert!(run(&[b"command.json", b"observations.bin"], Names::Pair(false)).is_ok());
    assert!(run(&[b"command.json"], Names::Pair(true)).is_ok());
    for rows in [
        vec![b"command.json".as_slice()],
        vec![b"command.json".as_slice(), b"command.json".as_slice()],
        vec![b"other".as_slice()],
    ] {
        assert!(run(&rows, Names::Pair(false)).is_err());
    }
    assert!(
        run(
            &[b"pidfd-owned-unreaped", b"pidfd-owned-unreaped"],
            Names::Checks
        )
        .is_err()
    );
    assert!(run(&[], Names::Checks).is_err());
    assert!(run(&[b"bad/name"], Names::Probes).is_err());
    assert!(
        visit(
            [Err(crate::study_provenance::ProvenanceError(
                "iterator error"
            ))],
            Names::Probes,
            |_| Ok(())
        )
        .is_err()
    );
    assert!(
        visit([Ok(b"probe".to_vec())], Names::Probes, |_| Err(
            crate::study_provenance::ProvenanceError("retained pair error")
        ))
        .is_err()
    );
}

#[test]
fn operational_fix1_directory_stamp() {
    use super::history::DirectoryStamp;
    let a = DirectoryStamp {
        identity: (1, 2),
        size: 4096,
        links: 2,
        modified: (3, 4),
        changed: (5, 6),
    };
    assert!(a.unchanged(&a).is_ok());
    for b in [
        DirectoryStamp {
            identity: (1, 3),
            ..a
        },
        DirectoryStamp { size: 8192, ..a },
        DirectoryStamp { links: 3, ..a },
        DirectoryStamp {
            modified: (3, 5),
            ..a
        },
        DirectoryStamp {
            changed: (5, 7),
            ..a
        },
    ] {
        assert!(a.unchanged(&b).is_err());
    }
}

#[test]
fn operational_fix1_final_sample_gate() {
    use super::completion::{assessment, final_sample_result};
    use crate::study_reference::operational::SampleValues;
    let sample = |rss, start, end| {
        let (valid, status) = assessment(rss, start, end, 20_000_000_000, 4);
        SampleValues {
            sequence: 2,
            started: (100, start),
            finished: (101, end),
            phase: 4,
            processes: Vec::new(),
            valid,
            status,
        }
    };
    for s in [
        sample(
            super::observation::rss_kib(b"VmRSS: broken kB\n").ok(),
            19_000_000_000,
            20_000_000_000,
        ),
        sample(Some(1), 19_000_000_000, 20_000_000_001),
        sample(Some(1), 19_000_000_000, 18_000_000_000),
        sample(
            Some(super::schedule::RSS_LIMIT + 1),
            19_000_000_000,
            20_000_000_000,
        ),
    ] {
        let mut unresolved = false;
        assert!(final_sample_result(Ok(s), &mut unresolved).is_err());
        assert!(unresolved);
        assert!(
            final_sample_result(
                Ok(sample(Some(1), 19_000_000_000, 20_000_000_000)),
                &mut unresolved
            )
            .is_err()
        );
    }
    let mut unresolved = false;
    assert!(
        final_sample_result(
            Ok(sample(
                Some(super::schedule::RSS_LIMIT),
                19_000_000_000,
                20_000_000_000
            )),
            &mut unresolved
        )
        .is_ok()
    );
    assert!(!unresolved);
    assert!(
        final_sample_result(
            Err(crate::study_provenance::ProvenanceError(
                "final clock/read error"
            )),
            &mut unresolved
        )
        .is_err()
    );
    assert!(unresolved);
}

#[test]
fn operational_fix1_clock_fault_cleanup_path() {
    use super::clock_fault::{Fault, checked};
    use super::schedule::{ClockFailureLatch, FallbackClock};
    for after in [false, true] {
        let fault = Fault::new(after);
        let latch = ClockFailureLatch::new();
        if after {
            assert_eq!(checked(10, Ok(12), Some(&fault), &latch).unwrap(), 2);
            fault.after_terms();
        }
        assert!(checked(10, Ok(13), Some(&fault), &latch).is_err());
        assert_eq!(fault.consumed(), 1);
        assert_eq!(checked(10, Ok(14), Some(&fault), &latch).unwrap(), 4);
        assert!(latch.requires_fallback(true));
        let mut fallback = FallbackClock::new();
        assert!(fallback.should_start(&latch, true));
        fallback.arm().unwrap();
        assert!(!fallback.should_start(&latch, true));
        assert!(!fallback.kill_due(None));
        assert!(!fallback.kill_due(Some(std::time::Duration::from_nanos(4_999_999_999))));
        assert!(fallback.kill_due(Some(std::time::Duration::from_secs(5))));
        assert!(!fallback.kill_due(Some(std::time::Duration::from_secs(6))));
        assert!(fallback.arm().is_err());
    }
    let latch = ClockFailureLatch::new();
    assert_eq!(checked(10, Ok(11), None, &latch).unwrap(), 1);
    assert!(!latch.requires_fallback(true));
    assert!(checked(10, Ok(9), None, &latch).is_err());
    assert!(latch.requires_fallback(true));
}
