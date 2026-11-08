//! Finite literal expectations, never native facts or production authority.
use super::schedule::{Action as A, Event as E, Reason, Scheduler, Token};
const W: Token = Token {
    pid: 11,
    start: 101,
};
const P: Token = Token {
    pid: 12,
    start: 102,
};
const G: Token = Token {
    pid: 13,
    start: 103,
};
const TEN: u64 = 10_000_000_000;
#[test]
fn operational_cleanup_clock_fallback_never_rebases_or_uses_invalid_duration() {
    use super::schedule::FallbackClock;
    use std::time::Duration;
    let mut f = FallbackClock::new();
    assert!(!f.kill_due(Some(Duration::from_secs(6))));
    f.arm().unwrap();
    assert!(f.arm().is_err());
    assert!(!f.kill_due(None));
    assert!(!f.kill_due(Some(Duration::from_nanos(4_999_999_999))));
    assert!(f.kill_due(Some(Duration::from_secs(5))));
    assert!(!f.kill_due(Some(Duration::from_nanos(5_000_000_001))));
}
#[test]
fn operational_action_failures_preserve_remaining_owned_actions() {
    use super::schedule::apply_all;
    for actions in [
        vec![A::Term(W), A::Term(P), A::Term(G)],
        vec![A::Kill(W), A::Kill(P), A::Kill(G)],
        vec![A::Reap(W), A::Term(P), A::Term(G)],
    ] {
        for failure in [0, 1] {
            let mut seen = Vec::new();
            let errors = apply_all(actions.clone(), |a| {
                let at = seen.len();
                seen.push(a);
                if at == failure {
                    Err(crate::study_provenance::ProvenanceError(
                        "literal failed owned action",
                    ))
                } else {
                    Ok(())
                }
            });
            assert_eq!(seen, actions);
            assert_eq!(errors.len(), 1);
            assert_eq!(errors[0].0, actions[failure]);
        }
    }
}
#[test]
fn operational_phase_sample_preserves_due_window() {
    let mut s = running();
    assert_eq!(
        s.event(
            E::PhaseSample {
                started: 2,
                finished: 3,
                rss: Some(41)
            },
            3
        )
        .unwrap(),
        []
    );
    assert_eq!(s.due(), TEN);
    assert_eq!(s.peak(), Some(41));
    assert_eq!(
        s.event(
            E::Sample {
                started: TEN,
                finished: 2 * TEN,
                rss: Some(40)
            },
            2 * TEN
        )
        .unwrap(),
        []
    );
    assert_eq!(s.due(), 2 * TEN);
    assert_eq!(
        s.event(
            E::PhaseSample {
                started: 2 * TEN,
                finished: 2 * TEN + 1,
                rss: None
            },
            2 * TEN + 1
        )
        .unwrap(),
        [A::Term(W)]
    );
    assert_eq!(s.due(), 2 * TEN);
}
#[test]
fn operational_partial_spawn_never_becomes_empty_custody() {
    use super::schedule::SpawnProgress;
    for fault in 0..5 {
        let mut p = SpawnProgress::spawned();
        for _ in 0..fault {
            p.advance().unwrap();
        }
        p.fail();
        assert!(p.unresolved());
        assert!(!p.ready());
        assert!(p.advance().is_err());
    }
    let mut p = SpawnProgress::spawned();
    for _ in 0..4 {
        p.advance().unwrap();
    }
    assert!(p.ready());
    assert!(!p.unresolved());
}
#[test]
fn operational_helper_success_reaps_without_stopping_worker() {
    let mut s = running();
    s.event(E::Register(P), 2).unwrap();
    assert_eq!(s.event(E::HelperComplete(P), 3).unwrap(), [A::Reap(P)]);
    assert_eq!(s.live(), [W]);
    assert_eq!(s.reason(), None);
    assert!(s.expand(false).is_ok());
    assert!(s.event(E::HelperComplete(W), 4).is_err());
}
#[test]
fn operational_marker_names_are_exact_and_bounded() {
    use super::markers::{MarkerKind, marker_name};
    let id = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    assert_eq!(
        marker_name(format!("{id}.pending.bin").as_bytes()).unwrap(),
        (id, MarkerKind::Pending)
    );
    assert_eq!(
        marker_name(format!("{id}.closed.bin").as_bytes()).unwrap(),
        (id, MarkerKind::Closed)
    );
    for bad in [
        "execution.lock",
        "../x.pending.bin",
        "A123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.pending.bin",
        "x.closed.bin",
    ] {
        assert!(marker_name(bad.as_bytes()).is_err());
    }
    assert!(marker_name(format!("{id}.pending.bin.tmp").as_bytes()).is_err());
}
fn waiting() -> Scheduler {
    let mut s = Scheduler::new();
    assert_eq!(s.event(E::Pending, 0).unwrap(), []);
    assert_eq!(s.event(E::Register(W), 0).unwrap(), []);
    assert_eq!(
        s.event(
            E::Sample {
                started: 0,
                finished: 1,
                rss: Some(40)
            },
            1
        )
        .unwrap(),
        []
    );
    s
}
fn running() -> Scheduler {
    let mut s = waiting();
    assert_eq!(s.event(E::ClaimDurable, 1).unwrap(), [A::Run(W)]);
    s
}
#[test]
fn d7_supervisor_s01_equal_threshold() {
    let mut s = running();
    assert_eq!(
        s.event(
            E::Sample {
                started: TEN,
                finished: TEN + 1,
                rss: Some(94_371_840)
            },
            TEN + 1
        )
        .unwrap(),
        []
    );
    assert_eq!(s.peak(), Some(94_371_840));
    assert_eq!(s.due(), 2 * TEN);
}
#[test]
fn d7_supervisor_s02_threshold_plus_one() {
    let mut s = running();
    assert_eq!(
        s.event(
            E::Sample {
                started: TEN,
                finished: TEN + 1,
                rss: Some(94_371_841)
            },
            TEN + 1
        )
        .unwrap(),
        [A::Term(W)]
    );
    s.term_batch_completed(TEN + 1).unwrap();
    assert_eq!(s.event(E::Tick, TEN + 5_000_000_000).unwrap(), []);
    assert_eq!(s.event(E::Tick, TEN + 5_000_000_001).unwrap(), [A::Kill(W)]);
    assert_eq!(s.reason(), Some(Reason::RssStop));
}
#[test]
fn d7_supervisor_s03_invalid_sample() {
    let mut s = running();
    assert_eq!(
        s.event(
            E::Sample {
                started: TEN,
                finished: TEN + 1,
                rss: None
            },
            TEN + 1
        )
        .unwrap(),
        [A::Term(W)]
    );
    assert_eq!(s.peak(), Some(40));
    assert_eq!(s.reason(), Some(Reason::InvalidSample));
}
#[test]
fn d7_supervisor_s04_late_sample() {
    let mut s = running();
    assert_eq!(
        s.event(
            E::Sample {
                started: TEN,
                finished: 2 * TEN,
                rss: Some(40)
            },
            2 * TEN
        )
        .unwrap(),
        []
    );
    assert_eq!(s.due(), 2 * TEN);
    let mut s = running();
    assert_eq!(
        s.event(
            E::Sample {
                started: TEN,
                finished: 2 * TEN + 1,
                rss: Some(40)
            },
            2 * TEN + 1
        )
        .unwrap(),
        [A::Term(W)]
    );
    let mut s = running();
    assert_eq!(s.event(E::Tick, 2 * TEN + 1).unwrap(), [A::Term(W)]);
}
#[test]
fn d7_supervisor_s05_worker_prestart_loss() {
    let mut s = running();
    s.event(E::Register(P), 2).unwrap();
    assert_eq!(s.event(E::Exited(W), 3).unwrap(), [A::Reap(W), A::Term(P)]);
    assert_eq!(s.reason(), Some(Reason::PrestartRefusal));
}
#[test]
fn d7_supervisor_s06_worker_poststart_loss() {
    let mut s = running();
    s.event(E::StartVisible, 2).unwrap();
    assert_eq!(s.event(E::Exited(W), 3).unwrap(), [A::Reap(W)]);
    assert_eq!(s.reason(), Some(Reason::WorkerFailure));
}
#[test]
fn d7_supervisor_s07_supervisor_eof() {
    let mut s = running();
    assert_eq!(s.event(E::ParentEof, 3).unwrap(), [A::StopBoundary]);
    assert!(!s.can_close());
}
#[test]
fn d7_supervisor_s08_python_loss() {
    let mut s = running();
    s.event(E::Register(P), 2).unwrap();
    s.event(E::Register(G), 2).unwrap();
    assert_eq!(
        s.event(E::Exited(P), 3).unwrap(),
        [A::Reap(P), A::Term(W), A::Term(G)]
    );
}
#[test]
fn d7_supervisor_s09_git_loss() {
    let mut s = running();
    s.event(E::Register(P), 2).unwrap();
    s.event(E::Register(G), 2).unwrap();
    assert_eq!(
        s.event(E::HelperFailed(G), 3).unwrap(),
        [A::Term(W), A::Term(P), A::Term(G)]
    );
    assert_eq!(s.reason(), Some(Reason::ProtocolFailure));
}
#[test]
fn d7_supervisor_s10_reparent() {
    let mut s = running();
    assert_eq!(s.event(E::KnownReparent(W), 3).unwrap(), []);
    assert_eq!(s.live(), vec![W]);
}
#[test]
fn d7_supervisor_s11_unknown_child() {
    let mut s = running();
    assert_eq!(s.event(E::UnknownChild, 3).unwrap(), [A::Term(W)]);
    s.event(E::Exited(W), 4).unwrap();
    assert!(!s.can_close());
    assert_eq!(s.reason(), Some(Reason::UnresolvedChildren));
}
#[test]
fn d7_supervisor_s12_pid_reuse() {
    let mut s = running();
    let other = Token {
        pid: W.pid,
        start: W.start + 1,
    };
    assert!(s.event(E::KnownReparent(other), 3).is_err());
    assert!(s.event(E::Register(other), 3).is_err());
    assert_eq!(s.event(E::IdentityLost(W), 3).unwrap(), []);
    assert!(!s.can_close());
}
#[test]
fn d7_supervisor_s13_cap_write_failure() {
    let mut s = running();
    assert_eq!(s.event(E::PublicationFailed, 3).unwrap(), [A::Term(W)]);
    assert_eq!(s.reason(), Some(Reason::PublicationFailure));
    s.event(E::Exited(W), 4).unwrap();
    assert!(!s.can_close());
}
#[test]
fn d7_supervisor_s14_teardown() {
    let mut s = running();
    s.event(E::StartVisible, 2).unwrap();
    assert_eq!(s.event(E::WorkerComplete, 3).unwrap(), []);
    assert!(!s.can_close());
    assert_eq!(s.event(E::Exited(W), 4).unwrap(), [A::Reap(W)]);
    assert!(!s.can_close());
    s.event(E::TeardownSample, 5).unwrap();
    assert!(s.can_close());
    assert_eq!(s.event(E::EndDurable, 6).unwrap(), [A::PublishClosed]);
}
#[test]
fn d7_supervisor_w23_claim_before_expansion() {
    let mut s = waiting();
    assert!(s.expand(false).is_err());
    assert_eq!(s.event(E::ClaimDurable, 1).unwrap(), [A::Run(W)]);
    s.expand(false).unwrap();
    assert!(s.expand(true).is_err());
    s.event(E::RequestBound, 2).unwrap();
    s.expand(true).unwrap();
}
#[test]
fn d7_supervisor_w24_unconfirmed_claim_refuses_expansion() {
    let mut s = waiting();
    s.event(E::ClaimVisible, 1).unwrap();
    assert!(s.expand(false).is_err());
    s.event(E::PublicationFailed, 2).unwrap();
    assert!(s.expand(false).is_err());
}

#[test]
fn operational_cleanup_wait_never_spins_on_eof_or_queued_output() {
    use super::schedule::WaitInterest;
    for out_eof in [false, true] {
        for err_eof in [false, true] {
            for pending in [false, true] {
                let stop = WaitInterest::new(true, out_eof, err_eof, pending);
                assert_eq!(
                    (stop.stdout, stop.stderr, stop.writable, stop.park),
                    (false, false, false, true)
                );
                let run = WaitInterest::new(false, out_eof, err_eof, pending);
                assert_eq!(
                    (run.stdout, run.stderr, run.writable, run.park),
                    (!out_eof, !err_eof, pending, false)
                );
            }
        }
    }
}

#[test]
fn operational_capacity_filesystem_raw_units_are_preserved() {
    use super::capacity::disk_available;
    assert_eq!(disk_available(4096, 2, 3, 8192, 3).unwrap(), 8192);
    for args in [
        (0, 2, 3, 0, 0),
        (u64::MAX, 2, 3, 0, 0),
        (4096, 2, 3, 8193, 3),
        (4096, 2, 3, 8192, 4),
    ] {
        assert!(disk_available(args.0, args.1, args.2, args.3, args.4).is_err());
    }
}

#[test]
fn operational_native_entry_has_closed_sixteen_plans() {
    use super::probe_plan::Selection;
    let ids = [
        "pidfd-owned-unreaped",
        "subreaper-known-unknown",
        "lock-ofd-inheritance",
        "worker-handshake-exec",
        "python-git-direct-broker",
        "git-no-descendants",
        "parent-eof-custody",
        "nice-ten",
        "monotonic-common-origin",
        "rss-ten-second-deadline",
        "rss-term-grace-kill",
        "proc-namespace-cgroup",
        "storage-mount-quota",
        "publisher-no-clobber-fsync",
        "crash-pending-block",
        "resolved-closure",
    ];
    for (index, id) in ids.into_iter().enumerate() {
        let args = [
            "--native-supervisor-check",
            id,
            "--probe-origin",
            "/unit/probe",
            "--probe-origin-sha256",
            &"a".repeat(64),
        ]
        .map(std::ffi::OsString::from);
        let parsed = Selection::parse(&args).unwrap();
        assert_eq!(parsed.index, index);
        assert_eq!(parsed.check, id);
        for i in 0..args.len() {
            let mut bad = args.clone();
            bad[i] = "invalid".into();
            assert!(Selection::parse(&bad).is_err());
        }
        let mut extra = args.to_vec();
        extra.push("--extra".into());
        assert!(Selection::parse(&extra).is_err());
        for end in 0..args.len() {
            assert!(Selection::parse(&args[..end]).is_err());
        }
    }
}

#[test]
fn operational_native_transfer_requires_exact_session_rights_and_ack() {
    use super::{
        handshake::Projection,
        probe_transfer::{Handoff, TransferProjection},
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
        lock: (2, 3),
        origin: 200,
        boot: *b"01234567-89ab-cdef-0123-456789abcdef",
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
    let mut raw = b"HKFDS1".to_vec();
    raw.extend([1, 2]);
    raw.extend(10u64.to_le_bytes());
    raw.extend(100u64.to_le_bytes());
    raw.extend(inner.encode());
    raw.extend([b'b'; 64]);
    let decoded = TransferProjection::decode(&raw, &outer, 1).unwrap();
    assert_eq!(decoded.inner, inner);
    assert_eq!(decoded.encode(), raw);
    for end in 0..raw.len() {
        assert!(TransferProjection::decode(&raw[..end], &outer, 1).is_err());
    }
    for at in [0, 6, 7, 8, 16, 24 + 6, 24 + 62, 24 + 98, raw.len() - 1] {
        let mut bad = raw.clone();
        bad[at] ^= 1;
        assert!(TransferProjection::decode(&bad, &outer, 1).is_err());
    }
    assert!(TransferProjection::decode(&raw, &outer, 6).is_err());
    let mut h = Handoff::new();
    assert!(!h.can_wait());
    assert!(h.ack().is_err());
    assert!(h.custody_unresolved());
    let mut h = Handoff::new();
    h.sent().unwrap();
    assert!(h.received(3, 2, false, false).is_err());
    assert!(h.custody_unresolved());
    for (count, truncated, duplicate) in [(1, false, false), (2, true, false), (2, false, true)] {
        let mut h = Handoff::new();
        h.sent().unwrap();
        assert!(h.received(count, 2, truncated, duplicate).is_err());
        assert!(h.custody_unresolved());
    }
    let mut h = Handoff::new();
    h.sent().unwrap();
    h.received(2, 2, false, false).unwrap();
    assert!(!h.can_wait());
    assert!(h.custody_unresolved());
    h.ack().unwrap();
    assert!(!h.can_wait());
    assert!(h.adopted(false).is_err());
    assert!(h.custody_unresolved());
    let mut h = Handoff::new();
    h.sent().unwrap();
    h.received(2, 2, false, false).unwrap();
    h.ack().unwrap();
    h.adopted(true).unwrap();
    assert!(h.can_wait());
    h.reaped().unwrap();
    assert!(!h.custody_unresolved());
    assert!(h.reaped().is_err());
}

#[test]
fn operational_transferred_pidfd_proc_fields_reject_missing_duplicate_reused() {
    use super::observation::pid_field;
    assert_eq!(pid_field(b"Pid:\t12\nNSpid:\t12\n", b"Pid:").unwrap(), 12);
    assert_eq!(
        pid_field(b"Name:\tleaf\nPPid:\t10\n", b"PPid:").unwrap(),
        10
    );
    for raw in [
        b"NSpid: 12\n".as_slice(),
        b"Pid: 12\nPid: 12\n",
        b"Pid: -1\n",
        b"Pid: 2147483648\n",
        b"Pid: 0\n",
        b"Pid: 12 extra\n",
    ] {
        assert!(pid_field(raw, b"Pid:").is_err());
    }
}

#[test]
fn operational_stop_grace_starts_after_last_term_attempt_once() {
    let mut s = running();
    assert_eq!(
        s.event(
            E::Sample {
                started: 10_000_000_000,
                finished: 10_000_000_000,
                rss: Some(super::schedule::RSS_LIMIT + 1)
            },
            10_000_000_000
        )
        .unwrap(),
        [A::Term(W)]
    );
    let mut unacked = running();
    unacked
        .event(
            E::Sample {
                started: 10_000_000_000,
                finished: 10_000_000_000,
                rss: Some(super::schedule::RSS_LIMIT + 1),
            },
            10_000_000_000,
        )
        .unwrap();
    assert_eq!(unacked.event(E::Tick, 15_000_000_000).unwrap(), []);
    s.term_batch_completed(10_000_000_001).unwrap();
    assert!(s.term_batch_completed(12_000_000_000).is_err());
    assert_eq!(s.event(E::Tick, 15_000_000_000).unwrap(), []);
    assert_eq!(s.event(E::Tick, 15_000_000_001).unwrap(), [A::Kill(W)]);
}

#[test]
fn operational_post_term_clock_loss_survives_read_recovery() {
    let latch = super::schedule::ClockFailureLatch::new();
    assert!(!latch.requires_fallback(true));
    assert!(latch.requires_fallback(false));
    assert!(latch.requires_fallback(true));
    assert!(latch.requires_fallback(true));
    let post_term = super::schedule::ClockFailureLatch::new();
    post_term.failed();
    assert!(post_term.requires_fallback(true));
}

#[test]
fn operational_native_transfer_boundary_rejects_rescue_as_custody() {
    use super::probe_transfer::Boundary;
    for (boundary, count, marker) in [
        (Boundary::RescueToMiddle, 1, b'M'),
        (Boundary::RescueToLeaf, 1, b'L'),
        (Boundary::CustodyToOuter, 2, b'C'),
    ] {
        assert_eq!(boundary.rights(), count);
        assert_eq!(boundary.marker(), marker);
        assert!(boundary.validate(6, marker, count, false).is_ok());
        for wrong in [b'M', b'L', b'C'] {
            if wrong != marker {
                assert!(boundary.validate(6, wrong, count, false).is_err());
            }
        }
        assert!(boundary.validate(6, marker, count + 1, false).is_err());
        assert!(boundary.validate(6, marker, count, true).is_err());
        assert_eq!(
            boundary.validate(1, marker, count, false).is_ok(),
            matches!(boundary, Boundary::CustodyToOuter)
        );
        assert!(boundary.validate(2, marker, count, false).is_err());
    }
}

#[test]
fn operational_spawn_reservation_refuses_after_due_stop_and_commits_without_clock() {
    let mut s = Scheduler::new();
    assert!(s.reserve().is_err());
    s.event(E::Pending, 0).unwrap();
    {
        let _reservation = s.reserve().unwrap();
    }
    assert!(s.live().is_empty());
    s.reserve().unwrap().commit(W);
    s.reserve().unwrap().commit(P);
    s.reserve().unwrap().commit(G);
    assert_eq!(s.live(), [W, P, G]);
    assert!(s.reserve().is_err());
    for missing in [false, true] {
        let mut s = running();
        s.event(
            E::Sample {
                started: TEN,
                finished: TEN,
                rss: if missing {
                    None
                } else {
                    Some(super::schedule::RSS_LIMIT + 1)
                },
            },
            TEN,
        )
        .unwrap();
        assert!(s.reserve().is_err());
    }
}
#[test]
fn operational_final_teardown_unknown_or_clock_loss_keeps_hold() {
    use super::schedule::closure_must_hold;
    assert!(!closure_must_hold(false, 0, false, false));
    for state in [
        (true, 0, false, false),
        (false, 1, false, false),
        (false, 0, true, false),
        (false, 0, false, true),
    ] {
        assert!(closure_must_hold(state.0, state.1, state.2, state.3));
    }
    let mut s = running();
    s.event(E::Exited(W), 1).unwrap();
    assert!(s.live().is_empty());
    s.event(E::UnknownChild, 2).unwrap();
    s.event(E::TeardownSample, 3).unwrap();
    assert!(!s.can_close());
}

#[test]
fn operational_native_parent_sample_cannot_mint_worker_claim() {
    let mut production = Scheduler::new();
    production.event(E::Pending, 0).unwrap();
    assert!(
        production
            .event(
                E::Sample {
                    started: 0,
                    finished: 1,
                    rss: Some(7)
                },
                1
            )
            .is_err()
    );
    let mut probe = Scheduler::new_probe();
    probe.event(E::Pending, 0).unwrap();
    probe
        .event(
            E::Sample {
                started: 0,
                finished: 1,
                rss: Some(7),
            },
            1,
        )
        .unwrap();
    assert_eq!(probe.due(), TEN);
    assert!(probe.event(E::ClaimDurable, 2).is_err());
    assert!(probe.expand(false).is_err());
    probe.reserve().unwrap().commit(W);
    probe
        .event(
            E::PhaseSample {
                started: 3,
                finished: 4,
                rss: Some(9),
            },
            4,
        )
        .unwrap();
    assert_eq!(probe.due(), TEN);
}

#[test]
fn operational_native_reap_cannot_complete_production_worker() {
    let mut ordinary = waiting();
    assert!(ordinary.probe_retire(W).is_err());
    let mut probe = Scheduler::new_probe();
    probe.event(E::Pending, 0).unwrap();
    probe.event(E::Register(W), 1).unwrap();
    assert_eq!(probe.probe_retire(W).unwrap(), [A::Reap(W)]);
    assert_eq!(probe.reason(), None);
    assert!(probe.live().is_empty());
    assert!(probe.probe_retire(W).is_err());
    assert!(probe.expand(false).is_err());
    let mut probe = Scheduler::new_probe();
    probe.event(E::Pending, 0).unwrap();
    probe.event(E::Register(W), 1).unwrap();
    probe
        .event(
            E::Sample {
                started: 1,
                finished: 2,
                rss: Some(1),
            },
            2,
        )
        .unwrap();
    probe.event(E::ClaimDurable, 2).unwrap();
    assert!(probe.probe_retire(W).is_err());
}

#[test]
fn operational_resource_children_follow_fixed_root_graph() {
    use crate::study_reference::operational::Requested;
    for root in 3..=5 {
        let w = Requested::preparation(root, "iris__emlp1", &[0]).unwrap();
        assert!(w.child(root, 2, "iris__emlp1", 0).is_ok());
        assert!(w.child(root, 1, "iris__emlp1", 0).is_err());
        assert!(w.child(root, 2, "iris__emlp2", 0).is_err());
        assert!(w.child(root, 2, "iris__emlp1", 1).is_err());
    }
    let full = Requested::train(true, false).unwrap();
    assert!(full.child(6, 3, "iris__emlp1", 0).is_ok());
    assert!(full.child(3, 2, "iris__emlp1", 0).is_ok());
    assert!(full.child(6, 4, "iris__emlp1", 0).is_err());
    let quick = Requested::train(true, true).unwrap();
    assert!(quick.child(6, 3, "iris__emlp1", 0).is_err());
    assert!(quick.child(3, 2, "iris__emlp1", 0).is_err());
    let selected = Requested::preparation(3, "iris__emlp1", &[0, 2]).unwrap();
    assert!(selected.child(3, 2, "iris__emlp1", 2).is_ok());
    assert!(selected.child(3, 2, "iris__emlp1", 1).is_err());
}

#[test]
fn operational_fix2_live_control_eof_stops_owned_children() {
    use super::protocol::ControlObservation;
    let (local, peer) = std::os::unix::net::UnixStream::pair().unwrap();
    peer.shutdown(std::net::Shutdown::Write).unwrap();
    let mut channel = super::protocol::Channel::new(local).unwrap();
    assert!(channel.turn().unwrap().is_empty());
    assert!(channel.eof && !channel.pending());
    let live = ControlObservation {
        kind: 0,
        phase: 4,
        result_present: false,
        write_pending: channel.pending(),
        eof: channel.eof,
        status: None,
    };
    let setup = || {
        let mut s = running();
        s.event(E::Register(P), 2).unwrap();
        s.event(E::Register(G), 2).unwrap();
        s
    };
    for bad in [
        live,
        ControlObservation { kind: 1, ..live },
        ControlObservation {
            kind: 2,
            phase: 3,
            ..live
        },
        ControlObservation { phase: 1, ..live },
        ControlObservation {
            phase: 5,
            write_pending: true,
            ..live
        },
        ControlObservation {
            kind: 1,
            phase: 5,
            ..live
        },
        ControlObservation {
            kind: 2,
            phase: 5,
            result_present: true,
            ..live
        },
        ControlObservation {
            kind: 1,
            phase: 4,
            result_present: true,
            ..live
        },
    ] {
        let mut s = setup();
        assert_eq!(
            s.observe_control(W, bad, 3).unwrap(),
            vec![A::Term(W), A::Term(P), A::Term(G)]
        );
        assert_eq!(s.reason(), Some(Reason::ProtocolFailure));
        assert_eq!(s.live(), vec![W, P, G]); // EOF is never wait/reap.
        assert!(s.observe_control(W, bad, 4).unwrap().is_empty());
        s.term_batch_completed(5).unwrap();
        assert!(s.event(E::Tick, 5 + 4_999_999_999).unwrap().is_empty());
        assert_eq!(
            s.event(E::Tick, 5 + 5_000_000_000).unwrap(),
            vec![A::Kill(W), A::Kill(P), A::Kill(G)]
        );
    }
    for allowed in [
        ControlObservation { eof: false, ..live },
        ControlObservation {
            status: Some(0),
            ..live
        },
        ControlObservation {
            status: Some(31),
            ..live
        },
        ControlObservation { kind: 2, ..live },
        ControlObservation {
            kind: 1,
            phase: 5,
            result_present: true,
            ..live
        },
        ControlObservation { phase: 5, ..live },
    ] {
        let mut s = setup();
        assert!(s.observe_control(W, allowed, 3).unwrap().is_empty());
        assert_eq!(s.reason(), None);
        assert_eq!(s.live(), vec![W, P, G]);
    }
    assert!(
        setup()
            .observe_control(
                Token {
                    pid: 99,
                    start: 999
                },
                live,
                3
            )
            .is_err()
    );
}
