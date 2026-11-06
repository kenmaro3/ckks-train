//! Synthetic engines exercise the real private driver, never a production result.
use super::*;
use crate::study_profile::{Checkpoint, PlannedUpdate};
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
};
struct ScriptClock {
    ticks: RefCell<VecDeque<u64>>,
    wall: Cell<u64>,
}
impl Clock for ScriptClock {
    fn now(&self) -> Result<u64> {
        self.ticks
            .borrow_mut()
            .pop_front()
            .ok_or(SessionError("script_exhausted"))
    }
    fn wall(&self) -> Result<u64> {
        Ok(self.wall.get())
    }
}
fn clock(ticks: &[u64]) -> Timer<ScriptClock> {
    Timer::new(ScriptClock {
        ticks: RefCell::new(ticks.iter().copied().collect()),
        wall: Cell::new(100),
    })
    .unwrap()
}
struct SyntheticEngine {
    n: usize,
    step_fail: bool,
    eval_fail_at: usize,
    poison: bool,
    wrong_step: bool,
}
impl Engine for SyntheticEngine {
    fn progress(&self) -> SessionProgress {
        SessionProgress {
            updates: self.n,
            samples: self.n * 2,
        }
    }
    fn poisoned(&self) -> bool {
        self.poison
    }
    fn step(&mut self) -> Result<PlainStep> {
        if self.step_fail {
            self.poison = true;
            return Err(SessionError("synthetic_step"));
        }
        let i = self.n;
        self.n += 1;
        Ok(PlainStep {
            update_index: if self.wrong_step { 99 } else { i },
            valid_samples: 2,
            processed_samples: self.n * 2,
        })
    }
    fn evaluate(&self) -> Result<PlainEvaluation> {
        if self.n == self.eval_fail_at {
            Err(SessionError("synthetic_evaluation"))
        } else {
            Ok(PlainEvaluation {
                correct: 1,
                test_count: 3,
            })
        }
    }
}
fn engine() -> SyntheticEngine {
    SyntheticEngine {
        n: 0,
        step_fail: false,
        eval_fail_at: usize::MAX,
        poison: false,
        wrong_step: false,
    }
}
fn plan(checkpoints: &[usize]) -> BatchPlan {
    let mut prev = 0;
    BatchPlan {
        updates: (0..3)
            .map(|index| PlannedUpdate {
                index,
                processed_start: 2 * index,
                valid_samples: 2,
                source_schedule_index: None,
                source_coefficient: 0.,
                mean_gradient_rate: 0.,
                delta_multiplier: 0.,
            })
            .collect(),
        checkpoints: checkpoints
            .iter()
            .map(|&n| {
                let c = Checkpoint {
                    completed_updates: n,
                    processed_samples: 2 * n,
                    crossed_logical_boundaries: vec![n],
                    exact_logical_boundary: true,
                    interval_updates: n - prev,
                    interval_samples: 2 * (n - prev),
                };
                prev = n;
                c
            })
            .collect(),
        processed_samples: 6,
        dropped_samples: 0,
    }
}
#[test]
fn disjoint_terminal_quick_eval_and_caller_gaps() {
    let p = plan(&[3]);
    let mut d = Driver::new(
        engine(),
        &p,
        3,
        2,
        clock(&[0, 3, 4, 9, 10, 15, 20, 23, 23, 30, 30, 34, 35]),
    )
    .unwrap();
    d.advance().unwrap();
    d.advance().unwrap();
    assert_eq!(d.prepare_finish(false).unwrap().test_count, 3);
    let t = d.complete().unwrap();
    assert_eq!(
        (t.setup, t.update, t.evaluation, t.finalization, t.total),
        (3, 10, 7, 7, 35)
    );
    assert_eq!(d.endpoints.len(), 0);
}
#[test]
fn reused_final_checkpoint_has_no_new_evaluation() {
    let mut p = plan(&[1, 2]);
    p.updates.truncate(2);
    p.processed_samples = 4;
    let mut d = Driver::new(
        engine(),
        &p,
        3,
        2,
        clock(&[0, 3, 4, 9, 10, 17, 18, 23, 24, 31, 40, 45, 46]),
    )
    .unwrap();
    d.advance().unwrap();
    d.advance().unwrap();
    d.prepare_finish(true).unwrap();
    let t = d.complete().unwrap();
    assert_eq!(
        (t.setup, t.update, t.evaluation, t.finalization, t.total),
        (3, 10, 14, 5, 46)
    );
    assert_eq!(
        (d.endpoints[0].update_ns, d.endpoints[0].evaluation_ns),
        (5, 7)
    );
    assert_eq!(
        (d.endpoints[1].update_ns, d.endpoints[1].evaluation_ns),
        (10, 14)
    );
    struct SyntheticReference;
    let _synthetic_only = SyntheticReference;
}
#[test]
fn failed_terminal_evaluation_charges_only_call_and_prior_finalization() {
    let p = plan(&[3]);
    let mut e = engine();
    e.eval_fail_at = 2;
    let mut d = Driver::new(
        e,
        &p,
        3,
        2,
        clock(&[0, 3, 4, 9, 10, 15, 20, 23, 23, 30, 31]),
    )
    .unwrap();
    d.advance().unwrap();
    d.advance().unwrap();
    assert!(d.prepare_finish(false).is_err());
    let t = d.failure_timing.as_ref().unwrap();
    assert_eq!(
        (t.setup, t.update, t.evaluation, t.finalization, t.total),
        (3, 10, 7, 3, 31)
    );
    assert!(d.endpoints.is_empty());
    assert!(d.prepare_finish(false).is_err());
}
#[test]
fn failed_checkpoint_after_update_retains_prior_success_snapshot() {
    let p = plan(&[1, 2, 3]);
    let mut e = engine();
    e.eval_fail_at = 2;
    let mut d = Driver::new(
        e,
        &p,
        3,
        3,
        clock(&[0, 3, 4, 9, 10, 17, 18, 23, 24, 31, 32]),
    )
    .unwrap();
    d.advance().unwrap();
    assert!(d.advance().is_err());
    assert_eq!(d.engine.progress().updates, 2);
    let t = d.failure_timing.as_ref().unwrap();
    assert_eq!(
        (t.update, t.evaluation, t.finalization, t.total),
        (10, 14, 0, 32)
    );
    assert_eq!(d.endpoints.len(), 1);
    assert_eq!(
        (d.endpoints[0].update_ns, d.endpoints[0].evaluation_ns),
        (5, 7)
    );
    assert!(d.prepare_finish(true).is_err());
}
fn increasing_clock() -> Timer<ScriptClock> {
    clock(&(0..200).collect::<Vec<_>>())
}
#[test]
fn prefixes_exhaustion_replay_poison_and_poststep_are_terminal() {
    let p = plan(&[1, 2, 3]);
    for n in 0..3 {
        let mut d = Driver::new(engine(), &p, 3, 3, increasing_clock()).unwrap();
        for _ in 0..n {
            d.advance().unwrap();
        }
        assert!(d.prepare_finish(true).is_err());
        assert!(d.advance().is_err());
    }
    let mut d = Driver::new(engine(), &p, 3, 3, increasing_clock()).unwrap();
    for _ in 0..3 {
        d.advance().unwrap();
    }
    assert!(d.advance().is_err());
    assert!(d.prepare_finish(true).is_err());
    assert_eq!(d.engine.n, 3);
    for fault in 0..4 {
        let mut d = Driver::new(engine(), &p, 3, 3, increasing_clock()).unwrap();
        match fault {
            0 => d.engine.step_fail = true,
            1 => d.engine.poison = true,
            2 => d.engine.wrong_step = true,
            _ => d.engine.n = 1,
        };
        assert!(d.advance().is_err());
        assert!(d.prepare_finish(true).is_err());
        assert!(d.advance().is_err());
    }
}
#[test]
fn failure_clock_never_fabricates_zero_timing() {
    let p = plan(&[3]);
    let mut d = Driver::new(engine(), &p, 3, 2, clock(&[0, 3, 5, 4, 6])).unwrap();
    assert!(d.advance().is_err());
    assert!(d.failure_timing.is_none());
}
#[test]
fn full_synthetic_run_produces_only_private_synthetic_marker() {
    let p = plan(&[1, 2, 3]);
    let mut d = Driver::new(engine(), &p, 3, 3, increasing_clock()).unwrap();
    for _ in 0..3 {
        d.advance().unwrap();
    }
    let obs = d.prepare_finish(true).unwrap();
    let timing = d.complete().unwrap();
    struct SyntheticResult {
        counts: PlainEvaluation,
        timing: TimingFact,
    }
    let s = SyntheticResult {
        counts: obs,
        timing,
    };
    assert_eq!(s.counts.test_count, 3);
    assert!(s.timing.total > 0);
}
#[test]
fn failed_setup_retains_observed_interval_and_invalid_wall_is_unavailable() {
    let mut timer = clock(&[0, 7, 8]);
    let t = finish_setup_failure(&mut timer).unwrap();
    assert_eq!(
        (t.setup, t.update, t.evaluation, t.finalization, t.total),
        (7, 0, 0, 0, 8)
    );
    let mut timer = clock(&[0, 7, 8]);
    timer.clock.wall.set(99);
    assert!(finish_setup_failure(&mut timer).is_none());
}
#[test]
fn invalid_terminal_wall_never_recovers_on_a_later_read() {
    let mut timer = clock(&[0, 1, 2, 3]);
    timer.clock.wall.set(99);
    assert!(timer.end().is_err());
    timer.clock.wall.set(101);
    assert!(
        timer.end().is_err(),
        "a later good wall reading must not repair the first failed endpoint"
    );
}

#[test]
fn initial_wall_failure_stops_before_any_setup_capture_or_engine_work() {
    struct InvalidInitialWall<'a> {
        wall_calls: &'a Cell<usize>,
    }
    impl Clock for InvalidInitialWall<'_> {
        fn now(&self) -> Result<u64> {
            Ok(0)
        }
        fn wall(&self) -> Result<u64> {
            self.wall_calls.set(self.wall_calls.get() + 1);
            Err(SessionError("initial_wall_unavailable"))
        }
    }
    let wall_calls = Cell::new(0);
    let setup_calls = Cell::new(0);
    let capture_calls = Cell::new(0);
    let engine_calls = Cell::new(0);
    // This is the same pre-work gate used by Session::start_with, with no real inputs,
    // context, completion or production capability. A rejected gate must not call it.
    let result = prepare_session(
        InvalidInitialWall {
            wall_calls: &wall_calls,
        },
        "initial_clock_failure",
        32,
        "quick",
        || {
            setup_calls.set(setup_calls.get() + 1);
            capture_calls.set(capture_calls.get() + 1);
            engine_calls.set(engine_calls.get() + 1);
            Ok(engine())
        },
    );
    assert_eq!(wall_calls.get(), 1);
    assert_eq!(
        (setup_calls.get(), capture_calls.get(), engine_calls.get()),
        (0, 0, 0),
        "known initial wall failure must prevent all setup work"
    );
    let failure = match result {
        Err(f) => f,
        Ok(_) => panic!("invalid clock admitted setup"),
    };
    assert_eq!(failure.error(), SessionError("initial_wall_unavailable"));
    assert_eq!(failure.detail().phase, "start");
    assert!(failure.detail().timing.is_none());
    assert!(failure.detail().identity.is_none());
    assert!(failure.detail().progress.is_none());
    assert!(failure.detail().last.is_none());
}

#[test]
fn setup_work_failure_keeps_observed_setup_timing() {
    let calls = Cell::new(0);
    let result = prepare_session(
        ScriptClock {
            ticks: RefCell::new([0, 7, 8].into()),
            wall: Cell::new(100),
        },
        "setup_work_failure",
        64,
        "quick",
        || -> Result<()> {
            calls.set(calls.get() + 1);
            Err(SessionError("synthetic_setup_failure"))
        },
    );
    let failure = match result {
        Err(f) => f,
        Ok(_) => panic!("failed setup succeeded"),
    };
    assert_eq!(calls.get(), 1);
    assert_eq!(failure.detail().phase, "start");
    assert_eq!(failure.error(), SessionError("synthetic_setup_failure"));
    let timing = failure.detail().timing.as_ref().unwrap();
    assert_eq!(
        (
            timing.setup,
            timing.update,
            timing.evaluation,
            timing.finalization,
            timing.total
        ),
        (7, 0, 0, 0, 8)
    );
}
