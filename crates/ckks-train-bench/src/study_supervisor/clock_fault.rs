//! Closed native-probe fault seam; ordinary monitors have no fault object.
use crate::study_provenance::{ProvenanceError, Result};
use std::cell::Cell;
pub(super) struct Fault {
    after_terms: bool,
    ready: Cell<bool>,
    consumed: Cell<u8>,
}
impl Fault {
    pub fn new(after_terms: bool) -> Self {
        Self {
            after_terms,
            ready: Cell::new(!after_terms),
            consumed: Cell::new(0),
        }
    }
    pub fn after_terms(&self) {
        if self.after_terms && self.consumed.get() == 0 {
            self.ready.set(true);
        }
    }
    pub fn consumed(&self) -> u8 {
        self.consumed.get()
    }
    fn read(&self, native: Result<u64>) -> Result<u64> {
        if self.ready.replace(false) && self.consumed.get() == 0 {
            self.consumed.set(1);
            Err(ProvenanceError("injected native probe clock read failure"))
        } else {
            native
        }
    }
}
pub(super) fn checked(
    origin: u64,
    native: Result<u64>,
    fault: Option<&Fault>,
    latch: &super::schedule::ClockFailureLatch,
) -> Result<u64> {
    let result = match fault {
        Some(f) => f.read(native),
        None => native,
    }
    .and_then(|now| super::protocol::elapsed(origin, now));
    if result.is_err() {
        latch.failed();
    }
    result
}
