//! Shared sampled outcome and sticky final-teardown acceptance decision.
use crate::{
    study_provenance::{ProvenanceError, Result},
    study_reference::operational::SampleValues,
};
pub(super) fn assessment(
    rss: Option<u64>,
    started: u64,
    finished: u64,
    deadline: u64,
    phase: u8,
) -> (bool, u8) {
    let valid = rss.is_some() && finished >= started && finished <= deadline;
    let status = if !valid {
        2
    } else if rss.is_some_and(|r| r > super::schedule::RSS_LIMIT) {
        1
    } else if phase == 4 {
        4
    } else {
        0
    };
    (valid, status)
}
pub(super) fn final_sample_result(
    sample: Result<SampleValues>,
    unresolved: &mut bool,
) -> Result<SampleValues> {
    match sample {
        Ok(sample) if !*unresolved && sample.phase == 4 && sample.valid && sample.status == 4 => {
            Ok(sample)
        }
        other => {
            *unresolved = true;
            match other {
                Err(error) => Err(error),
                Ok(_) => Err(ProvenanceError("invalid final teardown sample")),
            }
        }
    }
}
