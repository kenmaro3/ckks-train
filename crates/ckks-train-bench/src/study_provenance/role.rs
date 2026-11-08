//! Closed recipe inspection. Classification is not checked build authority.
use super::{
    ProvenanceError, Result,
    codec::{Capsule, TARGET},
    ensure,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutableRole {
    Reference,
    StudyTrain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Recipe {
    Reference,
    TrainCpuSerial,
    TrainCpuRayon,
    TrainCudaHost,
    ReferenceObserved,
    TrainCpuSerialObserved,
}

impl Recipe {
    pub fn classify(recipe: &str, algorithm: &str) -> Result<Self> {
        let r = match recipe {
            "study-reference-linux-clean-v1" => Self::Reference,
            "study-train-cpu-serial-linux-clean-v1" => Self::TrainCpuSerial,
            "study-train-cpu-rayon-linux-clean-v1" => Self::TrainCpuRayon,
            "study-train-cuda-host-linux-clean-v1" => Self::TrainCudaHost,
            "study-reference-linux-clean-v2" => Self::ReferenceObserved,
            "study-train-cpu-serial-linux-clean-v2" => Self::TrainCpuSerialObserved,
            _ => return Err(ProvenanceError("unsupported recipe")),
        };
        ensure(algorithm == r.algorithm(), "unsupported algorithm")?;
        Ok(r)
    }
    pub fn role(self) -> ExecutableRole {
        if matches!(self, Self::Reference | Self::ReferenceObserved) {
            ExecutableRole::Reference
        } else {
            ExecutableRole::StudyTrain
        }
    }
    pub fn algorithm(self) -> &'static str {
        if matches!(self, Self::Reference | Self::ReferenceObserved) {
            "study-reference-session-v1"
        } else {
            "study-train-session-v1"
        }
    }
    pub fn features(self) -> &'static [&'static str] {
        match self {
            Self::Reference
            | Self::TrainCpuSerial
            | Self::ReferenceObserved
            | Self::TrainCpuSerialObserved => &[],
            Self::TrainCpuRayon => &["parallel"],
            Self::TrainCudaHost => &["cuda", "parallel"],
        }
    }
    pub fn feature(self) -> Option<&'static str> {
        match self {
            Self::TrainCpuRayon => Some("parallel"),
            Self::TrainCudaHost => Some("cuda"),
            _ => None,
        }
    }
    pub fn bin(self) -> &'static str {
        if matches!(self, Self::Reference | Self::ReferenceObserved) {
            "study_reference"
        } else {
            "study_train"
        }
    }
    pub fn helper(self) -> &'static str {
        if matches!(self, Self::Reference | Self::ReferenceObserved) {
            "scripts/study_reference_build.py"
        } else {
            "scripts/study_train_build.py"
        }
    }
    pub fn subcommand(self) -> &'static str {
        match self {
            Self::Reference | Self::ReferenceObserved => "build",
            Self::TrainCpuSerial | Self::TrainCpuSerialObserved => "cpu-serial",
            Self::TrainCpuRayon => "cpu-rayon",
            Self::TrainCudaHost => "cuda-host",
        }
    }
    pub(super) fn capsule_policy(c: &Capsule) -> Result<Self> {
        let r = Self::classify(&c.recipe, &c.algorithm)?;
        ensure(
            c.target == TARGET
                && c.host == TARGET
                && c.profile == "release"
                && c.opt == "3"
                && c.debug == 0
                && c.features
                    .iter()
                    .map(String::as_str)
                    .eq(r.features().iter().copied()),
            "unsupported compile facts",
        )?;
        ensure(
            c.compiler
                .lines()
                .filter_map(|s| s.strip_prefix("host: "))
                .eq([TARGET]),
            "compiler host",
        )?;
        ensure(valid_commit(&c.commit), "Git revision")?;
        Ok(r)
    }
}
pub(super) fn valid_commit(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
