//! Fixed 4096-byte nonauthorizing bootstrap request. No planner/registry objects.
use super::*;
pub(in crate::study_reference::document::evidence) const TRAIN_RECIPE: &str =
    "study-train-cpu-serial-linux-clean-v2";
const PREP_RECIPE: &str = "study-reference-linux-clean-v2";
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct Train<'a> {
    pub(in crate::study_reference::document::evidence) registration: Id<'a>,
    pub(in crate::study_reference::document::evidence) profile: Id<'a>,
    pub(in crate::study_reference::document::evidence) mode: u8,
    pub(in crate::study_reference::document::evidence) request: &'a [u8],
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) enum Work<'a> {
    Train(Train<'a>),
    Preparation {
        operation: u8,
        profile: Id<'a>,
        seeds: [u64; 10],
        count: usize,
    },
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct RequestedWork<'a> {
    pub(in crate::study_reference::document::evidence) mode: Mode,
    pub(in crate::study_reference::document::evidence) body: Work<'a>,
}
pub(in crate::study_reference::document::evidence) struct FixedWork {
    bytes: [u8; 4096],
    len: usize,
}
impl FixedWork {
    pub(in crate::study_reference::document::evidence) fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    fn put(&mut self, b: &[u8]) -> Result<()> {
        let end = self
            .len
            .checked_add(b.len())
            .ok_or_else(|| err(DocumentErrorCode::TooLarge, "work.length"))?;
        ensure(end <= 4096, DocumentErrorCode::TooLarge, "work.length")?;
        self.bytes[self.len..end].copy_from_slice(b);
        self.len = end;
        Ok(())
    }
    fn n(&mut self, n: u64) -> Result<()> {
        self.put(&n.to_le_bytes())
    }
    fn b(&mut self, b: &[u8]) -> Result<()> {
        self.n(b.len() as u64)?;
        self.put(b)
    }
}
fn compiled_request(b: &[u8]) -> Result<u8> {
    same(b.len() == 62, "work.request.length")?;
    let mut c = Cursor::new(b);
    let scope = c.tag(1)?;
    same(c.take(5)? == [0, 0, 1, 1, 1], "work.request.options")?;
    for n in [
        67_108_864,
        1_000_000,
        2,
        2,
        if scope == 0 { 2400 } else { 2 },
        0,
        1,
    ] {
        same(c.u64()? == n, "work.request.compiled")?;
    }
    c.finish()?;
    Ok(scope)
}
impl<'a> RequestedWork<'a> {
    pub(in crate::study_reference::document::evidence) fn decode(
        b: &'a [u8],
        mode: Mode,
    ) -> Result<Self> {
        let mut c = cursor(b, "resource-requested-work", 1, mode, 4096)?;
        let body = if c.tag(1)? == 0 {
            same(c.tag(1)? == 0, "work.origin")?;
            let registration = Id::read(&mut c)?;
            let profile = Id::read(&mut c)?;
            same(
                profile.0 == "iris__emlp1" && c.u64()? == 0,
                "work.profile.seed",
            )?;
            same(c.tag(2)? == 0 && c.tag(2)? == 0, "work.scope.backend")?;
            let m = c.tag(1)?;
            same(c.tag(4)? == m && c.tag(4)? == 0, "work.placement.packing")?;
            same(
                registration.0 == if m == 0 { "main_i" } else { "main_ni" },
                "work.registration",
            )?;
            same(Id::read(&mut c)?.0 == TRAIN_RECIPE, "work.runtime")?;
            let request = c.bounded_bytes(62)?;
            compiled_request(request)?;
            same(c.tag(0)? == 0, "work.partitions")?;
            Work::Train(Train {
                registration,
                profile,
                mode: m,
                request,
            })
        } else {
            let operation = c.tag(5)?;
            let profile = Id::read(&mut c)?;
            let count = c.bounded_count(10, 8)?;
            same(count > 0, "work.seeds.empty")?;
            let mut seeds = [0; 10];
            for i in 0..count {
                let seed = c.u64()?;
                same(
                    seed < 10 && (i == 0 || seeds[i - 1] < seed),
                    "work.seeds.order.range",
                )?;
                seeds[i] = seed;
            }
            Work::Preparation {
                operation,
                profile,
                seeds,
                count,
            }
        };
        c.finish()?;
        Ok(Self { mode, body })
    }
    pub(in crate::study_reference::document::evidence) fn encode_fixed(&self) -> Result<FixedWork> {
        let mut w = FixedWork {
            bytes: [0; 4096],
            len: 0,
        };
        w.b(ENCODING_VERSION.as_bytes())?;
        // Fixed literal domains avoid a heap String on the pre-expansion route.
        w.b(if self.mode.synthetic() {
            b"hekit-test-resource-requested-work-v1"
        } else {
            b"hekit-study-resource-requested-work-v1"
        })?;
        w.put(&[u8::from(self.mode.synthetic())])?;
        match &self.body {
            Work::Train(t) => {
                w.put(&[0, 0])?;
                w.b(t.registration.0.as_bytes())?;
                w.b(t.profile.0.as_bytes())?;
                w.n(0)?;
                w.put(&[0, 0, t.mode, t.mode, 0])?;
                w.b(TRAIN_RECIPE.as_bytes())?;
                w.b(t.request)?;
                w.put(&[0])?;
            }
            Work::Preparation {
                operation,
                profile,
                seeds,
                count,
            } => {
                same(*count <= 10, "work.seeds.count")?;
                w.put(&[1, *operation])?;
                w.b(profile.0.as_bytes())?;
                w.n(*count as u64)?;
                for n in &seeds[..*count] {
                    w.n(*n)?;
                }
            }
        }
        RequestedWork::decode(w.as_slice(), self.mode)?;
        Ok(w)
    }
    pub(in crate::study_reference::document::evidence) fn train_request(&self) -> Result<&'a [u8]> {
        match &self.body {
            Work::Train(t) => Ok(t.request),
            _ => Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "work.train.required",
            )),
        }
    }
    pub(in crate::study_reference::document::evidence) fn bind_train(
        &self,
        case: &[u8],
        runtime: &str,
        request: &[u8],
        partitions: Option<u64>,
    ) -> Result<()> {
        use crate::study_profile::{CaseBackend, CaseOrigin, CasePacking, CaseRole, CaseScope};
        let Work::Train(t) = &self.body else {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "work.train.required",
            ));
        };
        let c = admission::parse_case_binary(case)?;
        same(
            c.origin == CaseOrigin::Family
                && c.registration_id.0 == t.registration.0
                && c.profile_id.0 == t.profile.0
                && c.seed == 0
                && c.scope == CaseScope::Full
                && c.backend == CaseBackend::Cpu
                && c.role == CaseRole::Main
                && c.mode as u8 == t.mode
                && c.placement as u8 == t.mode
                && c.packing == CasePacking::BatchedDiagonal
                && c.security_policy as u8 == t.mode
                && c.logical_batch == 20
                && c.eval_config.is_none(),
            "work.resolved.case",
        )?;
        same(
            runtime == TRAIN_RECIPE && request == t.request && partitions.is_none(),
            "work.resolved.request",
        )
    }
    pub(in crate::study_reference::document::evidence) fn bind_preparation_root(
        &self,
        operation: u8,
        profile: &str,
        seeds: &[u64],
    ) -> Result<()> {
        let Work::Preparation {
            operation: op,
            profile: p,
            seeds: s,
            count,
        } = &self.body
        else {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "work.preparation.required",
            ));
        };
        same(
            operation == *op && profile == p.0 && seeds == &s[..*count],
            "work.root.binding",
        )?;
        if operation != 3 {
            same(seeds == [0], "work.root.seed0")?;
        } else {
            same(
                seeds.first() == Some(&0) && seeds.len() <= 2,
                "work.root.selected_seed",
            )?;
        }
        Ok(())
    }
    /// Closed graph relation only. SAME originating owner is separately required.
    pub(in crate::study_reference::document::evidence) fn check_child(
        &self,
        parent: u8,
        child: u8,
        profile: &str,
        selected: u64,
    ) -> Result<()> {
        let (p, root, full, seeds) = match &self.body {
            Work::Train(t) => (t.profile.0, 6, compiled_request(t.request)? == 0, None),
            Work::Preparation {
                operation,
                profile,
                seeds,
                count,
            } => (profile.0, *operation, false, Some(&seeds[..*count])),
        };
        same(
            profile == p && selected < 10 && (root != 6 || selected == 0),
            "work.child.profile.seed",
        )?;
        let expected = [0, selected];
        let expected = if selected == 0 {
            &expected[..1]
        } else {
            &expected[..]
        };
        if let Some(s) = seeds {
            same(
                s == if root == 3 { expected } else { &[0] },
                "work.child.coverage",
            )?;
        }
        same(
            ((3..=5).contains(&root) && parent == root && child == 2)
                || (root == 6
                    && full
                    && ((parent == 6 && child == 3) || (parent == 3 && child == 2))),
            "work.child.edge",
        )
    }
}
