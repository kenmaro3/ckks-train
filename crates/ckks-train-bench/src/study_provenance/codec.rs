//! Bounded canonical NEW domains only; no scientific identity decoder.
use super::{ProvenanceError, Result, ensure};
use crate::data::sha256_hex;
pub(super) const LIMIT: usize = 8 * 1024 * 1024;
pub(super) const CAPSULE_LIMIT: usize = 2 * 1024 * 1024;
const TEXT: usize = 4096;
const COUNT: usize = 65_536;
const ENCODING: &str = "hekit-study-reference-encoding-v1";
pub(super) const RECIPE: &str = "study-reference-linux-clean-v1";
pub(super) const ALGORITHM: &str = "study-reference-session-v1";
pub(super) const TARGET: &str = "x86_64-unknown-linux-gnu";

pub(super) fn hash_valid(h: &str) -> bool {
    h.len() == 64
        && h.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(super) fn relative(p: &[u8]) -> bool {
    std::str::from_utf8(p).is_ok()
        && !p.is_empty()
        && !p.contains(&0)
        && !p.contains(&b'\\')
        && p.split(|b| *b == b'/')
            .all(|x| !x.is_empty() && x != b"." && x != b"..")
}
fn ordered<T: Ord>(items: impl Iterator<Item = T>) -> Result<()> {
    let mut last = None;
    for item in items {
        if let Some(old) = last {
            ensure(old < item, "noncanonical order/duplicate")?;
        }
        last = Some(item);
    }
    Ok(())
}
fn checked_end(offset: usize, length: usize, cap: usize) -> Result<usize> {
    let end = offset
        .checked_add(length)
        .ok_or(ProvenanceError("size arithmetic overflow"))?;
    ensure(end <= cap, "byte cap/truncated domain")?;
    Ok(end)
}
struct R<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> R<'a> {
    fn new(b: &'a [u8], domain: &str, cap: usize) -> Result<Self> {
        ensure(b.len() <= cap, "domain byte cap")?;
        let mut r = Self { b, p: 0 };
        ensure(
            r.t(TEXT)? == ENCODING && r.t(TEXT)? == domain,
            "domain header",
        )?;
        Ok(r)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = checked_end(self.p, n, self.b.len())?;
        let x = &self.b[self.p..end];
        self.p = end;
        Ok(x)
    }
    fn n(&mut self) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| ProvenanceError("u64"))?,
        ))
        .map_err(|_| ProvenanceError("usize overflow"))
    }
    fn q(&mut self, max: u8) -> Result<u8> {
        let q = self.take(1)?[0];
        ensure(q <= max, "unknown tag")?;
        Ok(q)
    }
    fn b(&mut self, cap: usize) -> Result<Vec<u8>> {
        let n = self.n()?;
        ensure(n <= cap, "field byte cap")?;
        Ok(self.take(n)?.to_vec())
    }
    fn t(&mut self, cap: usize) -> Result<String> {
        String::from_utf8(self.b(cap)?).map_err(|_| ProvenanceError("invalid UTF8"))
    }
    fn h(&mut self) -> Result<String> {
        let h = self.t(TEXT)?;
        ensure(hash_valid(&h), "invalid SHA256")?;
        Ok(h)
    }
    fn count(&mut self, cap: usize) -> Result<usize> {
        let n = self.n()?;
        ensure(n <= cap && n <= self.b.len() - self.p, "count cap")?;
        Ok(n)
    }
    fn argv(&mut self) -> Result<Vec<Vec<u8>>> {
        let n = self.count(256)?;
        (0..n).map(|_| self.b(TEXT)).collect()
    }
    fn finish(&self) -> Result<()> {
        ensure(self.p == self.b.len(), "trailing domain bytes")
    }
}
struct W {
    b: Vec<u8>,
    cap: usize,
}
impl W {
    fn new(domain: &str, cap: usize) -> Result<Self> {
        let mut w = Self { b: Vec::new(), cap };
        w.t(ENCODING, TEXT)?;
        w.t(domain, TEXT)?;
        Ok(w)
    }
    fn add(&mut self, b: &[u8]) -> Result<()> {
        checked_end(self.b.len(), b.len(), self.cap)?;
        self.b.extend_from_slice(b);
        Ok(())
    }
    fn n(&mut self, n: usize) -> Result<()> {
        self.add(
            &u64::try_from(n)
                .map_err(|_| ProvenanceError("u64 overflow"))?
                .to_le_bytes(),
        )
    }
    fn q(&mut self, q: u8) -> Result<()> {
        self.add(&[q])
    }
    fn b(&mut self, b: &[u8], cap: usize) -> Result<()> {
        ensure(b.len() <= cap, "field byte cap")?;
        self.n(b.len())?;
        self.add(b)
    }
    fn t(&mut self, t: &str, cap: usize) -> Result<()> {
        self.b(t.as_bytes(), cap)
    }
    fn h(&mut self, h: &str) -> Result<()> {
        ensure(hash_valid(h), "SHA256")?;
        self.t(h, TEXT)
    }
    fn argv(&mut self, v: &[Vec<u8>]) -> Result<()> {
        ensure(v.len() <= 256, "argv cap")?;
        self.n(v.len())?;
        for b in v {
            self.b(b, TEXT)?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum FileValue {
    Absent,
    Regular(String),
    Link(Vec<u8>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Manifest {
    pub role: u8,
    pub records: Vec<(Vec<u8>, FileValue)>,
}
impl Manifest {
    const DOMAIN: &'static str = "hekit-study-build-files-v1";
    pub fn decode(b: &[u8]) -> Result<Self> {
        let mut r = R::new(b, Self::DOMAIN, LIMIT)?;
        let role = r.q(3)?;
        let n = r.count(if role == 0 { 512 } else { COUNT })?;
        let mut records = Vec::new();
        for _ in 0..n {
            let p = r.b(TEXT)?;
            let v = match r.q(2)? {
                0 => FileValue::Absent,
                1 => FileValue::Regular(r.h()?),
                _ => FileValue::Link(r.b(TEXT)?),
            };
            records.push((p, v));
        }
        r.finish()?;
        let x = Self { role, records };
        ensure(x.encode()? == b, "manifest canonical equality")?;
        Ok(x)
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        ensure(
            self.role <= 3 && self.records.len() <= if self.role == 0 { 512 } else { COUNT },
            "manifest bounds",
        )?;
        ordered(self.records.iter().map(|r| &r.0))?;
        let mut w = W::new(Self::DOMAIN, LIMIT)?;
        w.q(self.role)?;
        w.n(self.records.len())?;
        for (p, v) in &self.records {
            w.b(p, TEXT)?;
            match v {
                FileValue::Absent => w.q(0)?,
                FileValue::Regular(h) => {
                    w.q(1)?;
                    w.h(h)?
                }
                FileValue::Link(t) => {
                    w.q(2)?;
                    w.b(t, TEXT)?
                }
            }
        }
        Ok(w.b)
    }
    pub fn policy(&self) -> Result<()> {
        for (p, v) in &self.records {
            ensure(!p.is_empty() && !p.contains(&0), "manifest path")?;
            if self.role == 0 {
                ensure(
                    relative(p) && matches!(v, FileValue::Regular(_)),
                    "source manifest policy",
                )?;
            } else {
                ensure(
                    p.starts_with(b"/")
                        && !p.split(|b| *b == b'/').any(|c| c == b"." || c == b".."),
                    "absolute lexical path",
                )?;
            }
            if self.role == 1 {
                ensure(matches!(v, FileValue::Absent), "config must be absent")?;
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Environment(pub Vec<(String, Option<Vec<u8>>)>);
impl Environment {
    const DOMAIN: &'static str = "hekit-study-build-environment-v1";
    pub fn decode(b: &[u8]) -> Result<Self> {
        let mut r = R::new(b, Self::DOMAIN, LIMIT)?;
        let n = r.count(COUNT)?;
        let mut v = Vec::new();
        for _ in 0..n {
            let key = r.t(TEXT)?;
            let value = if r.q(1)? == 0 { None } else { Some(r.b(TEXT)?) };
            v.push((key, value));
        }
        r.finish()?;
        let x = Self(v);
        ensure(x.encode()? == b, "environment canonical equality")?;
        Ok(x)
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        ordered(self.0.iter().map(|x| &x.0))?;
        ensure(self.0.len() <= COUNT, "environment cap")?;
        let mut w = W::new(Self::DOMAIN, LIMIT)?;
        w.n(self.0.len())?;
        for (k, v) in &self.0 {
            ensure(
                !k.is_empty() && k.is_ascii() && !k.contains('\0'),
                "ASCII environment key",
            )?;
            w.t(k, TEXT)?;
            w.q(u8::from(v.is_some()))?;
            if let Some(v) = v {
                w.b(v, TEXT)?;
            }
        }
        Ok(w.b)
    }
    pub fn get(&self, key: &str) -> Result<&[u8]> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.as_deref())
            .ok_or(ProvenanceError("required environment key"))
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Package {
    pub id: String,
    pub manifest: Vec<u8>,
    pub tag: u8,
    pub source: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Capsule {
    pub recipe: String,
    pub algorithm: String,
    pub manifests: Vec<String>,
    pub environment: String,
    pub lock: String,
    pub commit: String,
    pub compiler: String,
    pub cargo: String,
    pub target: String,
    pub host: String,
    pub profile: String,
    pub opt: String,
    pub debug: u8,
    pub features: Vec<String>,
    pub version: String,
    pub metadata: String,
    pub packages: Vec<Package>,
    pub metadata_argv: Vec<Vec<u8>>,
    pub build_argv: Vec<Vec<u8>>,
    pub cwd: Vec<u8>,
}
impl Capsule {
    const DOMAIN: &'static str = "hekit-study-compile-capsule-v1";
    pub fn decode(b: &[u8]) -> Result<Self> {
        let mut r = R::new(b, Self::DOMAIN, CAPSULE_LIMIT)?;
        let recipe = r.t(TEXT)?;
        let algorithm = r.t(TEXT)?;
        let manifests = (0..4).map(|_| r.h()).collect::<Result<_>>()?;
        let environment = r.h()?;
        let lock = r.h()?;
        let commit = r.t(TEXT)?;
        let compiler = r.t(16384)?;
        let cargo = r.t(TEXT)?;
        let target = r.t(TEXT)?;
        let host = r.t(TEXT)?;
        let profile = r.t(TEXT)?;
        let opt = r.t(TEXT)?;
        let debug = r.q(1)?;
        let n = r.count(64)?;
        let features = (0..n).map(|_| r.t(TEXT)).collect::<Result<_>>()?;
        let version = r.t(TEXT)?;
        let metadata = r.h()?;
        let n = r.count(COUNT)?;
        let mut packages = Vec::new();
        for _ in 0..n {
            packages.push(Package {
                id: r.t(TEXT)?,
                manifest: r.b(TEXT)?,
                tag: r.q(1)?,
                source: r.t(TEXT)?,
            });
        }
        let metadata_argv = r.argv()?;
        let build_argv = r.argv()?;
        let cwd = r.b(TEXT)?;
        r.finish()?;
        let x = Self {
            recipe,
            algorithm,
            manifests,
            environment,
            lock,
            commit,
            compiler,
            cargo,
            target,
            host,
            profile,
            opt,
            debug,
            features,
            version,
            metadata,
            packages,
            metadata_argv,
            build_argv,
            cwd,
        };
        ensure(x.encode()? == b, "capsule canonical equality")?;
        Ok(x)
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut w = W::new(Self::DOMAIN, CAPSULE_LIMIT)?;
        w.t(&self.recipe, TEXT)?;
        w.t(&self.algorithm, TEXT)?;
        ensure(self.manifests.len() == 4, "four manifests")?;
        for h in &self.manifests {
            w.h(h)?;
        }
        w.h(&self.environment)?;
        w.h(&self.lock)?;
        w.t(&self.commit, TEXT)?;
        w.t(&self.compiler, 16384)?;
        w.t(&self.cargo, TEXT)?;
        for t in [&self.target, &self.host, &self.profile, &self.opt] {
            w.t(t, TEXT)?;
        }
        ensure(self.debug <= 1, "debug tag")?;
        w.q(self.debug)?;
        ensure(self.features.len() <= 64, "feature cap")?;
        ordered(self.features.iter())?;
        w.n(self.features.len())?;
        for f in &self.features {
            w.t(f, TEXT)?;
        }
        w.t(&self.version, TEXT)?;
        w.h(&self.metadata)?;
        ensure(self.packages.len() <= COUNT, "package cap")?;
        ordered(self.packages.iter().map(|p| &p.id))?;
        w.n(self.packages.len())?;
        for p in &self.packages {
            w.t(&p.id, TEXT)?;
            w.b(&p.manifest, TEXT)?;
            ensure(p.tag <= 1, "package tag")?;
            w.q(p.tag)?;
            w.t(&p.source, TEXT)?;
        }
        w.argv(&self.metadata_argv)?;
        w.argv(&self.build_argv)?;
        w.b(&self.cwd, TEXT)?;
        Ok(w.b)
    }
    pub fn policy(&self) -> Result<()> {
        ensure(
            self.recipe == RECIPE && self.algorithm == ALGORITHM,
            "unsupported recipe",
        )?;
        ensure(
            self.target == TARGET
                && self.host == TARGET
                && self.profile == "release"
                && self.opt == "3"
                && self.debug == 0
                && self.features.is_empty(),
            "unsupported compile facts",
        )?;
        ensure(
            self.commit.len() == 40
                && self
                    .commit
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "Git revision",
        )
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Artifact {
    pub id: String,
    pub manifest: Vec<u8>,
    pub name: String,
    pub src: Vec<u8>,
    pub kinds: Vec<String>,
    pub executable: Option<Vec<u8>>,
}
type ArtifactKey<'a> = (
    &'a str,
    &'a str,
    &'a [u8],
    &'a [u8],
    &'a [String],
    &'a Option<Vec<u8>>,
);
impl Artifact {
    fn key(&self) -> ArtifactKey<'_> {
        (
            &self.id,
            &self.name,
            &self.src,
            &self.manifest,
            &self.kinds,
            &self.executable,
        )
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Receipt {
    pub capsule: Vec<u8>,
    pub files: Vec<Vec<u8>>,
    pub environment: Vec<u8>,
    pub executable: Vec<u8>,
    pub executable_hash: String,
    pub post: Vec<String>,
    pub metadata: String,
    pub emitted: Vec<Artifact>,
    pub argv: Vec<Vec<u8>>,
}
impl Receipt {
    const DOMAIN: &'static str = "hekit-study-build-receipt-v1";
    pub fn decode(b: &[u8]) -> Result<Self> {
        let mut r = R::new(b, Self::DOMAIN, LIMIT)?;
        let capsule = r.b(CAPSULE_LIMIT)?;
        let files = (0..4).map(|_| r.b(LIMIT)).collect::<Result<_>>()?;
        let environment = r.b(LIMIT)?;
        let executable = r.b(TEXT)?;
        let executable_hash = r.h()?;
        let post = (0..5).map(|_| r.h()).collect::<Result<_>>()?;
        let metadata = r.h()?;
        let n = r.count(COUNT)?;
        let mut emitted = Vec::new();
        for _ in 0..n {
            let id = r.t(TEXT)?;
            let manifest = r.b(TEXT)?;
            let name = r.t(TEXT)?;
            let src = r.b(TEXT)?;
            let n = r.count(COUNT)?;
            let kinds = (0..n).map(|_| r.t(TEXT)).collect::<Result<_>>()?;
            let executable = if r.q(1)? == 0 { None } else { Some(r.b(TEXT)?) };
            emitted.push(Artifact {
                id,
                manifest,
                name,
                src,
                kinds,
                executable,
            });
        }
        let argv = r.argv()?;
        r.finish()?;
        let x = Self {
            capsule,
            files,
            environment,
            executable,
            executable_hash,
            post,
            metadata,
            emitted,
            argv,
        };
        ensure(x.encode()? == b, "receipt canonical equality")?;
        Ok(x)
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.preflight()?;
        let c = Capsule::decode(&self.capsule)?;
        ensure(
            self.files.len() == 4 && self.post.len() == 5,
            "receipt lists",
        )?;
        for (i, b) in self.files.iter().enumerate() {
            let m = Manifest::decode(b)?;
            ensure(
                usize::from(m.role) == i && sha256_hex(b) == c.manifests[i],
                "embedded manifest linkage",
            )?;
        }
        Environment::decode(&self.environment)?;
        ensure(
            sha256_hex(&self.environment) == c.environment,
            "embedded environment linkage",
        )?;
        let mut w = W::new(Self::DOMAIN, LIMIT)?;
        w.b(&self.capsule, CAPSULE_LIMIT)?;
        for b in &self.files {
            w.b(b, LIMIT)?;
        }
        w.b(&self.environment, LIMIT)?;
        w.b(&self.executable, TEXT)?;
        w.h(&self.executable_hash)?;
        for h in &self.post {
            w.h(h)?;
        }
        w.h(&self.metadata)?;
        ensure(self.emitted.len() <= COUNT, "artifact cap")?;
        ordered(self.emitted.iter().map(Artifact::key))?;
        w.n(self.emitted.len())?;
        for a in &self.emitted {
            w.t(&a.id, TEXT)?;
            w.b(&a.manifest, TEXT)?;
            w.t(&a.name, TEXT)?;
            w.b(&a.src, TEXT)?;
            ensure(a.kinds.len() <= COUNT, "kind cap")?;
            ordered(a.kinds.iter())?;
            w.n(a.kinds.len())?;
            for k in &a.kinds {
                w.t(k, TEXT)?;
            }
            w.q(u8::from(a.executable.is_some()))?;
            if let Some(e) = &a.executable {
                w.b(e, TEXT)?;
            }
        }
        w.argv(&self.argv)?;
        Ok(w.b)
    }
    fn preflight(&self) -> Result<usize> {
        ensure(
            self.files.len() == 4
                && self.post.len() == 5
                && self.emitted.len() <= COUNT
                && self.argv.len() <= 256,
            "receipt counts",
        )?;
        let mut size = 0;
        let mut blob = |length: usize, cap: usize| -> Result<()> {
            ensure(length <= cap, "field byte cap")?;
            size = checked_end(size, 8, LIMIT)?;
            size = checked_end(size, length, LIMIT)?;
            Ok(())
        };
        blob(ENCODING.len(), TEXT)?;
        blob(Self::DOMAIN.len(), TEXT)?;
        blob(self.capsule.len(), CAPSULE_LIMIT)?;
        for b in &self.files {
            blob(b.len(), LIMIT)?;
        }
        blob(self.environment.len(), LIMIT)?;
        blob(self.executable.len(), TEXT)?;
        for h in std::iter::once(&self.executable_hash)
            .chain(self.post.iter())
            .chain(std::iter::once(&self.metadata))
        {
            ensure(hash_valid(h), "SHA256")?;
            blob(h.len(), TEXT)?;
        }
        size = checked_end(size, 8, LIMIT)?;
        for a in &self.emitted {
            ensure(a.kinds.len() <= COUNT, "kind cap")?;
            for n in [a.id.len(), a.manifest.len(), a.name.len(), a.src.len()] {
                ensure(n <= TEXT, "artifact field cap")?;
                size = checked_end(checked_end(size, 8, LIMIT)?, n, LIMIT)?;
            }
            size = checked_end(size, 8, LIMIT)?;
            for k in &a.kinds {
                ensure(k.len() <= TEXT, "kind field cap")?;
                size = checked_end(checked_end(size, 8, LIMIT)?, k.len(), LIMIT)?;
            }
            size = checked_end(size, 1, LIMIT)?;
            if let Some(e) = &a.executable {
                ensure(e.len() <= TEXT, "executable field cap")?;
                size = checked_end(checked_end(size, 8, LIMIT)?, e.len(), LIMIT)?;
            }
        }
        size = checked_end(size, 8, LIMIT)?;
        for arg in &self.argv {
            ensure(arg.len() <= TEXT, "argv field cap")?;
            size = checked_end(checked_end(size, 8, LIMIT)?, arg.len(), LIMIT)?;
        }
        Ok(size)
    }
    pub fn policy(&self) -> Result<Capsule> {
        let c = Capsule::decode(&self.capsule)?;
        c.policy()?;
        self.check_post(&c)?;
        Ok(c)
    }
    pub fn closed_policy(&self) -> Result<Capsule> {
        let c = Capsule::decode(&self.capsule)?;
        super::role::Recipe::capsule_policy(&c)?;
        self.check_post(&c)?;
        Ok(c)
    }
    pub fn check_manifest_links(&self, c: &Capsule) -> Result<()> {
        ensure(
            self.files.len() == 4 && c.manifests.len() == 4,
            "four manifests",
        )?;
        for (i, b) in self.files.iter().enumerate() {
            let manifest = Manifest::decode(b)?;
            ensure(
                usize::from(manifest.role) == i && sha256_hex(b) == c.manifests[i],
                "embedded manifest linkage",
            )?;
        }
        ensure(
            sha256_hex(&self.environment) == c.environment,
            "embedded environment linkage",
        )
    }
    pub fn check_post(&self, c: &Capsule) -> Result<()> {
        self.check_manifest_links(c)?;
        ensure(
            self.post.len() == 5
                && self.post[..4] == c.manifests
                && self.post[4] == c.environment
                && self.metadata == c.metadata,
            "pre/post mismatch",
        )?;
        for b in &self.files {
            Manifest::decode(b)?.policy()?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Approval {
    pub id: String,
    pub recipe: String,
    pub algorithm: String,
    pub designs: Vec<(String, String)>,
    pub commit: String,
    pub source: String,
    pub builds: Vec<(String, String, String)>,
}
impl Approval {
    const DOMAIN: &'static str = "hekit-study-local-approval-v1";
    pub fn decode(b: &[u8]) -> Result<Self> {
        let mut r = R::new(b, Self::DOMAIN, LIMIT)?;
        let id = r.t(TEXT)?;
        let recipe = r.t(TEXT)?;
        let algorithm = r.t(TEXT)?;
        let n = r.count(32)?;
        let designs = (0..n)
            .map(|_| Ok((r.t(TEXT)?, r.h()?)))
            .collect::<Result<_>>()?;
        let commit = r.t(TEXT)?;
        let source = r.h()?;
        let n = r.count(64)?;
        let builds = (0..n)
            .map(|_| Ok((r.h()?, r.h()?, r.h()?)))
            .collect::<Result<_>>()?;
        r.finish()?;
        let x = Self {
            id,
            recipe,
            algorithm,
            designs,
            commit,
            source,
            builds,
        };
        ensure(x.encode()? == b, "approval canonical equality")?;
        Ok(x)
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut w = W::new(Self::DOMAIN, LIMIT)?;
        w.t(&self.id, TEXT)?;
        w.t(&self.recipe, TEXT)?;
        w.t(&self.algorithm, TEXT)?;
        ensure(
            self.designs.len() <= 32 && self.builds.len() <= 64,
            "approval caps",
        )?;
        ordered(self.designs.iter().map(|x| &x.0))?;
        w.n(self.designs.len())?;
        for (p, h) in &self.designs {
            w.t(p, TEXT)?;
            w.h(h)?;
        }
        w.t(&self.commit, TEXT)?;
        w.h(&self.source)?;
        ordered(self.builds.iter().map(|x| &x.0))?;
        w.n(self.builds.len())?;
        for (r, c, e) in &self.builds {
            w.h(r)?;
            w.h(c)?;
            w.h(e)?;
        }
        Ok(w.b)
    }
}

/// V2 keeps role claims untrusted until the complete receipt join.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ApprovalBuild {
    pub role: u8,
    pub recipe: String,
    pub algorithm: String,
    pub receipt: String,
    pub capsule: String,
    pub executable: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ApprovalV2 {
    pub id: String,
    pub designs: Vec<(String, String)>,
    pub commit: String,
    pub source: String,
    pub builds: Vec<ApprovalBuild>,
}
impl ApprovalV2 {
    const DOMAIN: &'static str = "hekit-study-local-approval-v2";
    pub fn decode(b: &[u8]) -> Result<Self> {
        let mut r = R::new(b, Self::DOMAIN, LIMIT)?;
        let id = r.t(TEXT)?;
        let n = r.count(32)?;
        let designs = (0..n)
            .map(|_| Ok((r.t(TEXT)?, r.h()?)))
            .collect::<Result<_>>()?;
        let commit = r.t(TEXT)?;
        let source = r.h()?;
        let n = r.count(64)?;
        let builds = (0..n)
            .map(|_| {
                Ok(ApprovalBuild {
                    role: r.q(1)?,
                    recipe: r.t(TEXT)?,
                    algorithm: r.t(TEXT)?,
                    receipt: r.h()?,
                    capsule: r.h()?,
                    executable: r.h()?,
                })
            })
            .collect::<Result<_>>()?;
        r.finish()?;
        let a = Self {
            id,
            designs,
            commit,
            source,
            builds,
        };
        ensure(a.encode()? == b, "approval canonical equality")?;
        Ok(a)
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut w = W::new(Self::DOMAIN, LIMIT)?;
        w.t(&self.id, TEXT)?;
        ensure(
            self.designs.len() <= 32 && self.builds.len() <= 64,
            "approval caps",
        )?;
        ordered(self.designs.iter().map(|x| &x.0))?;
        w.n(self.designs.len())?;
        for (p, h) in &self.designs {
            w.t(p, TEXT)?;
            w.h(h)?;
        }
        w.t(&self.commit, TEXT)?;
        w.h(&self.source)?;
        ordered(self.builds.iter().map(|b| &b.receipt))?;
        w.n(self.builds.len())?;
        for b in &self.builds {
            ensure(b.role <= 1, "approval role")?;
            w.q(b.role)?;
            w.t(&b.recipe, TEXT)?;
            w.t(&b.algorithm, TEXT)?;
            w.h(&b.receipt)?;
            w.h(&b.capsule)?;
            w.h(&b.executable)?;
        }
        Ok(w.b)
    }
}
#[derive(Clone, Debug)]
pub(super) enum ApprovalVersion {
    V1(Approval),
    V2(ApprovalV2),
    V3(
        ApprovalV2,
        Box<super::resource_operational::approval::OriginProjection>,
    ),
}
#[derive(Clone, Debug)]
pub(super) struct LocalApproval {
    pub id: String,
    pub designs: Vec<(String, String)>,
    pub commit: String,
    pub source: String,
    pub version: ApprovalVersion,
}
impl LocalApproval {
    pub fn decode(b: &[u8]) -> Result<Self> {
        ensure(b.len() <= LIMIT, "domain byte cap")?;
        let mut r = R { b, p: 0 };
        ensure(r.t(TEXT)? == ENCODING, "domain header")?;
        match r.t(TEXT)?.as_str() {
            Approval::DOMAIN => {
                let a = Approval::decode(b)?;
                Ok(Self {
                    id: a.id.clone(),
                    designs: a.designs.clone(),
                    commit: a.commit.clone(),
                    source: a.source.clone(),
                    version: ApprovalVersion::V1(a),
                })
            }
            ApprovalV2::DOMAIN => {
                let a = ApprovalV2::decode(b)?;
                Ok(Self {
                    id: a.id.clone(),
                    designs: a.designs.clone(),
                    commit: a.commit.clone(),
                    source: a.source.clone(),
                    version: ApprovalVersion::V2(a),
                })
            }
            "hekit-study-local-approval-v3" => {
                let scanned = super::resource_operational::approval::scan_approval(
                    &mut std::io::Cursor::new(b),
                    &sha256_hex(b),
                )?;
                let mut body = b[..scanned.body_end].to_vec();
                let version = 8 + ENCODING.len() + 8 + ApprovalV2::DOMAIN.len() - 1;
                body[version] = b'2';
                let a = ApprovalV2::decode(&body)?;
                Ok(Self {
                    id: a.id.clone(),
                    designs: a.designs.clone(),
                    commit: a.commit.clone(),
                    source: a.source.clone(),
                    version: ApprovalVersion::V3(a, Box::new(scanned.origin)),
                })
            }
            _ => Err(ProvenanceError("approval domain")),
        }
    }
    pub fn policy(&self, designs: &[&str]) -> Result<()> {
        use super::role::{ExecutableRole, Recipe, valid_commit};
        ensure(
            !self.id.is_empty() && valid_commit(&self.commit),
            "approval policy",
        )?;
        ensure(
            self.designs
                .iter()
                .map(|d| d.0.as_str())
                .eq(designs.iter().copied()),
            "approved design set",
        )?;
        match &self.version {
            ApprovalVersion::V1(a) => ensure(
                a.recipe == RECIPE && a.algorithm == ALGORITHM,
                "approval v1 Reference only",
            ),
            ApprovalVersion::V2(a) | ApprovalVersion::V3(a, _) => {
                for b in &a.builds {
                    let r = Recipe::classify(&b.recipe, &b.algorithm)?;
                    ensure(
                        b.role == u8::from(r.role() == ExecutableRole::StudyTrain),
                        "approval role/recipe mismatch",
                    )?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(s: &str) -> Vec<u8> {
        s.split_whitespace()
            .collect::<String>()
            .as_bytes()
            .chunks_exact(2)
            .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap(), 16).unwrap())
            .collect()
    }
    #[test]
    fn literal_domains_canonical_and_all_truncations_reject() {
        let cap = fixture(include_str!("fixtures/capsule.hex"));
        assert_eq!(Capsule::decode(&cap).unwrap().encode().unwrap(), cap);
        let rec = fixture(include_str!("fixtures/receipt.hex"));
        assert_eq!(Receipt::decode(&rec).unwrap().encode().unwrap(), rec);
        let app = fixture(include_str!("fixtures/approval.hex"));
        assert_eq!(Approval::decode(&app).unwrap().encode().unwrap(), app);
        for n in 0..rec.len() {
            assert!(Receipt::decode(&rec[..n]).is_err(), "cut {n}");
        }
        let mut appended = rec.clone();
        appended.push(0);
        assert!(Receipt::decode(&appended).is_err());
        let mut changed = Receipt::decode(&rec).unwrap();
        changed.files[0] = fixture(include_str!("fixtures/tag_source_regular.hex"));
        assert!(changed.encode().is_err());
        let mut c = Capsule::decode(&cap).unwrap();
        c.packages.reverse();
        assert!(c.encode().is_err());
        assert!(Receipt::decode(&rec).unwrap().policy().is_err());
    }
    #[test]
    fn raw_environment_and_file_roles_are_preserved() {
        let e = fixture(include_str!("fixtures/environment.hex"));
        let env = Environment::decode(&e).unwrap();
        assert_eq!(
            env.0,
            vec![
                ("A".into(), None),
                ("AA".into(), Some(vec![])),
                ("Z".into(), Some(vec![255, 0, 127]))
            ]
        );
        assert_eq!(env.encode().unwrap(), e);
        let dep = fixture(include_str!("fixtures/files_dependency.hex"));
        let m = Manifest::decode(&dep).unwrap();
        assert_eq!(m.records.len(), 4);
        assert_eq!(m.encode().unwrap(), dep);
        let link = fixture(include_str!("fixtures/tag_source_symlink.hex"));
        assert!(Manifest::decode(&link).unwrap().policy().is_err());
        let absent = fixture(include_str!("fixtures/tag_source_absent.hex"));
        assert!(Manifest::decode(&absent).unwrap().policy().is_err());
        let mut huge = e.clone();
        huge[..8].fill(255);
        assert!(Environment::decode(&huge).is_err());
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    #[test]
    fn checked_arithmetic_is_not_a_field_cap_test() {
        assert_eq!(
            checked_end(usize::MAX - 1, 1, usize::MAX).unwrap(),
            usize::MAX
        );
        assert_eq!(
            checked_end(usize::MAX, 1, usize::MAX).unwrap_err().0,
            "size arithmetic overflow"
        );
        assert!(checked_end(0, 2, 1).is_err());
        // Host-width conversion is distinct from field caps; 64-bit native
        // conversion succeeds, narrower-host branch explicitly checks rejection.
        let raw = u64::MAX.to_le_bytes();
        let mut reader = R { b: &raw, p: 0 };
        if usize::BITS == 64 {
            assert_eq!(reader.n().unwrap(), usize::MAX);
        } else {
            assert_eq!(reader.n().unwrap_err().0, "usize overflow");
        }
    }
    #[test]
    fn exact_list_bounds_and_receipt_preflight() {
        for cap in [256, 64, COUNT, 512, 32] {
            let mut bytes = (cap as u64).to_le_bytes().to_vec();
            bytes.resize(8 + cap, 0);
            assert_eq!(R { b: &bytes, p: 0 }.count(cap).unwrap(), cap);
            bytes[..8].copy_from_slice(&((cap + 1) as u64).to_le_bytes());
            assert!(R { b: &bytes, p: 0 }.count(cap).is_err());
        }
        let bytes = super::super::role_tests::literal("reference_receipt");
        let mut r = Receipt::decode(&bytes).unwrap();
        assert_eq!(r.preflight().unwrap(), bytes.len());
        // Invalid large embedded bytes reject at total-size preflight before
        // their parser or content hashes can run.
        r.files[0] = vec![0; LIMIT];
        assert_eq!(r.encode().unwrap_err().0, "byte cap/truncated domain");
        let c = Capsule::decode(&super::super::role_tests::literal("reference_capsule")).unwrap();
        let mut feature_bound = c.clone();
        feature_bound.features = (0..64).map(|i| format!("f{i:02}")).collect();
        Capsule::decode(&feature_bound.encode().unwrap()).unwrap();
        assert!(super::super::role::Recipe::capsule_policy(&feature_bound).is_err());
        feature_bound.features.push("z".into());
        assert!(feature_bound.encode().is_err());
        let mut args = c;
        args.metadata_argv = vec![Vec::new(); 256];
        args.encode().unwrap();
        args.metadata_argv.push(Vec::new());
        assert!(args.encode().is_err());
    }
    #[test]
    fn canonical_capsule_and_receipt_exact_total_caps() {
        let literal = super::super::role_tests::literal;
        let mut c = Capsule::decode(&literal("reference_capsule")).unwrap();
        c.packages.clear();
        c.cwd.clear();
        let base = c.encode().unwrap().len();
        let mut remaining = CAPSULE_LIMIT - base;
        let mut i = 0;
        while remaining > TEXT {
            let id = format!("p{i:06}");
            let overhead = 8 + id.len() + 8 + 1 + 8;
            let n = TEXT.min(remaining - overhead);
            c.packages.push(Package {
                id,
                manifest: vec![b'x'; n],
                tag: 0,
                source: String::new(),
            });
            remaining -= overhead + n;
            i += 1;
        }
        c.cwd = vec![b'x'; remaining];
        let bytes = c.encode().unwrap();
        assert_eq!(bytes.len(), CAPSULE_LIMIT);
        assert_eq!(Capsule::decode(&bytes).unwrap(), c);
        c.cwd.push(b'x');
        assert!(c.encode().is_err());
        assert!(Capsule::decode(&[bytes.as_slice(), &[0]].concat()).is_err());

        let mut r = Receipt::decode(&literal("reference_receipt")).unwrap();
        r.emitted.clear();
        r.argv.clear();
        r.executable.clear();
        let base = r.preflight().unwrap();
        let mut remaining = LIMIT - base;
        let mut i = 0;
        while remaining > TEXT {
            let id = format!("p{i:06}");
            // id/path/name/src lengths, kinds count, executable option tag.
            let overhead = 8 + id.len() + 8 + 8 + 8 + 8 + 1;
            let n = TEXT.min(remaining - overhead);
            r.emitted.push(Artifact {
                id,
                manifest: vec![b'x'; n],
                name: String::new(),
                src: Vec::new(),
                kinds: Vec::new(),
                executable: None,
            });
            remaining -= overhead + n;
            i += 1;
        }
        r.executable = vec![b'x'; remaining];
        let bytes = r.encode().unwrap();
        assert_eq!(bytes.len(), LIMIT);
        assert_eq!(Receipt::decode(&bytes).unwrap(), r);
        r.executable.push(b'x');
        assert!(r.encode().is_err());
        assert!(Receipt::decode(&[bytes.as_slice(), &[0]].concat()).is_err());
        // These are canonical transport containers, deliberately not closed
        // semantic recipes or valid source/executable inventory evidence.
    }
    #[test]
    fn exact_enclosing_and_field_caps() {
        for cap in [LIMIT, CAPSULE_LIMIT] {
            let mut w = W { b: Vec::new(), cap };
            w.add(&vec![0; cap]).unwrap();
            assert!(w.add(&[0]).is_err());
        }
        for cap in [TEXT, 16384] {
            let mut w = W::new("synthetic-bound-only", LIMIT).unwrap();
            w.b(&vec![0; cap], cap).unwrap();
            assert!(w.b(&vec![0; cap + 1], cap).is_err());
        }
    }
}
