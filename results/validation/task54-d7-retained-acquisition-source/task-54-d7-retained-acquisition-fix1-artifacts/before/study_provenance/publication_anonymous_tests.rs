//! Deterministic five-operation contract only; no claim about native inode custody.
use super::*;

/// Omitting unlink is deliberate: a publisher with an anonymous owned inode has
/// no pathname cleanup capability. Reintroducing a cleanup operation breaks this contract.
#[derive(Default)]
struct FiveOps {
    calls: Vec<u8>,
    fail: Option<u8>,
    final_visible: bool,
}
impl FiveOps {
    fn step(&mut self, op: u8) -> Result<()> {
        self.calls.push(op);
        ensure(self.fail != Some(op), "scripted boundary failure")
    }
}
impl Ops for FiveOps {
    fn create(&mut self) -> Result<()> {
        self.step(b'C')
    }
    fn write(&mut self, _: &[u8]) -> Result<()> {
        self.step(b'W')
    }
    fn file_sync(&mut self) -> Result<()> {
        self.step(b'F')
    }
    fn link(&mut self) -> Result<()> {
        self.step(b'L')?;
        ensure(!self.final_visible, "no clobber")?;
        self.final_visible = true;
        Ok(())
    }
    fn dir_sync(&mut self) -> Result<()> {
        self.step(b'D')
    }
}

#[test]
fn anonymous_five_operations_preserve_visibility_boundary() {
    for (fault, expected, visible) in [
        (Some(b'C'), &b"C"[..], false),
        (Some(b'W'), &b"CW"[..], false),
        (Some(b'F'), &b"CWF"[..], false),
        (Some(b'L'), &b"CWFL"[..], false),
        (Some(b'D'), &b"CWFLD"[..], true),
        (None, &b"CWFLD"[..], true),
    ] {
        let mut ops = FiveOps {
            fail: fault,
            ..Default::default()
        };
        let result = publish_ops(&mut ops, b"owned-ok");
        assert_eq!(ops.calls, expected);
        assert_eq!(ops.final_visible, visible);
        match fault {
            None => assert!(result.is_ok()),
            Some(b'D') => assert!(matches!(result, Err(PublishError::VisibleUnconfirmed))),
            _ => assert!(matches!(result, Err(PublishError::BeforeLink))),
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
fn publication_native_anonymous_success() {
    let d = SyntheticDirectory::native("anonymous-success");
    let mut native = Native::new(&d.file, "target.lock.json".into());
    native.create().unwrap();
    let owned = native.file.as_ref().unwrap();
    assert_eq!(owned.metadata().unwrap().nlink(), 0);
    assert_eq!(owned.metadata().unwrap().mode() & 0o7777, 0o600);
    let original = inode(owned).unwrap();
    assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
    native.write(b"owned-ok").unwrap();
    native.file_sync().unwrap();
    native.link().unwrap();
    native.dir_sync().unwrap();
    native.confirm_final().unwrap();
    assert_eq!(
        inode(&File::open(d.path.join("target.lock.json")).unwrap()).unwrap(),
        original
    );
    drop(native);
    assert_eq!(
        std::fs::read(d.path.join("target.lock.json")).unwrap(),
        b"owned-ok"
    );
}

#[test]
fn anonymous_recovery_has_only_two_sync_operations() {
    let mut ops = FiveOps {
        final_visible: true,
        ..Default::default()
    };
    recover_ops(&mut ops).unwrap();
    assert_eq!(ops.calls, b"FD");
    assert!(ops.final_visible);
}

#[cfg(not(target_os = "linux"))]
#[test]
fn anonymous_non_linux_refuses_without_creating_entries() {
    let d = SyntheticDirectory::new("unsupported-anonymous");
    let mut native = Native::new(&d.file, "target.lock.json".into());
    assert!(matches!(
        publish_ops(&mut native, b"owned-ok"),
        Err(PublishError::BeforeLink)
    ));
    assert!(native.file.is_none());
    assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
}

#[cfg(target_os = "linux")]
mod linux {
    use super::super::super::{anonymous, test_hooks};
    use super::*;
    use std::{
        os::unix::fs::{PermissionsExt, symlink},
        sync::{Arc, Mutex, mpsc},
        time::Duration,
    };
    const FINAL: &str = "target.lock.json";
    const TEMP: &str = ".tmp-target-lock-0";
    fn private_bytes(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    fn record() -> (test_hooks::Scope, Arc<Mutex<Vec<&'static str>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let copy = calls.clone();
        (
            test_hooks::Scope::install(move |s| {
                copy.lock().unwrap().push(s);
                Ok(())
            }),
            calls,
        )
    }
    fn fail(at: &'static str) -> test_hooks::Scope {
        test_hooks::Scope::install(move |step| ensure(step != at, "injected safe-call boundary"))
    }
    /// Actor waits for the exact caller cut. Publisher never handles T or M.
    fn actor(
        path: PathBuf,
    ) -> (
        mpsc::Sender<()>,
        mpsc::Receiver<()>,
        std::thread::JoinHandle<()>,
    ) {
        let (go_tx, go_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            go_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            std::fs::rename(path.join(TEMP), path.parent().unwrap().join("moved")).unwrap();
            private_bytes(&path.join(TEMP), b"foreign!");
            done_tx.send(()).unwrap();
        });
        (go_tx, done_rx, thread)
    }
    fn coordinate(tx: &mpsc::Sender<()>, rx: &mpsc::Receiver<()>) {
        tx.send(()).unwrap();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }
    fn sentinel_assertions(d: &SyntheticDirectory, original: (u64, u64)) {
        assert_eq!(std::fs::read(d.path.join(TEMP)).unwrap(), b"foreign!");
        assert_eq!(
            std::fs::read(d.path.parent().unwrap().join("moved")).unwrap(),
            b"sentinel"
        );
        assert_eq!(
            inode(&File::open(d.path.parent().unwrap().join("moved")).unwrap()).unwrap(),
            original
        );
    }

    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn publication_native_no_clobber() {
        for kind in ["regular", "symlink", "competing"] {
            let d = SyntheticDirectory::native(&format!("no-clobber-{kind}"));
            match kind {
                "regular" | "symlink" => {
                    private_bytes(&d.path.join("existing"), b"foreign!");
                    if kind == "regular" {
                        std::fs::hard_link(d.path.join("existing"), d.path.join(FINAL)).unwrap();
                    } else {
                        symlink("existing", d.path.join(FINAL)).unwrap();
                    }
                    let before = std::fs::symlink_metadata(d.path.join(FINAL)).unwrap().ino();
                    let mut native = Native::new(&d.file, FINAL.into());
                    assert!(matches!(
                        publish_ops(&mut native, b"owned-ok"),
                        Err(PublishError::BeforeLink)
                    ));
                    drop(native);
                    assert_eq!(
                        std::fs::symlink_metadata(d.path.join(FINAL)).unwrap().ino(),
                        before
                    );
                    assert_eq!(std::fs::read(d.path.join("existing")).unwrap(), b"foreign!");
                }
                _ => {
                    let (ready_tx, ready_rx) = mpsc::channel();
                    let mut starts = Vec::new();
                    let mut threads = Vec::new();
                    for _ in 0..2 {
                        let file = File::open(&d.path).unwrap();
                        let ready_tx = ready_tx.clone();
                        let (start_tx, start_rx) = mpsc::channel();
                        starts.push(start_tx);
                        threads.push(std::thread::spawn(move || {
                            let mut n = Native::new(&file, FINAL.into());
                            n.create().unwrap();
                            n.write(b"owned-ok").unwrap();
                            n.file_sync().unwrap();
                            let original = inode(n.file.as_ref().unwrap()).unwrap();
                            ready_tx.send(()).unwrap();
                            start_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                            let linked = n.link().is_ok();
                            if linked {
                                n.dir_sync().unwrap();
                                n.confirm_final().unwrap();
                            }
                            (linked, original)
                        }));
                    }
                    for _ in 0..2 {
                        ready_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                    }
                    for start in starts {
                        start.send(()).unwrap();
                    }
                    let outcomes = threads
                        .into_iter()
                        .map(|t| t.join().unwrap())
                        .collect::<Vec<_>>();
                    assert_eq!(outcomes.iter().filter(|(won, _)| *won).count(), 1);
                    let winner = outcomes.iter().find(|(won, _)| *won).unwrap().1;
                    assert_eq!(
                        inode(&File::open(d.path.join(FINAL)).unwrap()).unwrap(),
                        winner
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn publication_native_postcheck_windows() {
        for cut in [
            "link",
            "release",
            "write",
            "file-sync",
            "link-failure",
            "fchmod",
        ] {
            let d = SyntheticDirectory::native(&format!("postcheck-{cut}"));
            private_bytes(&d.path.join(TEMP), b"sentinel");
            let sentinel = inode(&File::open(d.path.join(TEMP)).unwrap()).unwrap();
            let (tx, rx, thread) = actor(d.path.clone());
            let mut n = Native::new(&d.file, FINAL.into());
            if cut == "fchmod" {
                let _scope = test_hooks::Scope::install(move |s| {
                    if s == "c6" {
                        coordinate(&tx, &rx);
                        return Err(ProvenanceError("injected fchmod error"));
                    }
                    Ok(())
                });
                assert!(matches!(
                    publish_ops(&mut n, b"owned-ok"),
                    Err(PublishError::BeforeLink)
                ));
            } else {
                n.create().unwrap();
                let original = inode(n.file.as_ref().unwrap()).unwrap();
                if cut == "write" {
                    let _scope = fail("W");
                    assert!(n.write(b"owned-ok").is_err());
                } else {
                    n.write(b"owned-ok").unwrap();
                    if cut == "file-sync" {
                        let _scope = fail("F");
                        assert!(n.file_sync().is_err());
                    } else {
                        n.file_sync().unwrap();
                        if cut == "link" {
                            coordinate(&tx, &rx);
                        }
                        if cut == "link-failure" {
                            let _scope = fail("L");
                            assert!(n.link().is_err());
                        } else {
                            n.link().unwrap();
                            n.dir_sync().unwrap();
                            n.confirm_final().unwrap();
                        }
                    }
                }
                if cut != "link" {
                    coordinate(&tx, &rx);
                }
                if matches!(cut, "link" | "release") {
                    assert_eq!(
                        inode(&File::open(d.path.join(FINAL)).unwrap()).unwrap(),
                        original
                    );
                }
            }
            drop(n);
            thread.join().unwrap();
            sentinel_assertions(&d, sentinel);
            if matches!(cut, "link" | "release") {
                assert_eq!(std::fs::read(d.path.join(FINAL)).unwrap(), b"owned-ok");
            } else {
                assert!(!d.path.join(FINAL).exists());
            }
        }
    }

    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn publication_native_legacy_prefix() {
        for count in [1, 64] {
            let d = SyntheticDirectory::native(&format!("prefix-{count}"));
            for i in 0..count {
                private_bytes(&d.path.join(format!(".tmp-target-lock-{i}")), b"foreign!");
            }
            let mut n = Native::new(&d.file, FINAL.into());
            publish_ops(&mut n, b"owned-ok").unwrap();
            n.confirm_final().unwrap();
            drop(n);
            for i in 0..count {
                assert_eq!(
                    std::fs::read(d.path.join(format!(".tmp-target-lock-{i}"))).unwrap(),
                    b"foreign!"
                );
            }
            assert_eq!(std::fs::read_dir(&d.path).unwrap().count(), count + 1);
        }
    }

    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn publication_native_drop() {
        for cut in ["unused", "write-progress", "fchmod", "child-exit"] {
            let d = SyntheticDirectory::native(&format!("drop-{cut}"));
            if cut == "child-exit" {
                run_child(&d, "before-link");
            } else {
                let mut n = Native::new(&d.file, FINAL.into());
                if cut == "fchmod" {
                    let _scope = fail("c6");
                    assert!(n.create().is_err());
                } else {
                    n.create().unwrap();
                    if cut == "write-progress" {
                        // Real progress on O, followed by explicit injected W boundary failure.
                        n.file.as_mut().unwrap().write_all(b"own").unwrap();
                        let _scope = fail("W");
                        assert!(n.write(b"ed-ok").is_err());
                    }
                }
                drop(n);
            }
            assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
        }
    }
    fn run_child(d: &SyntheticDirectory, cut: &str) {
        use std::{process::Command, time::Instant};
        let expected_code = match cut {
            "before-link" => 70,
            "after-link" => 71,
            _ => panic!("unsupported native child cut"),
        };
        let mut child=Command::new(std::env::current_exe().unwrap())
            .args(["--exact","--ignored","study_provenance::publication::tests::anonymous_contract::linux::publication_native_child","--test-threads=1"])
            .env("HEKIT_PUBLICATION_CHILD_PATH",&d.path).env("HEKIT_PUBLICATION_CHILD_CUT",cut)
            .spawn().unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert_eq!(
                    status.code(),
                    Some(expected_code),
                    "native child did not reach its requested cut"
                );
                break;
            }
            if Instant::now() >= until {
                child.kill().unwrap();
                let _ = child.wait();
                panic!("child timeout");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    #[test]
    #[ignore = "internal native child; invoke only through separately released exact parent harness"]
    fn publication_native_child() {
        let path = std::env::var_os("HEKIT_PUBLICATION_CHILD_PATH")
            .expect("native child requires its parent fixture");
        let cut = std::env::var("HEKIT_PUBLICATION_CHILD_CUT")
            .expect("native child requires an explicit cut");
        let exit_code = match cut.as_str() {
            "before-link" => 70,
            "after-link" => 71,
            _ => panic!("unsupported native child cut"),
        };
        let file = File::open(path).unwrap();
        let mut n = Native::new(&file, FINAL.into());
        n.create().unwrap();
        n.write(b"owned-ok").unwrap();
        n.file_sync().unwrap();
        if cut == "after-link" {
            n.link().unwrap();
        }
        // Deliberately bypass Rust destructors: process FD release is a separate observation.
        // Nonzero intentional outcome also rejects a zero-tests-success launch.
        std::process::exit(exit_code);
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn publication_native_visible_uncertain() {
        for cut in ["dir-sync", "child-exit"] {
            let d = SyntheticDirectory::native(&format!("visible-{cut}"));
            if cut == "child-exit" {
                run_child(&d, "after-link");
            } else {
                let _scope = fail("D");
                let mut n = Native::new(&d.file, FINAL.into());
                assert!(matches!(
                    publish_ops(&mut n, b"owned-ok"),
                    Err(PublishError::VisibleUnconfirmed)
                ));
            }
            assert_eq!(std::fs::read(d.path.join(FINAL)).unwrap(), b"owned-ok");
            assert_eq!(std::fs::read_dir(&d.path).unwrap().count(), 1);
            let mut recovered = Native::recovery(
                &d.file,
                FINAL.into(),
                File::open(d.path.join(FINAL)).unwrap(),
            );
            recover_ops(&mut recovered).unwrap();
            recovered.confirm_final().unwrap();
        }
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn publication_native_proc_refusal() {
        let d = SyntheticDirectory::native("proc-refusal");
        // Validates a private ordinary tree, never substitutes it for the fixed /proc root.
        assert!(anonymous::proc_directory(&d.file).is_err());
        let _scope = test_hooks::Scope::install(|s| {
            if s == "c3" {
                return Err(std::io::Error::from_raw_os_error(libc::ENOSYS).into());
            }
            Ok(())
        });
        let mut n = Native::new(&d.file, FINAL.into());
        assert!(matches!(
            publish_ops(&mut n, b"owned-ok"),
            Err(PublishError::BeforeLink)
        ));
        assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn publication_native_final_swap() {
        let d = SyntheticDirectory::native("final-swap");
        let caller = Caller::new(&d, 0);
        let path = caller.path();
        let moved = path.with_file_name("moved-final");
        let actor_path = path.clone();
        let actor_moved = moved.clone();
        let _scope = test_hooks::Scope::install(move |s| {
            if s == "final-check" {
                std::fs::rename(&actor_path, &actor_moved).unwrap();
                private_bytes(&actor_path, b"foreign!");
            }
            Ok(())
        });
        assert!(matches!(
            caller.publish(),
            Err(PublishError::VisibleUnconfirmed)
        ));
        assert_eq!(std::fs::read(path).unwrap(), b"foreign!");
        assert_eq!(std::fs::read(&moved).unwrap(), b"owned-ok");
        assert_eq!(
            inode(&File::open(moved).unwrap()).unwrap(),
            test_hooks::owned_identity().unwrap()
        );
    }

    /// Private test construction of the four real shared callers, never a context,
    /// a scientific record, a live gate, or a Reference/Train public capability.
    enum Caller {
        Store(Store),
        Target(TargetStore, ProfileGuard),
    }
    impl Caller {
        fn new(d: &SyntheticDirectory, selector: usize) -> Self {
            let root_mount = mount(&d.file).unwrap();
            let anchors = Anchors {
                files: vec![d.file.try_clone().unwrap()],
                names: vec![],
                private_from: 0,
                mount_from: 0,
                root_mount,
            };
            if selector == 3 {
                let expected = d.guard();
                let store = TargetStore {
                    anchors,
                    path: d.path.clone(),
                    guard: expected,
                };
                let guard = store.acquire().unwrap();
                Self::Target(store, guard)
            } else {
                let kind = [Kind::Reference, Kind::Diagnostic, Kind::Failure][selector];
                let path = d.path.join(kind.directory());
                std::fs::create_dir(&path).unwrap();
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
                Self::Store(Store {
                    anchors,
                    root: d.file.try_clone().unwrap(),
                    parent: File::open(&path).unwrap(),
                    path,
                    kind,
                })
            }
        }
        fn publish(&self) -> std::result::Result<DurableBytes, PublishError> {
            match self {
                Self::Store(s) => s.publish("shared", b"owned-ok", 8),
                Self::Target(s, g) => s.publish(g, b"owned-ok"),
            }
        }
        fn path(&self) -> PathBuf {
            match self {
                Self::Store(s) => s.path.join(s.basename("shared").unwrap()),
                Self::Target(s, _) => s.path(),
            }
        }
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn shared_anonymous_publication_faults() {
        for selector in 0..4 {
            for (fault, expected, visible) in [
                ("C", "C", false),
                ("W", "CW", false),
                ("F", "CWF", false),
                ("L", "CWFL", false),
                ("D", "CWFLD", true),
                ("none", "CWFLD", true),
            ] {
                let d = SyntheticDirectory::native(&format!("shared-fault-{selector}-{fault}"));
                let caller = Caller::new(&d, selector);
                let events = Arc::new(Mutex::new(Vec::new()));
                let copy = events.clone();
                let _scope = test_hooks::Scope::install(move |s| {
                    if matches!(s, "C" | "W" | "F" | "L" | "D") {
                        copy.lock().unwrap().push(s);
                    }
                    ensure(s != fault, "injected operation failure")
                });
                let result = caller.publish();
                assert_eq!(events.lock().unwrap().concat(), expected);
                assert_eq!(caller.path().exists(), visible);
                match fault {
                    "none" => {
                        let bytes = result.unwrap();
                        assert!(!bytes.temporary_cleanup_not_performed());
                    }
                    "D" => assert!(matches!(result, Err(PublishError::VisibleUnconfirmed))),
                    _ => assert!(matches!(result, Err(PublishError::BeforeLink))),
                }
                if visible {
                    assert_eq!(std::fs::read(caller.path()).unwrap(), b"owned-ok");
                }
            }
        }
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn shared_anonymous_late_confirmation_faults() {
        for selector in 0..4 {
            for fault in ["final-check", "anchor-check"] {
                let d = SyntheticDirectory::native(&format!("shared-late-{selector}-{fault}"));
                let caller = Caller::new(&d, selector);
                let _scope = fail(fault);
                assert!(matches!(
                    caller.publish(),
                    Err(PublishError::VisibleUnconfirmed)
                ));
                assert_eq!(std::fs::read(caller.path()).unwrap(), b"owned-ok");
            }
        }
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn shared_anonymous_no_clobber() {
        for selector in 0..4 {
            let d = SyntheticDirectory::native(&format!("shared-existing-{selector}"));
            let caller = Caller::new(&d, selector);
            caller.publish().unwrap();
            let original = inode(&File::open(caller.path()).unwrap()).unwrap();
            assert!(matches!(caller.publish(), Err(PublishError::BeforeLink)));
            assert_eq!(
                inode(&File::open(caller.path()).unwrap()).unwrap(),
                original
            );
            assert_eq!(std::fs::read(caller.path()).unwrap(), b"owned-ok");
        }
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn shared_anonymous_postcheck_windows() {
        for selector in 0..4 {
            for (cut, fault, expected, visible) in [
                ("L", false, "CWFLD", true),
                ("release", false, "CWFLD", true),
                ("W", true, "CW", false),
                ("F", true, "CWF", false),
                ("L", true, "CWFL", false),
                ("c6", true, "C", false),
            ] {
                let d =
                    SyntheticDirectory::native(&format!("shared-actor-{selector}-{cut}-{fault}"));
                let caller = Caller::new(&d, selector);
                let parent = caller.path().parent().unwrap().to_path_buf();
                let (tx, rx, thread) = actor(parent.clone());
                let events = Arc::new(Mutex::new(Vec::new()));
                let copy = events.clone();
                let fixture_parent = parent.clone();
                let scope = test_hooks::Scope::install(move |s| {
                    if matches!(s, "C" | "W" | "F" | "L" | "D") {
                        copy.lock().unwrap().push(s);
                    }
                    // After Target claim_new, before anonymous C internals. This preserves
                    // the real profile-empty policy instead of fabricating an admissible remnant.
                    if s == "C" {
                        private_bytes(&fixture_parent.join(TEMP), b"sentinel");
                    }
                    if s == cut {
                        coordinate(&tx, &rx);
                        if fault {
                            return Err(std::io::Error::from_raw_os_error(libc::ENOSPC).into());
                        }
                    }
                    Ok(())
                });
                let result = caller.publish();
                drop(scope);
                thread.join().unwrap();
                assert_eq!(events.lock().unwrap().concat(), expected);
                assert_eq!(std::fs::read(parent.join(TEMP)).unwrap(), b"foreign!");
                assert_eq!(
                    std::fs::read(parent.parent().unwrap().join("moved")).unwrap(),
                    b"sentinel"
                );
                assert_eq!(caller.path().exists(), visible);
                if visible {
                    assert!(result.is_ok());
                    assert_eq!(std::fs::read(caller.path()).unwrap(), b"owned-ok");
                    assert_eq!(
                        inode(&File::open(caller.path()).unwrap()).unwrap(),
                        test_hooks::owned_identity().unwrap()
                    );
                } else {
                    assert!(matches!(result, Err(PublishError::BeforeLink)));
                }
            }
        }
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn shared_legacy_temp_preservation() {
        for name in [
            "shared.reference.json",
            "shared.diagnostic.json",
            "shared.failure.json",
            FINAL,
        ] {
            for count in 0..=64 {
                let d = SyntheticDirectory::native(&format!("legacy-{name}-{count}"));
                let id = if name == FINAL {
                    "target-lock"
                } else {
                    "shared"
                };
                for i in 0..count {
                    private_bytes(&d.path.join(format!(".tmp-{id}-{i}")), b"foreign!");
                }
                // Intentionally low-level: Target claim_new must still reject remnants.
                let mut n = Native::new(&d.file, name.into());
                publish_ops(&mut n, b"owned-ok").unwrap();
                n.confirm_final().unwrap();
                drop(n);
                for i in 0..count {
                    assert_eq!(
                        std::fs::read(d.path.join(format!(".tmp-{id}-{i}"))).unwrap(),
                        b"foreign!"
                    );
                }
                assert_eq!(std::fs::read_dir(&d.path).unwrap().count(), count + 1);
            }
        }
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn anonymous_create_substep_failures() {
        for (index, fault) in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
            .into_iter()
            .enumerate()
        {
            let d = SyntheticDirectory::native(&format!("create-cut-{fault}"));
            let events = Arc::new(Mutex::new(Vec::new()));
            let copy = events.clone();
            let _scope = test_hooks::Scope::install(move |s| {
                if matches!(s, "c1" | "c2" | "c3" | "c4" | "c5" | "c6" | "c7" | "c8") {
                    copy.lock().unwrap().push(s);
                }
                ensure(s != fault, "injected C substep failure")
            });
            let mut n = Native::new(&d.file, FINAL.into());
            assert!(matches!(
                publish_ops(&mut n, b"owned-ok"),
                Err(PublishError::BeforeLink)
            ));
            assert_eq!(events.lock().unwrap().len(), index + 1);
            assert!(n.file.is_none());
            assert!(n.proc_fds.is_none());
            assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
        }
        let d = SyntheticDirectory::native("create-transfer");
        let _scope = fail("c5-owned");
        let mut n = Native::new(&d.file, FINAL.into());
        assert!(n.create().is_err());
        assert!(n.file.is_none());
        assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn anonymous_create_refusal() {
        for errno in [
            libc::EOPNOTSUPP,
            libc::ENOSPC,
            libc::EROFS,
            libc::EACCES,
            libc::EINVAL,
            libc::ENOENT,
        ] {
            let d = SyntheticDirectory::native(&format!("create-errno-{errno}"));
            let _scope = test_hooks::Scope::install(move |s| {
                if s == "c5" {
                    return Err(std::io::Error::from_raw_os_error(errno).into());
                }
                Ok(())
            });
            let mut n = Native::new(&d.file, FINAL.into());
            assert!(matches!(
                publish_ops(&mut n, b"owned-ok"),
                Err(PublishError::BeforeLink)
            ));
            assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
        }
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn proc_capture_refusal() {
        // Scripted safe-call failure positions; no host proc/mount/security changes.
        for (label, cut, errno) in [
            ("missing", "c1", libc::ENOENT),
            ("symlink", "c1", libc::ELOOP),
            ("nonproc", "c2", libc::EINVAL),
            ("enosys", "c3", libc::ENOSYS),
            ("eperm", "c3", libc::EPERM),
            ("beneath", "c3", libc::EXDEV),
            ("noxdev", "c3", libc::EXDEV),
            ("magic", "c3", libc::ELOOP),
            ("nonproc-fd", "c4", libc::EINVAL),
            ("missing-fd", "c8", libc::ENOENT),
            ("wrong-inode", "c8", libc::EINVAL),
        ] {
            let d = SyntheticDirectory::native(&format!("proc-{label}"));
            let _scope = test_hooks::Scope::install(move |s| {
                if s == cut {
                    return Err(std::io::Error::from_raw_os_error(errno).into());
                }
                Ok(())
            });
            let mut n = Native::new(&d.file, FINAL.into());
            assert!(matches!(
                publish_ops(&mut n, b"owned-ok"),
                Err(PublishError::BeforeLink)
            ));
            assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
        }
        let d = SyntheticDirectory::native("proc-origin");
        let (_scope, events) = record();
        let mut n = Native::new(&d.file, FINAL.into());
        n.create().unwrap();
        n.write(b"owned-ok").unwrap();
        n.file_sync().unwrap();
        n.proc_fds.as_mut().unwrap().invalidate_process_for_test();
        assert!(n.link().is_err());
        let ops = events
            .lock()
            .unwrap()
            .iter()
            .copied()
            .filter(|s| matches!(*s, "C" | "W" | "F" | "L" | "D"))
            .collect::<Vec<_>>();
        assert_eq!(ops, ["C", "W", "F", "L"]);
        assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
    }
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn anonymous_success_records_five_operations_and_eight_create_substeps() {
        let d = SyntheticDirectory::native("record-order");
        let (_scope, events) = record();
        let mut n = Native::new(&d.file, FINAL.into());
        publish_ops(&mut n, b"owned-ok").unwrap();
        n.confirm_final().unwrap();
        assert_eq!(
            *events.lock().unwrap(),
            [
                "C",
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "c5-owned",
                "c6",
                "c7",
                "c8",
                "c8-done",
                "W",
                "F",
                "L",
                "D",
                "final-check"
            ]
        );
    }
}

/// Pure lifetime model around the real five-operation driver. This checks owner
/// scope/release accounting; it cannot establish Linux FD identity or reclamation.
#[test]
fn anonymous_owned_lifetime_model() {
    use std::{
        cell::Cell,
        panic::{AssertUnwindSafe, catch_unwind},
        rc::Rc,
    };
    struct OwnedModel {
        ops: FiveOps,
        acquired: bool,
        releases: Rc<Cell<usize>>,
    }
    impl Drop for OwnedModel {
        fn drop(&mut self) {
            if self.acquired {
                self.releases.set(self.releases.get() + 1);
            }
        }
    }
    impl Ops for OwnedModel {
        fn create(&mut self) -> Result<()> {
            self.ops.create()?;
            self.acquired = true;
            Ok(())
        }
        fn write(&mut self, b: &[u8]) -> Result<()> {
            self.ops.write(b)
        }
        fn file_sync(&mut self) -> Result<()> {
            self.ops.file_sync()
        }
        fn link(&mut self) -> Result<()> {
            self.ops.link()
        }
        fn dir_sync(&mut self) -> Result<()> {
            self.ops.dir_sync()
        }
    }
    for cut in ["success", "W", "F", "L", "D", "unused", "unwind", "lost"] {
        let releases = Rc::new(Cell::new(0));
        let mut owner = OwnedModel {
            ops: FiveOps {
                fail: match cut {
                    "W" => Some(b'W'),
                    "F" => Some(b'F'),
                    "L" => Some(b'L'),
                    "D" => Some(b'D'),
                    _ => None,
                },
                ..Default::default()
            },
            acquired: false,
            releases: releases.clone(),
        };
        if cut == "unwind" {
            assert!(
                catch_unwind(AssertUnwindSafe(move || {
                    owner.create().unwrap();
                    panic!("scripted pre-link unwind");
                }))
                .is_err()
            );
        } else {
            if cut == "unused" {
                owner.create().unwrap();
                assert!(!owner.ops.final_visible);
            } else if cut == "lost" {
                owner.create().unwrap();
                owner.write(b"owned-ok").unwrap();
                owner.file_sync().unwrap();
                owner.link().unwrap();
                assert!(owner.ops.final_visible);
            } else {
                let result = publish_ops(&mut owner, b"owned-ok");
                assert_eq!(result.is_ok(), cut == "success");
            }
            assert_eq!(releases.get(), 0);
            drop(owner);
        }
        assert_eq!(releases.get(), 1, "{cut}");
    }
}
