use super::*;
#[derive(Default)]
struct Fake {
    fail: Option<&'static str>,
    events: Vec<&'static str>,
    exists: bool,
    temp: bool,
    final_bytes: Vec<u8>,
    pending: Vec<u8>,
}
impl Ops for Fake {
    fn create(&mut self) -> Result<()> {
        self.events.push("create");
        if self.fail == Some("create") {
            return Err(ProvenanceError("injected"));
        }
        Ok(())
    }
    fn write(&mut self, b: &[u8]) -> Result<()> {
        self.events.push("write");
        if self.fail == Some("write") {
            return Err(ProvenanceError("injected"));
        }
        self.pending = b.to_vec();
        Ok(())
    }
    fn file_sync(&mut self) -> Result<()> {
        self.events.push("file_sync");
        ensure(self.fail != Some("file_sync"), "injected")
    }
    fn link(&mut self) -> Result<()> {
        self.events.push("link");
        ensure(self.fail != Some("link") && !self.exists, "collision")?;
        self.exists = true;
        self.final_bytes = self.pending.clone();
        Ok(())
    }
    fn dir_sync(&mut self) -> Result<()> {
        self.events.push("dir_sync");
        ensure(self.fail != Some("dir_sync"), "injected")
    }
}
#[test]
fn every_publication_fault_withholds_capability_and_preserves_final() {
    for step in ["create", "write", "file_sync", "link", "dir_sync"] {
        let mut f = Fake {
            fail: Some(step),
            ..Default::default()
        };
        let r = publish_ops(&mut f, b"complete");
        assert!(r.is_err(), "{step}");
        if step == "dir_sync" {
            assert!(matches!(r, Err(PublishError::VisibleUnconfirmed)));
            assert_eq!(f.final_bytes, b"complete");
        } else {
            assert!(!f.exists);
        }
    }
    let mut f = Fake::default();
    publish_ops(&mut f, b"complete").unwrap();
    assert_eq!(
        f.events,
        ["create", "write", "file_sync", "link", "dir_sync"]
    );
    assert!(!f.temp);
}
#[test]
fn competing_publishers_never_replace_first_final() {
    let mut f = Fake::default();
    publish_ops(&mut f, b"first").unwrap();
    assert!(publish_ops(&mut f, b"second").is_err());
    assert_eq!(f.final_bytes, b"first");
}
#[test]
fn recovery_never_cleans_any_temporary_name() {
    let mut f = Fake {
        exists: true,
        temp: true,
        final_bytes: b"complete".to_vec(),
        ..Default::default()
    };
    recover_ops(&mut f).unwrap();
    assert!(f.temp);
    assert_eq!(f.events, ["file_sync", "dir_sync"]);
    f.fail = Some("dir_sync");
    assert!(recover_ops(&mut f).is_err());
    assert!(f.temp);
}
#[test]
fn publication_mount_equality_is_not_an_ancestor_requirement() {
    let root = Mount { device: 4, id: 9 };
    let kind = Mount { device: 4, id: 9 };
    assert!(same_publication_mount(root, kind).is_ok());
    assert!(same_publication_mount(root, Mount { device: 4, id: 10 }).is_err());
    assert!(same_publication_mount(root, Mount { device: 5, id: 9 }).is_err());
    assert!(mount_type("ext4", 0xef53).is_ok());
    assert!(mount_type("ext3", 0xef53).is_err());
    assert!(mount_type("xfs", 0x58465342).is_ok());
    assert!(mount_type("overlay", 0x794c7630).is_err());
}
#[test]
fn fifo_and_nonregular_reads_fail_without_waiting() {
    use std::os::unix::fs::PermissionsExt;
    let p = std::env::temp_dir().join(format!("hekit-d6c2a-fifo-{}", std::process::id()));
    std::fs::create_dir(&p).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o700)).unwrap();
    let fifo = p.join("fifo");
    let c = cstring(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    assert!(read_path(&fifo, 1024).is_err());
    assert!(read_path(&p, 1024).is_err());
    std::fs::remove_file(fifo).unwrap();
    std::fs::remove_dir(p).unwrap();
}
#[test]
fn short_writes_and_eintr_are_retried_without_losing_bytes() {
    struct Short {
        bytes: Vec<u8>,
        interrupted: bool,
    }
    impl std::io::Write for Short {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let n = b.len().min(2);
            self.bytes.extend_from_slice(&b[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Short {
        bytes: vec![],
        interrupted: false,
    };
    write_complete(&mut writer, b"complete bytes").unwrap();
    assert_eq!(writer.bytes, b"complete bytes");
}
#[test]
fn concurrent_synthetic_publishers_get_exactly_one_capability() {
    use std::sync::{Arc, Barrier, Mutex};
    struct Race {
        shared: Arc<Mutex<Option<Vec<u8>>>>,
        barrier: Arc<Barrier>,
        bytes: Vec<u8>,
    }
    impl Ops for Race {
        fn create(&mut self) -> Result<()> {
            Ok(())
        }
        fn write(&mut self, b: &[u8]) -> Result<()> {
            self.bytes = b.into();
            Ok(())
        }
        fn file_sync(&mut self) -> Result<()> {
            Ok(())
        }
        fn link(&mut self) -> Result<()> {
            self.barrier.wait();
            let mut shared = self.shared.lock().unwrap();
            ensure(shared.is_none(), "collision")?;
            *shared = Some(self.bytes.clone());
            Ok(())
        }
        fn dir_sync(&mut self) -> Result<()> {
            Ok(())
        }
    }
    let shared = Arc::new(Mutex::new(None));
    let barrier = Arc::new(Barrier::new(2));
    let handles = (0..2)
        .map(|i| {
            let mut race = Race {
                shared: shared.clone(),
                barrier: barrier.clone(),
                bytes: vec![],
            };
            std::thread::spawn(move || publish_ops(&mut race, &[i]).is_ok())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        handles
            .into_iter()
            .filter_map(|h| h.join().ok())
            .filter(|b| *b)
            .count(),
        1
    );
    assert!(
        shared
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|b| b == &[0] || b == &[1])
    );
}
#[test]
fn recovery_file_sync_failure_preserves_temporary_name() {
    let mut f = Fake {
        fail: Some("file_sync"),
        temp: true,
        exists: true,
        ..Default::default()
    };
    assert!(recover_ops(&mut f).is_err());
    assert!(f.temp);
    assert_eq!(f.events, ["file_sync"]);
}
#[test]
fn target_stamped_read_rejects_byte_identical_inode_replacement() {
    use std::os::unix::fs::PermissionsExt;
    let p = std::env::temp_dir().join(format!(
        "hekit-synthetic-target-stamp-{}",
        std::process::id()
    ));
    std::fs::create_dir(&p).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(p.join("reference.json"), b"literal synthetic bytes").unwrap();
    std::fs::set_permissions(
        p.join("reference.json"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    let parent = File::open(&p).unwrap();
    let (_, before) = read_stamped_at(&parent, "reference.json", 1024).unwrap();
    std::fs::rename(p.join("reference.json"), p.join("old.json")).unwrap();
    std::fs::write(p.join("reference.json"), b"literal synthetic bytes").unwrap();
    std::fs::set_permissions(
        p.join("reference.json"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    let (_, after) = read_stamped_at(&parent, "reference.json", 1024).unwrap();
    assert_ne!(before, after);
    std::fs::remove_dir_all(p).unwrap();
}

/// Portable native syscall checks inside an explicit private synthetic namespace.
/// They never bypass a production context/mount gate or construct scientific capabilities.
struct SyntheticDirectory {
    path: PathBuf,
    file: File,
    cleanup: PathBuf,
}
impl SyntheticDirectory {
    fn new(label: &str) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let path = std::env::temp_dir().join(format!(
            "hekit-synthetic-target-{label}-{}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let file = File::open(&path).unwrap();
        let cleanup = path.clone();
        Self {
            path,
            file,
            cleanup,
        }
    }
    #[cfg(target_os = "linux")]
    fn native(label: &str) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let root = PathBuf::from(
            std::env::var_os("HEKIT_PUBLICATION_NATIVE_SCRATCH")
                .expect("separately authorized native scratch root required"),
        );
        let root_fd = File::open(&root).unwrap();
        safe_directory(&root_fd, true).unwrap();
        assert_ne!(
            root_fd.metadata().unwrap().uid(),
            0,
            "ordinary unprivileged native account required"
        );
        mount(&root_fd).unwrap();
        let cleanup = root.join(format!("hekit-publication-{label}-{}", std::process::id()));
        std::fs::create_dir(&cleanup).unwrap();
        std::fs::set_permissions(&cleanup, std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = cleanup.join("attempt");
        std::fs::create_dir(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let file = File::open(&path).unwrap();
        same_publication_mount(mount(&root_fd).unwrap(), mount(&file).unwrap()).unwrap();
        Self {
            path,
            file,
            cleanup,
        }
    }
    fn guard(&self) -> (u64, u64) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(self.path.join(".profile-guard"), []).unwrap();
        std::fs::set_permissions(
            self.path.join(".profile-guard"),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        inode(&File::open(self.path.join(".profile-guard")).unwrap()).unwrap()
    }
}
impl Drop for SyntheticDirectory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.cleanup).unwrap();
    }
}
#[test]
fn target_unknown_entries_block_new_claim_without_removal() {
    let d = SyntheticDirectory::new("states");
    d.guard();
    profile_empty(&d.file).unwrap();
    for name in ["cases", "target.lock.json", ".tmp-target-lock-0", "unknown"] {
        let p = d.path.join(name);
        std::fs::write(&p, b"orphan").unwrap();
        assert!(profile_empty(&d.file).is_err());
        assert_eq!(std::fs::read(&p).unwrap(), b"orphan");
        std::fs::remove_file(p).unwrap();
    }
    std::fs::create_dir(d.path.join("cases")).unwrap();
    assert!(profile_empty(&d.file).is_err());
}
#[test]
fn target_nofollow_regular_owner_mode_and_guard_link_count_checks() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let d = SyntheticDirectory::new("types");
    let p = d.path.join("target.lock.json");
    std::fs::write(&p, b"synthetic").unwrap();
    for mode in [0o644, 0o666, 0o400] {
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
        assert!(read_stamped_at(&d.file, "target.lock.json", 64).is_err());
    }
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
    read_stamped_at(&d.file, "target.lock.json", 64).unwrap();
    std::fs::hard_link(&p, d.path.join("hardlink")).unwrap();
    read_stamped_at(&d.file, "target.lock.json", 64).unwrap();
    std::fs::remove_file(d.path.join("hardlink")).unwrap();
    std::fs::rename(&p, d.path.join("real")).unwrap();
    symlink("real", &p).unwrap();
    assert!(read_stamped_at(&d.file, "target.lock.json", 64).is_err());
    std::fs::remove_file(&p).unwrap();
    let name = cstring(p.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(read_stamped_at(&d.file, "target.lock.json", 64).is_err());
    std::fs::remove_file(&p).unwrap();
    std::fs::create_dir(&p).unwrap();
    assert!(read_stamped_at(&d.file, "target.lock.json", 64).is_err());
}
#[test]
fn target_ancestor_and_profile_swaps_reject_through_held_parent() {
    use std::os::unix::fs::symlink;
    let d = SyntheticDirectory::new("swap");
    let name = OsStr::new("profile");
    std::fs::create_dir(d.path.join(name)).unwrap();
    let held = File::open(d.path.join(name)).unwrap();
    check_child(&d.file, &held, name).unwrap();
    std::fs::rename(d.path.join(name), d.path.join("old")).unwrap();
    std::fs::create_dir(d.path.join(name)).unwrap();
    assert!(check_child(&d.file, &held, name).is_err());
    std::fs::remove_dir(d.path.join(name)).unwrap();
    symlink("old", d.path.join(name)).unwrap();
    assert!(check_child(&d.file, &held, name).is_err());
}
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
fn target_native_engine_preserves_unknown_temp_and_does_not_overwrite() {
    let d = SyntheticDirectory::native("engine");
    std::fs::write(d.path.join(".tmp-target-lock-0"), b"unknown").unwrap();
    let make = || Native::new(&d.file, "target.lock.json".into());
    publish_ops(&mut make(), b"synthetic target bytes").unwrap();
    assert!(publish_ops(&mut make(), b"replacement").is_err());
    assert_eq!(
        std::fs::read(d.path.join("target.lock.json")).unwrap(),
        b"synthetic target bytes"
    );
    assert_eq!(
        std::fs::read(d.path.join(".tmp-target-lock-0")).unwrap(),
        b"unknown"
    );
}
// The old named-temp before-check test is archived verbatim in the author ledger.
// The migrated publisher owns anonymous O; T is an independent actor sentinel.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
fn target_actor_sentinel_replacement_cannot_redirect_owned_inode() {
    let d = SyntheticDirectory::native("owned-temp");
    let p = d.path.join(".tmp-target-lock-0");
    std::fs::write(&p, b"sentinel").unwrap();
    let mut o = Native::new(&d.file, "target.lock.json".into());
    o.create().unwrap();
    o.write(b"owned-ok").unwrap();
    o.file_sync().unwrap();
    std::fs::rename(&p, d.path.parent().unwrap().join("moved")).unwrap();
    std::fs::write(&p, b"foreign!").unwrap();
    o.link().unwrap();
    o.dir_sync().unwrap();
    o.confirm_final().unwrap();
    drop(o);
    assert_eq!(std::fs::read(p).unwrap(), b"foreign!");
    assert_eq!(
        std::fs::read(d.path.join("target.lock.json")).unwrap(),
        b"owned-ok"
    );
    assert_eq!(
        std::fs::read(d.path.parent().unwrap().join("moved")).unwrap(),
        b"sentinel"
    );
}
#[test]
fn target_fresh_guard_fds_serialize_threads_and_other_profiles_do_not_wait() {
    use std::{sync::mpsc, time::Duration};
    let a = SyntheticDirectory::new("flock-a");
    let b = SyntheticDirectory::new("flock-b");
    let ia = a.guard();
    let ib = b.guard();
    let held = fresh_guard(&a.file, ia).unwrap();
    let file = a.file.try_clone().unwrap();
    let (tx, rx) = mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let guard = fresh_guard(&file, ia).unwrap();
        tx.send(()).unwrap();
        drop(guard);
    });
    assert!(rx.recv_timeout(Duration::from_millis(40)).is_err());
    let independent = fresh_guard(&b.file, ib).unwrap();
    drop(independent);
    drop(held);
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    waiter.join().unwrap();
}
#[cfg(target_os = "linux")]
#[test]
fn target_native_linux_guard_child() {
    let Some(path) = std::env::var_os("HEKIT_SYNTHETIC_GUARD_CHILD") else {
        return;
    };
    let parent = File::open(&path).unwrap();
    let guard_file = openat(
        parent.as_raw_fd(),
        OsStr::new(".profile-guard"),
        FILE_FLAGS,
        0,
    )
    .unwrap();
    let _guard = fresh_guard(&parent, inode(&guard_file).unwrap()).unwrap();
    std::fs::write(Path::new(&path).join("child-acquired"), b"synthetic").unwrap();
}
#[cfg(target_os = "linux")]
#[test]
fn target_native_linux_separate_process_uses_fresh_flock_description() {
    use std::{process::Command, time::Duration};
    let d = SyntheticDirectory::new("process-lock");
    let expected = d.guard();
    let guard = fresh_guard(&d.file, expected).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "study_provenance::publication::tests::target_native_linux_guard_child",
            "--test-threads=1",
        ])
        .env("HEKIT_SYNTHETIC_GUARD_CHILD", &d.path)
        .spawn()
        .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert!(!d.path.join("child-acquired").exists());
    drop(guard);
    assert!(child.wait().unwrap().success());
    assert_eq!(
        std::fs::read(d.path.join("child-acquired")).unwrap(),
        b"synthetic"
    );
}

#[test]
fn target_guard_rejects_nonempty_multiple_links_and_replaced_inode() {
    let d = SyntheticDirectory::new("guard-identity");
    let expected = d.guard();
    let path = d.path.join(".profile-guard");
    std::fs::hard_link(&path, d.path.join("extra-link")).unwrap();
    assert!(fresh_guard(&d.file, expected).is_err());
    std::fs::remove_file(d.path.join("extra-link")).unwrap();
    std::fs::write(&path, b"not empty").unwrap();
    assert!(fresh_guard(&d.file, expected).is_err());
    std::fs::rename(&path, d.path.join("old-guard")).unwrap();
    d.guard();
    assert!(fresh_guard(&d.file, expected).is_err());
}
#[test]
fn target_recovery_preserves_linked_owned_orphan_after_postlink_failure() {
    let d = SyntheticDirectory::new("recovery-orphan");
    use std::os::unix::fs::PermissionsExt;
    let temp = d.path.join(".tmp-target-lock-0");
    std::fs::write(&temp, b"synthetic exact expected bytes").unwrap();
    std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::hard_link(&temp, d.path.join("target.lock.json")).unwrap();
    let mut o = Native::recovery(
        &d.file,
        "target.lock.json".into(),
        File::open(&temp).unwrap(),
    );
    let (bytes, _) = read_stamped_at(&d.file, "target.lock.json", 1024).unwrap();
    assert_eq!(bytes, b"synthetic exact expected bytes");
    recover_ops(&mut o).unwrap();
    assert!(temp.exists());
    assert_eq!(
        std::fs::metadata(d.path.join("target.lock.json"))
            .unwrap()
            .nlink(),
        2
    );
}
#[cfg(target_os = "linux")]
fn synthetic_guarded_publish(parent: &File, expected: (u64, u64)) -> bool {
    let Ok(_guard) = fresh_guard(parent, expected) else {
        return false;
    };
    if profile_empty(parent).is_err() {
        return false;
    }
    let mut o = Native::new(parent, "target.lock.json".into());
    publish_ops(&mut o, b"PRIVATE SYNTHETIC lock bytes").is_ok()
}
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
fn target_independent_thread_handles_publish_exactly_once_under_guard() {
    use std::sync::{Arc, Barrier};
    let d = SyntheticDirectory::native("thread-race");
    let expected = d.guard();
    let barrier = Arc::new(Barrier::new(2));
    let threads = (0..2)
        .map(|_| {
            let parent = File::open(&d.path).unwrap();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                synthetic_guarded_publish(&parent, expected)
            })
        })
        .collect::<Vec<_>>();
    let winners = threads
        .into_iter()
        .map(|t| usize::from(t.join().unwrap()))
        .sum::<usize>();
    assert_eq!(winners, 1);
    assert_eq!(
        std::fs::read(d.path.join("target.lock.json")).unwrap(),
        b"PRIVATE SYNTHETIC lock bytes"
    );
}
#[cfg(target_os = "linux")]
#[test]
#[ignore = "internal native child; invoke only through separately released exact parent harness"]
fn target_native_linux_publication_child() {
    let path = std::env::var_os("HEKIT_SYNTHETIC_PUBLISH_CHILD")
        .expect("native publication child requires its parent fixture");
    let parent = File::open(&path).unwrap();
    let file = openat(
        parent.as_raw_fd(),
        OsStr::new(".profile-guard"),
        FILE_FLAGS,
        0,
    )
    .unwrap();
    std::process::exit(
        // Dedicated nonzero outcomes distinguish a real helper from a harness
        // selecting zero tests (which exits 0).
        if synthetic_guarded_publish(&parent, inode(&file).unwrap()) {
            70
        } else {
            71
        },
    );
}
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
fn target_native_linux_processes_publish_exactly_once() {
    use std::process::Command;
    let d = SyntheticDirectory::native("process-race");
    d.guard();
    let children = (0..2)
        .map(|_| {
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "--ignored",
                    "study_provenance::publication::tests::target_native_linux_publication_child",
                    "--test-threads=1",
                ])
                .env("HEKIT_SYNTHETIC_PUBLISH_CHILD", &d.path)
                .spawn()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let statuses = children
        .into_iter()
        .map(|mut child| child.wait().unwrap().code().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(statuses.iter().filter(|&&s| s == 70).count(), 1);
    assert_eq!(statuses.iter().filter(|&&s| s == 71).count(), 1);
    assert_eq!(
        std::fs::read(d.path.join("target.lock.json")).unwrap(),
        b"PRIVATE SYNTHETIC lock bytes"
    );
}
#[test]
fn target_directory_creation_never_repairs_an_existing_unsafe_object() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let d = SyntheticDirectory::new("private-child");
    let child = private_child(&d.file, "new").unwrap();
    assert_eq!(child.metadata().unwrap().mode() & 0o7777, 0o700);
    std::fs::set_permissions(d.path.join("new"), std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(private_child(&d.file, "new").is_err());
    assert_eq!(
        std::fs::metadata(d.path.join("new")).unwrap().mode() & 0o7777,
        0o755
    );
    symlink("new", d.path.join("alias")).unwrap();
    assert!(private_child(&d.file, "alias").is_err());
    std::fs::write(d.path.join("file"), b"unknown").unwrap();
    assert!(private_child(&d.file, "file").is_err());
    assert_eq!(std::fs::read(d.path.join("file")).unwrap(), b"unknown");
}

#[test]
fn reader_guard_open_policy_is_derived_from_the_closed_role() {
    use super::super::role::ExecutableRole;
    assert_eq!(
        target_open_policy(ExecutableRole::Reference),
        TargetOpenPolicy::CreateReference
    );
    assert_eq!(
        target_open_policy(ExecutableRole::StudyTrain),
        TargetOpenPolicy::ExistingTrain
    );
}

#[test]
fn reader_guard_existing_children_never_create_or_repair() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let d = SyntheticDirectory::new("read-existing-child");
    for name in ["campaigns", "campaign", "profiles", "p-slot"] {
        assert!(existing_private_child(&d.file, name).is_err());
        assert!(!d.path.join(name).exists());
    }
    std::fs::create_dir(d.path.join("ready")).unwrap();
    std::fs::set_permissions(d.path.join("ready"), std::fs::Permissions::from_mode(0o700)).unwrap();
    let f = existing_private_child(&d.file, "ready").unwrap();
    assert_eq!(
        inode(&f).unwrap(),
        inode(&File::open(d.path.join("ready")).unwrap()).unwrap()
    );
    symlink("ready", d.path.join("alias")).unwrap();
    assert!(existing_private_child(&d.file, "alias").is_err());
    std::fs::write(d.path.join("ordinary"), b"untouched").unwrap();
    assert!(existing_private_child(&d.file, "ordinary").is_err());
    std::fs::set_permissions(d.path.join("ready"), std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(existing_private_child(&d.file, "ready").is_err());
    assert_eq!(
        std::fs::metadata(d.path.join("ready")).unwrap().mode() & 0o7777,
        0o755
    );
    assert_eq!(
        std::fs::read(d.path.join("ordinary")).unwrap(),
        b"untouched"
    );
}

#[test]
fn reader_guard_existing_guard_never_creates_repairs_or_follows_links() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let d = SyntheticDirectory::new("read-existing-guard");
    assert!(existing_profile_guard(&d.file).is_err());
    assert!(std::fs::read_dir(&d.path).unwrap().next().is_none());
    let expected = d.guard();
    assert_eq!(
        inode(&existing_profile_guard(&d.file).unwrap()).unwrap(),
        expected
    );
    std::fs::hard_link(d.path.join(".profile-guard"), d.path.join("extra-link")).unwrap();
    assert!(existing_profile_guard(&d.file).is_err());
    std::fs::remove_file(d.path.join("extra-link")).unwrap();
    std::fs::write(d.path.join(".profile-guard"), b"unknown").unwrap();
    assert!(existing_profile_guard(&d.file).is_err());
    assert_eq!(
        std::fs::read(d.path.join(".profile-guard")).unwrap(),
        b"unknown"
    );
    std::fs::write(d.path.join(".profile-guard"), []).unwrap();
    std::fs::set_permissions(
        d.path.join(".profile-guard"),
        std::fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    assert!(existing_profile_guard(&d.file).is_err());
    assert_eq!(
        std::fs::metadata(d.path.join(".profile-guard"))
            .unwrap()
            .mode()
            & 0o7777,
        0o644
    );
    std::fs::rename(d.path.join(".profile-guard"), d.path.join("preserved")).unwrap();
    symlink("preserved", d.path.join(".profile-guard")).unwrap();
    assert!(existing_profile_guard(&d.file).is_err());
}

#[path = "publication_anonymous_tests.rs"]
mod anonymous_contract;
