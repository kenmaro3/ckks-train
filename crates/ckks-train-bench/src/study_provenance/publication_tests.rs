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
        self.temp = true;
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
        let e = if self.temp { "dir_sync1" } else { "dir_sync2" };
        self.events.push(e);
        ensure(self.fail != Some(e), "injected")
    }
    fn unlink_owned(&mut self) -> Result<()> {
        self.events.push("unlink");
        ensure(self.fail != Some("unlink"), "injected")?;
        self.temp = false;
        Ok(())
    }
}
#[test]
fn every_publication_fault_withholds_capability_and_preserves_final() {
    for step in [
        "create",
        "write",
        "file_sync",
        "link",
        "dir_sync1",
        "unlink",
        "dir_sync2",
    ] {
        let mut f = Fake {
            fail: Some(step),
            ..Default::default()
        };
        let r = publish_ops(&mut f, b"complete");
        assert!(r.is_err(), "{step}");
        if ["dir_sync1", "unlink", "dir_sync2"].contains(&step) {
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
        [
            "create",
            "write",
            "file_sync",
            "link",
            "dir_sync1",
            "unlink",
            "dir_sync2"
        ]
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
    assert_eq!(f.events, ["file_sync", "dir_sync1"]);
    f.fail = Some("dir_sync1");
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
        temp: bool,
    }
    impl Ops for Race {
        fn create(&mut self) -> Result<()> {
            self.temp = true;
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
        fn unlink_owned(&mut self) -> Result<()> {
            assert!(self.temp);
            self.temp = false;
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
                temp: false,
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
