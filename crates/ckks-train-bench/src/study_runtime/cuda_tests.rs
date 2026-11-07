use super::cuda::*;
use std::{
    fs,
    io::Cursor,
    os::unix::fs::symlink,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "ckks-runtime-module-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p.canonicalize().unwrap())
    }
    fn file(&self) -> PathBuf {
        self.0.join("module")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn cuda_module_bounded_hash_rejects_growth_without_prefix_success() {
    assert_eq!(
        hash_bounded(&mut Cursor::new(b"abc"), 3).unwrap(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert!(hash_bounded(&mut Cursor::new(b"abcd"), 3).is_err());
    assert!(hash_bounded(&mut Cursor::new(b"a"), 0).is_err());
}
#[test]
fn cuda_module_descriptor_rechecks_replacement_and_in_place_change() {
    let dir = Temp::new();
    let p = dir.file();
    fs::write(&p, b"abc").unwrap();
    let held = RetainedModule::open(&p).unwrap();
    held.recheck().unwrap();
    fs::write(&p, b"abd").unwrap();
    assert!(held.recheck().is_err());
    let held = RetainedModule::open(&p).unwrap();
    fs::rename(&p, dir.0.join("old")).unwrap();
    fs::write(&p, b"abd").unwrap();
    assert!(held.recheck().is_err());
}
#[test]
fn cuda_module_refuses_symlink_directory_and_oversize_before_hash() {
    let dir = Temp::new();
    let p = dir.file();
    fs::write(&p, b"abc").unwrap();
    let alias = dir.0.join("alias");
    symlink(&p, &alias).unwrap();
    assert!(RetainedModule::open(&alias).is_err());
    assert!(RetainedModule::open(&dir.0).is_err());
    let f = fs::OpenOptions::new().write(true).open(&p).unwrap();
    f.set_len(MODULE_CAP + 1).unwrap();
    assert!(RetainedModule::open(&p).is_err());
}
#[test]
fn cuda_module_owner_null_and_substitution_are_rejected() {
    assert!(resolve_owner(std::ptr::null()).is_err());
    extern "C" fn marker() {}
    let owner = resolve_owner(marker as *const () as *const std::ffi::c_void).unwrap();
    require_same_owner(&owner, &owner).unwrap();
    let mut changed = owner.clone();
    changed.base = changed.base.wrapping_add(1);
    assert!(require_same_owner(&owner, &changed).is_err());
    let mut changed = owner.clone();
    changed.path = PathBuf::from("/wrong/module");
    assert!(require_same_owner(&owner, &changed).is_err());
}
