//! Fixed private checkpoints; compiled out of production. Never syscall-success forgery.
#![forbid(unsafe_code)]
use super::{File, Result, inode};
use std::cell::RefCell;
type Observer = Box<dyn FnMut(&'static str) -> Result<()>>;
thread_local! {
    static OBSERVER: RefCell<Option<Observer>> = RefCell::new(None);
    static OWNED: RefCell<Option<(u64,u64)>> = const { RefCell::new(None) };
}
pub(super) fn owned(file: &File) -> Result<()> {
    let identity = inode(file)?;
    OWNED.with(|o| *o.borrow_mut() = Some(identity));
    Ok(())
}
pub(super) fn owned_identity() -> Option<(u64, u64)> {
    OWNED.with(|o| *o.borrow())
}

pub(super) fn checkpoint(name: &'static str) -> Result<()> {
    OBSERVER.with(|o| match o.borrow_mut().as_mut() {
        Some(observer) => observer(name),
        None => Ok(()),
    })
}
pub(super) struct Scope;
impl Scope {
    pub(super) fn install(observer: impl FnMut(&'static str) -> Result<()> + 'static) -> Self {
        OBSERVER.with(|o| {
            assert!(o.borrow().is_none());
            *o.borrow_mut() = Some(Box::new(observer));
        });
        OWNED.with(|o| *o.borrow_mut() = None);
        Self
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        OBSERVER.with(|o| *o.borrow_mut() = None);
    }
}
