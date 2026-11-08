//! Exact operational source package. Role tags never select new origins.
use crate::{
    study_provenance::{Result, ensure},
    study_reference::operational::FileReference,
};
pub(crate) fn package_path(r: &FileReference) -> Result<()> {
    ensure(
        r.root == 3
            && r.path.starts_with("resource/linux-cpu-observed-v1/")
            && super::super::codec::relative(r.path.as_bytes())
            && r.length <= 8 * 1024 * 1024,
        "operational package Policy path",
    )
}
pub(crate) fn receipt_join(
    actual: &FileReference,
    expected: &FileReference,
    held_parent: (u64, u64),
    current_parent: (u64, u64),
) -> Result<()> {
    ensure(
        expected.root == 4
            && !expected.path.contains('/')
            && actual == expected
            && held_parent == current_parent,
        "native current receipt/Authority origin mismatch",
    )
}
#[cfg(target_os = "linux")]
#[path = "package_linux.rs"]
mod native;
#[cfg(target_os = "linux")]
pub(crate) use native::Package;
