//! Linux direct-child supervision and authenticated operational brokerage.
// The live consumers are Linux-only. Other hosts compile the inert formats and
// pure tests; the Linux strict-Clippy pass still checks live dead code/imports.
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]
#![forbid(unsafe_code)]
mod capacity;
mod clock_fault;
mod completion;
mod handshake;
mod history;
mod markers;
mod observation;
mod protocol;
mod schedule;
pub(crate) fn git_arguments(
    root: &[u8],
    suffix: &[&str],
) -> crate::study_provenance::Result<Vec<Vec<u8>>> {
    protocol::git_argv(root, suffix)
}
pub(crate) mod broker;
#[cfg(target_os = "linux")]
mod child;
#[cfg(target_os = "linux")]
mod controller;
#[cfg(target_os = "linux")]
mod monitor;
#[cfg(target_os = "linux")]
mod native;
pub(crate) fn launch(
    root: &std::path::Path,
    approval: &std::path::Path,
    hash: &str,
    receipts: &[std::path::PathBuf],
    args: &[std::ffi::OsString],
    work: crate::study_reference::operational::Requested,
) -> crate::study_provenance::Result<()> {
    #[cfg(target_os = "linux")]
    {
        controller::launch(root, approval, hash, receipts, args, work)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (root, approval, hash, receipts, args, work);
        Err(crate::study_provenance::ProvenanceError(
            "observed supervisor requires native Linux",
        ))
    }
}
pub(crate) fn worker(
    args: &[std::ffi::OsString],
    root: &std::path::Path,
    approval: &str,
    work: crate::study_reference::operational::Requested,
) -> crate::study_provenance::Result<broker::WorkerOwner> {
    #[cfg(target_os = "linux")]
    {
        child::worker(args, root, approval, work)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (args, root, approval, work);
        Err(crate::study_provenance::ProvenanceError(
            "private worker requires native Linux",
        ))
    }
}
pub(crate) fn helper(args: &[std::ffi::OsString]) -> crate::study_provenance::Result<()> {
    #[cfg(target_os = "linux")]
    {
        child::helper(args)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = args;
        Err(crate::study_provenance::ProvenanceError(
            "private helper requires native Linux",
        ))
    }
}
#[cfg(test)]
mod protocol_tests;
#[cfg(test)]
mod tests;

mod probe_plan;

mod probe_transfer;

#[cfg(target_os = "linux")]
mod broker_dispatch;

#[cfg(target_os = "linux")]
mod probe_child;

#[cfg(target_os = "linux")]
mod probe_adoption;
#[cfg(target_os = "linux")]
mod probe_driver;
pub(crate) fn probe_entry(
    args: &[std::ffi::OsString],
) -> Option<crate::study_provenance::Result<()>> {
    let flag = args.first()?;
    if flag != "--native-supervisor-check" && flag != "--private-native-probe" {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        Some(if flag == "--native-supervisor-check" {
            probe_driver::launch(args)
        } else {
            probe_child::prepare(args).and_then(|p| match p.role {
                "idle" => probe_child::idle(p),
                "middle" => probe_adoption::middle(p),
                "leaf" => probe_adoption::leaf(p),
                _ => Err(crate::study_provenance::ProvenanceError(
                    "unknown private native role",
                )),
            })
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        Some(Err(crate::study_provenance::ProvenanceError(
            "native probe requires Linux; no observation emitted",
        )))
    }
}
#[cfg(target_os = "linux")]
mod probe_index;
