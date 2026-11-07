//! Future finite CPU block acceptance, ignored until independently released.
use super::{BlockPlanOwner, CpuBlockSession};

#[test]
#[ignore = "requires independent root CPU runtime release"]
fn p5_cpu_block_bridge_contract() {
    // RED: these actual bridge types must exist before any future execution.
    assert!(std::mem::size_of::<BlockPlanOwner>() > 0);
    assert!(std::mem::size_of::<CpuBlockSession<'static>>() > 0);
}
