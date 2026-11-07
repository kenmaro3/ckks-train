use super::*;
use std::cell::RefCell;

// Native calls are the only substituted layer. These tests execute the production
// query sequencing, validation and comparisons; no fake can build a public observation.
#[derive(Clone)]
struct Queries {
    trace: RefCell<Vec<&'static str>>,
    fail: Option<&'static str>,
    driver: i32,
    nvrtc: (i32, i32),
    count: i32,
    device: sys::CUdevice,
    uuid: [c_char; 16],
    pci: [c_char; 32],
    capability: (i32, i32),
}

impl Queries {
    fn valid() -> Self {
        let mut pci = [0; 32];
        for (dest, src) in pci.iter_mut().zip(b"0000:01:00.0") {
            *dest = *src as c_char;
        }
        Self {
            trace: RefCell::new(Vec::new()),
            fail: None,
            driver: 13010,
            nvrtc: (13, 0),
            count: 1,
            device: 0,
            uuid: [0x80u8 as c_char; 16],
            pci,
            capability: (8, 0),
        }
    }
    fn call(&self, name: &'static str) -> Result<()> {
        self.trace.borrow_mut().push(name);
        if self.fail == Some(name) {
            Err(unavailable(name))
        } else {
            Ok(())
        }
    }
}

impl DeviceQueries for Queries {
    fn init(&self) -> Result<()> {
        self.call("init")
    }
    fn driver_version(&self) -> Result<i32> {
        self.call("driver")?;
        Ok(self.driver)
    }
    fn nvrtc_version(&self) -> Result<(i32, i32)> {
        self.call("nvrtc")?;
        Ok(self.nvrtc)
    }
    fn device_count(&self) -> Result<i32> {
        self.call("count")?;
        Ok(self.count)
    }
    fn device(&self, ordinal: c_int) -> Result<sys::CUdevice> {
        assert_eq!(ordinal, 0);
        self.call("device")?;
        Ok(self.device)
    }
    fn uuid(&self, device: sys::CUdevice) -> Result<[c_char; 16]> {
        assert_eq!(device, self.device);
        self.call("uuid")?;
        Ok(self.uuid)
    }
    fn pci(&self, device: sys::CUdevice) -> Result<[c_char; 32]> {
        assert_eq!(device, self.device);
        self.call("pci")?;
        Ok(self.pci)
    }
    fn attribute(&self, device: sys::CUdevice, attribute: sys::CUdevice_attribute) -> Result<i32> {
        assert_eq!(device, self.device);
        match attribute {
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => {
                self.call("major")?;
                Ok(self.capability.0)
            }
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => {
                self.call("minor")?;
                Ok(self.capability.1)
            }
            _ => panic!("unexpected device query"),
        }
    }
}

#[test]
fn zero_ordinal_device_and_minor_are_valid_and_uuid_is_raw_bytes() {
    let q = Queries::valid();
    let facts = query_facts(&q, 0).unwrap();
    assert_eq!(facts.driver_version, 13010);
    assert_eq!(facts.nvrtc_version, (13, 0));
    assert_eq!(facts.device_count, 1);
    assert_eq!(facts.device, 0);
    assert_eq!(facts.physical.ordinal(), 0);
    assert_eq!(facts.physical.uuid_bytes(), &[128; 16]);
    assert_eq!(facts.physical.pci_bus_id(), "0000:01:00.0");
    assert_eq!(facts.physical.compute_capability(), (8, 0));
    assert_eq!(
        *q.trace.borrow(),
        [
            "init", "driver", "count", "device", "nvrtc", "uuid", "pci", "major", "minor"
        ]
    );
}

#[test]
fn invalid_ordinals_reject_before_discovery_or_device_lookup() {
    let overflow = usize::try_from(c_int::MAX).unwrap() + 1;
    assert!(CudaRuntimeObservation::observe(overflow).is_err());
    assert_eq!(checked_ordinal(c_int::MAX as usize).unwrap(), c_int::MAX);
    let q = Queries::valid();
    assert!(query_facts(&q, 1).is_err());
    assert!(!q.trace.borrow().contains(&"device"));
    assert!(!q.trace.borrow().contains(&"nvrtc"));
    assert_eq!(*q.trace.borrow(), ["init", "driver", "count"]);
    let q = Queries::valid();
    assert!(query_facts(&q, overflow).is_err());
    assert!(q.trace.borrow().is_empty());
}

#[test]
fn unavailable_numbers_cannot_become_positive_or_default_facts() {
    for bad in [-1, 0] {
        let mut q = Queries::valid();
        q.driver = bad;
        assert!(query_facts(&q, 0).is_err());
        let mut q = Queries::valid();
        q.count = bad;
        assert!(query_facts(&q, 0).is_err());
        let mut q = Queries::valid();
        q.nvrtc.0 = bad;
        assert!(query_facts(&q, 0).is_err());
        let mut q = Queries::valid();
        q.capability.0 = bad;
        assert!(query_facts(&q, 0).is_err());
    }
    let mut q = Queries::valid();
    q.nvrtc.1 = -1;
    assert!(query_facts(&q, 0).is_err());
    let mut q = Queries::valid();
    q.capability.1 = -1;
    assert!(query_facts(&q, 0).is_err());
    let mut q = Queries::valid();
    q.driver = i32::MAX;
    assert_eq!(query_facts(&q, 0).unwrap().driver_version, 2147483647);
}

#[test]
fn each_failed_query_stops_before_later_calls() {
    let sequence = [
        "init", "driver", "count", "device", "nvrtc", "uuid", "pci", "major", "minor",
    ];
    for (index, stage) in sequence.iter().enumerate() {
        let mut q = Queries::valid();
        q.fail = Some(stage);
        assert!(query_facts(&q, 0).is_err(), "{stage}");
        assert_eq!(*q.trace.borrow(), sequence[..=index]);
    }
}

#[test]
fn physical_identity_requires_uuid_and_bounded_ascii_pci() {
    let mut q = Queries::valid();
    q.uuid = [0; 16];
    assert!(query_facts(&q, 0).is_err());
    let mut q = Queries::valid();
    q.uuid = [0xffu8 as c_char; 16];
    assert_eq!(
        query_facts(&q, 0).unwrap().physical.uuid_bytes(),
        &[255; 16]
    );
    for pci in [[0; 32], [b'x' as c_char; 32]] {
        let mut q = Queries::valid();
        q.pci = pci;
        assert!(query_facts(&q, 0).is_err());
    }
    for bad in [0xffu8, b'\n', 0x7f] {
        let mut q = Queries::valid();
        q.pci[0] = bad as c_char;
        assert!(query_facts(&q, 0).is_err());
    }
}

#[test]
fn recheck_rejects_each_changed_fact_even_at_same_ordinal() {
    let original = query_facts(&Queries::valid(), 0).unwrap();
    assert!(require_same_facts(&original, &original).is_ok());
    let mut variants = Vec::new();
    let mut q = Queries::valid();
    q.driver += 1;
    variants.push(q);
    let mut q = Queries::valid();
    q.nvrtc.1 = 1;
    variants.push(q);
    let mut q = Queries::valid();
    q.count = 2;
    variants.push(q);
    let mut q = Queries::valid();
    q.device = 1;
    variants.push(q);
    let mut q = Queries::valid();
    q.uuid[0] = 1;
    variants.push(q);
    let mut q = Queries::valid();
    q.pci[6] = b'2' as c_char;
    variants.push(q);
    let mut q = Queries::valid();
    q.capability.1 = 1;
    variants.push(q);
    for q in variants {
        let changed = query_facts(&q, 0).unwrap();
        assert!(require_same_facts(&original, &changed).is_err());
    }
}

#[test]
fn real_device_capability_must_match_the_compilation_request() {
    let q = Queries::valid();
    assert!(query_created_device(&q, 0, 0, (8, 0)).is_ok());
    assert!(query_created_device(&q, 0, 0, (8, 1)).is_err());
}

#[test]
fn compile_options_preserve_fmad_and_exact_architecture_without_other_flags() {
    let options = kernel_compile_options(8, 0);
    assert_eq!(options.fmad, Some(false));
    assert_eq!(options.options, ["--gpu-architecture=compute_80"]);
    assert_eq!(options.ftz, None);
    assert_eq!(options.prec_sqrt, None);
    assert_eq!(options.prec_div, None);
    assert_eq!(options.use_fast_math, None);
    assert_eq!(options.maxrregcount, None);
    assert!(options.include_paths.is_empty());
    assert_eq!(options.arch, None);
    assert_eq!(options.name, None);
    assert_eq!(
        compile_option_projection(8, 0),
        ["--fmad=false", "--gpu-architecture=compute_80"]
    );
    assert_eq!(
        compile_option_projection(10, 1),
        ["--fmad=false", "--gpu-architecture=compute_101"]
    );
}

#[test]
fn guarded_study_errors_do_not_return_loader_repair_advice() {
    let panic_result: Result<()> = study_guarded("libraries", || panic!("set LD_LIBRARY_PATH"));
    let error_result: Result<()> = study_guarded("libraries", || {
        Err(HekitError::Device("set LD_LIBRARY_PATH".into()))
    });
    for result in [panic_result, error_result] {
        let error = result.unwrap_err().to_string();
        assert!(error.contains("libraries"));
        assert!(!error.contains("LD_LIBRARY_PATH"));
    }
}

#[test]
fn null_or_substituted_export_and_singleton_identities_reject() {
    let first = 1u8;
    let second = 2u8;
    let p = (&first as *const u8).cast::<c_void>();
    let other = (&second as *const u8).cast::<c_void>();
    let export = CudaExportedFunction {
        library: CudaLibraryKind::Driver,
        name: "cuInit",
        address: p,
    };
    assert!(validate_exports(std::slice::from_ref(&export)).is_ok());
    let mut null = export.clone();
    null.address = std::ptr::null();
    assert!(validate_exports(&[null]).is_err());
    let before = RuntimeIdentity {
        driver: p,
        nvrtc: other,
        exports: vec![export],
    };
    assert!(require_same_runtime(&before, &before).is_ok());
    let mut changed = before.clone();
    changed.driver = other;
    assert!(require_same_runtime(&before, &changed).is_err());
    let mut changed = before.clone();
    changed.nvrtc = p;
    assert!(require_same_runtime(&before, &changed).is_err());
    let mut changed = before.clone();
    changed.exports[0].address = other;
    assert!(require_same_runtime(&before, &changed).is_err());
    let mut changed = before.clone();
    changed.exports[0].library = CudaLibraryKind::Nvrtc;
    assert!(require_same_runtime(&before, &changed).is_err());
}
