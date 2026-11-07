//! Synthetic frozen ratified transport fixtures. No context, native observation or readiness.
use super::runtime_descriptors::*;
use crate::study_profile::{CaseBackend, CaseMode, CasePlacement};
const P_CPU: &str = "210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76311f0000000000000068656b69742d73747564792d706879736963616c2d62696e64696e672d7631000100000000000000430001000000000000004f0001000000000000004800";
const R_CPU: &str = "210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76311d0000000000000068656b69742d73747564792d72756e74696d652d706f6c6963792d763100250000000000000073747564792d747261696e2d6370752d73657269616c2d6c696e75782d636c65616e2d763100010000000000000000000040000000000000003563356531663635626462383965353135316538343232656636393964363361343232663764346264376662336164323838646362353934326530643231346600";
const P_CUDA: &str = "210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76311f0000000000000068656b69742d73747564792d706879736963616c2d62696e64696e672d7631000100000000000000430001000000000000004f000100000000000000480100000000000000001000000000000000000102030405060708090a0b0c0d0e0f0c00000000000000303030303a30313a30302e3008000000000000000000000000000000";
const R_CUDA: &str = "210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76311d0000000000000068656b69742d73747564792d72756e74696d652d706f6c6963792d763102240000000000000073747564792d747261696e2d637564612d686f73742d6c696e75782d636c65616e2d76310102000000000000000001010000000000000003400000000000000062326130333232653761613862663738626336386336313863333630373064616139353565323830306164656232653538356530653632666362323734363933010018000000000000002f6f70742f637564612f6c69622fff6472697665722e736f400000000000000061616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161d2320000000000000019000000000000002f6f70742f637564612f6c69622f6c69626e767274632e736f4000000000000000626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262620d00000000000000000000000000000004000000000000000040000000000000006363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636301400000000000000064646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464024000000000000000656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565650340000000000000006666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666602000000000000000c000000000000002d2d666d61643d66616c73651d000000000000002d2d6770752d6172636869746563747572653d636f6d707574655f3830";

fn literal(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
fn selected(cuda: bool, ni: bool) -> PolicySelection {
    PolicySelection {
        backend: if cuda {
            CaseBackend::Cuda
        } else {
            CaseBackend::Cpu
        },
        mode: if ni {
            CaseMode::Noninteractive
        } else {
            CaseMode::Interactive
        },
        placement: if ni {
            CasePlacement::WhenNeeded
        } else {
            CasePlacement::ClientRefresh
        },
        bootstrap_engine: u8::from(ni),
    }
}
fn physical(cuda: bool) -> PhysicalDescriptor {
    PhysicalDescriptor::parse(&literal(if cuda { P_CUDA } else { P_CPU })).unwrap()
}
fn policy(cuda: bool) -> RuntimePolicy {
    RuntimePolicy::parse(
        &literal(if cuda { R_CUDA } else { R_CPU }),
        &physical(cuda),
        selected(cuda, cuda),
    )
    .unwrap()
}
#[test]
fn synthetic_descriptor_four_independent_literals() {
    for (cuda, plen, phash, rlen, rhash) in [
        (
            false,
            111,
            "5c5e1f65bdb89e5151e8422ef699d63a422f7d4bd7fb3ad288dcb5942e0d214f",
            209,
            "d9da1fc171c8f60e848c916d2926a7f7579903aac8157a9d273aa7eb01e80fb5",
        ),
        (
            true,
            179,
            "b2a0322e7aa8bf78bc68c618c36070daa955e2800adeb2e585e0e62fcb274693",
            816,
            "def6285a4a1dd5276e7a28fbe8a851460a693f80f2fa2aa34a70172e2570a7d9",
        ),
    ] {
        let p = physical(cuda);
        let r = policy(cuda);
        assert_eq!(p.canonical_bytes().len(), plen);
        assert_eq!(p.sha256(), phash);
        assert_eq!(
            p.body.encode().unwrap(),
            literal(if cuda { P_CUDA } else { P_CPU })
        );
        let encoded = r.encode(&p, selected(cuda, cuda)).unwrap();
        assert_eq!(
            encoded.canonical_bytes(),
            literal(if cuda { R_CUDA } else { R_CPU })
        );
        assert_eq!(encoded.canonical_bytes().len(), rlen);
        assert_eq!(encoded.sha256(), rhash);
    }
}
#[test]
fn synthetic_descriptor_rejects_transport_corruption() {
    for (hex, cuda) in [
        (P_CPU, false),
        (P_CUDA, true),
        (R_CPU, false),
        (R_CUDA, true),
    ] {
        let original = literal(hex);
        let accepts = |b: &[u8]| {
            if hex == P_CPU || hex == P_CUDA {
                PhysicalDescriptor::parse(b).is_ok()
            } else {
                RuntimePolicy::parse(b, &physical(cuda), selected(cuda, cuda)).is_ok()
            }
        };
        for n in 0..original.len() {
            assert!(!accepts(&original[..n]), "prefix {n}");
        }
        let mut b = original.clone();
        b.push(0);
        assert!(!accepts(&b));
        let mut b = original.clone();
        b[8] = b'X';
        assert!(!accepts(&b));
        let mut b = original.clone();
        b[49] = b'X'; // First byte of the independently frozen domain Text.
        assert!(!accepts(&b));
        let mut b = original.clone();
        b[..8].fill(255);
        assert!(!accepts(&b));
        assert!(!accepts(&vec![0; super::A_MAX + 1]));
    }
    let mut b = literal(P_CPU);
    b[89] = 255;
    assert!(PhysicalDescriptor::parse(&b).is_err());
    let mut b = literal(P_CPU);
    b[80] = 2;
    assert!(PhysicalDescriptor::parse(&b).is_err());
    let mut b = literal(P_CPU);
    b[99] = 255;
    assert!(PhysicalDescriptor::parse(&b).is_err());
    let mut b = literal(R_CPU);
    b[78] = 3;
    assert!(RuntimePolicy::parse(&b, &physical(false), selected(false, false)).is_err());
}
#[test]
fn synthetic_physical_identity_bounds_and_swaps() {
    let p = physical(true);
    let mut b = p.body.clone();
    let d = b.cuda_device.as_mut().unwrap();
    assert_eq!(d.ordinal, 0);
    assert_eq!(d.compute_minor, 0);
    for invalid in [0, u64::MAX] {
        let mut b = p.body.clone();
        b.cuda_device.as_mut().unwrap().compute_major = invalid;
        assert!(b.encode().is_err());
    }
    for bad in ["", "a\nb", "é", "12345678901234567890123456789012"] {
        let mut b = p.body.clone();
        b.cuda_device.as_mut().unwrap().pci_bus_id = super::Text(bad.into());
        assert!(b.encode().is_err());
    }
    let mut b = p.body.clone();
    b.cuda_device.as_mut().unwrap().uuid.fill(0);
    assert!(b.encode().is_err());
    let mut b = p.body.clone();
    b.cuda_device.as_mut().unwrap().ordinal = u64::MAX;
    assert!(b.encode().is_err());
    let mut b = p.body.clone();
    b.cpu_model = super::Observation::Available(super::Text(String::new()));
    assert!(b.encode().is_err());
    let mut b = p.body.clone();
    b.os = super::Observation::Unavailable(super::Text("missing".into()));
    assert!(b.encode().is_err());
    let mut b = p.body.clone();
    b.hostname = super::Observation::Available(super::Text("X".into()));
    let swapped = PhysicalDescriptor::from_body(b).unwrap();
    assert!(policy(true).encode(&swapped, selected(true, true)).is_err());
    assert!(policy(false).encode(&p, selected(false, false)).is_err());
}
#[test]
fn synthetic_runtime_alignment_and_route_claims_rejected() {
    let p = physical(false);
    let r = policy(false);
    for (form, backend, workers, streams, route) in [
        (3, 0, 1, None, 0),
        (0, 1, 1, None, 0),
        (0, 0, 0, None, 0),
        (0, 0, 2, None, 0),
        (0, 0, 1, Some(1), 0),
        (0, 0, 1, None, 1),
        (0, 0, 1, None, 2),
        (0, 0, 1, None, 4),
    ] {
        let mut r = r.clone();
        r.form = form;
        r.backend = backend;
        r.execution_threads = workers;
        r.stream_count = streams;
        r.refresh_route = route;
        assert!(r.encode(&p, selected(false, false)).is_err());
    }
    let mut r = r.clone();
    r.form = 1;
    r.recipe = super::Text("study-train-cpu-rayon-linux-clean-v1".into());
    assert!(r.encode(&p, selected(false, false)).is_ok());
    r.recipe = super::Text("wrong".into());
    assert!(r.encode(&p, selected(false, false)).is_err());
    let mut selection = selected(false, false);
    selection.placement = CasePlacement::WhenNeeded;
    assert!(policy(false).encode(&p, selection).is_err());
    let mut selection = selected(true, true);
    selection.bootstrap_engine = 2;
    assert!(policy(true).encode(&physical(true), selection).is_err());
    let mut r = policy(true);
    r.stream_count = Some(2);
    assert!(r.encode(&physical(true), selected(true, true)).is_err());
}
#[test]
fn synthetic_cuda_paths_versions_kernels_options_and_hashes() {
    let p = physical(true);
    let r = policy(true);
    let c = r.cuda_runtime.as_ref().unwrap();
    let mut raw_tag = literal(R_CUDA);
    raw_tag[216] = 1;
    assert!(RuntimePolicy::parse(&raw_tag, &p, selected(true, true)).is_err());
    assert!(c.driver_path.bytes.contains(&255));
    assert_eq!(c.nvrtc_minor, 0);
    for field in 0..3 {
        let mut r = r.clone();
        let c = r.cuda_runtime.as_mut().unwrap();
        match field {
            0 => c.driver_version = 0,
            1 => c.nvrtc_major = 0,
            _ => c.nvrtc_minor = u64::MAX,
        };
        assert!(r.encode(&p, selected(true, true)).is_err());
    }
    for bad in [
        b"relative".to_vec(),
        b"/a/../b".to_vec(),
        b"/a//b".to_vec(),
        b"/a\0b".to_vec(),
        vec![b'/'; 4097],
    ] {
        let mut r = r.clone();
        r.cuda_runtime.as_mut().unwrap().driver_path.bytes = bad;
        assert!(r.encode(&p, selected(true, true)).is_err());
    }
    let mut r2 = r.clone();
    let c = r2.cuda_runtime.as_mut().unwrap();
    c.nvrtc_path = c.driver_path.clone();
    assert!(r2.encode(&p, selected(true, true)).is_err());
    for field in 0..5 {
        let mut r = r.clone();
        let c = r.cuda_runtime.as_mut().unwrap();
        match field {
            0 => c.kernels.0.swap(0, 1),
            1 => {
                c.kernels.0.pop();
            }
            2 => c.intended_options.0.swap(0, 1),
            3 => c.intended_options.0[1] = super::Text("--gpu-architecture=compute_080".into()),
            _ => c.kernels.0[0].source_sha256 = super::Hash("A".repeat(64)),
        };
        assert!(r.encode(&p, selected(true, true)).is_err());
    }
    let mut r2 = r.clone();
    r2.physical_binding_sha256 = super::Hash("f".repeat(64));
    assert!(r2.encode(&p, selected(true, true)).is_err());
    // Recomputed inner hashes cannot make a substituted source inventory match the original view.
    let mut substituted = r.clone();
    substituted.cuda_runtime.as_mut().unwrap().kernels.0[0].source_sha256 =
        super::Hash("a".repeat(64));
    assert!(
        substituted
            .cuda_runtime
            .as_ref()
            .unwrap()
            .matches_observed(r.cuda_runtime.as_ref().unwrap())
            .is_err()
    );
}

#[test]
fn synthetic_runtime_other_supported_routes_and_missing_cuda_rejected() {
    let p = physical(false);
    let mut r = policy(false);
    r.refresh_route = 1;
    assert!(r.encode(&p, selected(false, true)).is_ok());
    r.form = 1;
    r.recipe = super::Text("study-train-cpu-rayon-linux-clean-v1".into());
    r.execution_threads = 2;
    assert!(r.encode(&p, selected(false, true)).is_ok());
    let p = physical(true);
    let mut r = policy(true);
    r.refresh_route = 2;
    assert!(r.encode(&p, selected(true, false)).is_ok());
    r.cuda_runtime = None;
    assert!(r.encode(&p, selected(true, false)).is_err());
    let mut r = policy(false);
    r.cuda_runtime = policy(true).cuda_runtime;
    assert!(r.encode(&physical(false), selected(false, false)).is_err());
}

// Actual host observation only; no approved context or CUDA observer is forged.
// Deferred Linux inventory: reads proc CPU/task data and calls uname/gethostname.
#[cfg(target_os = "linux")]
#[test]
fn linux_physical_host_projection_excludes_live_thread_telemetry() {
    use crate::study_provenance::{Host, Observation};
    let before = Host::observe();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (stop_tx, stop_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        ready_tx.send(()).unwrap();
        let _ = stop_rx.recv();
    });
    ready_rx.recv().unwrap();
    let during = Host::observe();
    stop_tx.send(()).unwrap();
    worker.join().unwrap();
    let observed = |h: &Host| match h.observed_threads() {
        Observation::Available(n) => *n,
        _ => panic!("Linux thread observation unavailable"),
    };
    // The extra thread is known alive, but unrelated process threads may change.
    // No equality/inequality of that global telemetry is an identity condition.
    let _telemetry = (observed(&before), observed(&during));
    assert_eq!(
        PhysicalBody::host(&before, None).unwrap().encode().unwrap(),
        PhysicalBody::host(&during, None).unwrap().encode().unwrap()
    );
}
