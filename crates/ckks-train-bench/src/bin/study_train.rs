//! Explicit prospective-study request frontend; execution remains D7-gated.
fn main() -> std::process::ExitCode {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if let Some(result) = ckks_train_bench::study_run::native_probe_entry(&args) {
        return match result {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{error}");
                std::process::ExitCode::FAILURE
            }
        };
    }

    match ckks_train_bench::study_run::run(args) {
        Ok(diagnostic) => eprintln!("{diagnostic}"),
        Err(error) => eprintln!("{error}"),
    }
    std::process::ExitCode::FAILURE
}
