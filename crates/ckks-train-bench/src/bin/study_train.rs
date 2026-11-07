//! Explicit prospective-study request frontend; execution remains D7-gated.
fn main() -> std::process::ExitCode {
    match ckks_train_bench::study_run::run(std::env::args_os().skip(1).collect()) {
        Ok(diagnostic) => eprintln!("{diagnostic}"),
        Err(error) => eprintln!("{error}"),
    }
    std::process::ExitCode::FAILURE
}
