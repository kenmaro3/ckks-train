// Compile-fail ownership witness for the private helper's retained source borrow.
// No gate, native open, runtime or external implementation is involved.
use std::fs::File;
fn private_link_source(_: &File, _: &File) {}
fn witness(file: File, parent: &File) {
    let source = &file;
    drop(file);
    private_link_source(source, parent);
}
