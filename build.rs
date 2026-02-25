fn main() {
    println!("cargo:rerun-if-changed=kernel.c");
    println!("cargo:rerun-if-changed=ring_buffer.c");
    println!("cargo:rerun-if-changed=kernel.h");
    println!("cargo:rerun-if-changed=ring_buffer.h");

    cc::Build::new()
        .file("kernel.c")
        .file("ring_buffer.c")
        .flag_if_supported("-std=c23")
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx512f")
        .compile("polymarket_kernel_c");
}
