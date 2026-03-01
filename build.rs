fn main() {
    println!("cargo:rerun-if-changed=c_src/kernel.c");
    println!("cargo:rerun-if-changed=c_src/ring_buffer.c");
    println!("cargo:rerun-if-changed=c_src/analytics.c");
    println!("cargo:rerun-if-changed=c_src/kernel.h");
    println!("cargo:rerun-if-changed=c_src/ring_buffer.h");
    println!("cargo:rerun-if-changed=c_src/analytics.h");

    cc::Build::new()
        .include("c_src")
        .file("c_src/kernel.c")
        .file("c_src/ring_buffer.c")
        .file("c_src/analytics.c")
        .flag_if_supported("-std=c23")
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx512f")
        .compile("polymarket_kernel_c");
}
