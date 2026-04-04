from setuptools import Extension, setup

ext_modules = [
    Extension(
        "bs_p._core",
        sources=["src/bs_p/_core_module.c", "c_src/kernel.c"],
        include_dirs=["c_src"],
        extra_compile_args=["-O3", "-std=c11"],
        extra_link_args=["-lm"],
    )
]

setup(ext_modules=ext_modules)
