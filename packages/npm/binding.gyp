{
  "targets": [
    {
      "target_name": "bs_p_core",
      "sources": [
        "native/addon.c",
        "c_src/kernel.c"
      ],
      "include_dirs": ["c_src"],
      "cflags": ["-O3", "-std=c11"],
      "libraries": ["-lm"]
    }
  ]
}
