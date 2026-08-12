-- xmake/nvidia.lua: NVIDIA CUDA implementations.
-- Only included from the top-level xmake.lua when --nv-gpu=y is configured.
-- Configure the CUDA SDK once with:
--     xmake f --nv-gpu=y --cuda=/home/chengyang/cuda-12.4

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")

    set_languages("cxx17")        -- CUDA language + C++17 host standard
    set_warnings("all", "error")
    add_cuflags("-Xcompiler", "-fPIC", {force = true}) -- host side must be PIC for the final .so
    add_culdflags("-Xcompiler", "-fPIC", {force = true}) -- devlink step (nvcc -dlink) must be PIC too
    add_cuflags("-fmad=false")            -- no FMA fusion: bitwise parity with the CPU ops
    add_cugencodes("sm_80")               -- A800
    -- REQUIRED: static targets are not device-linked by default; the final
    -- shared lib has no .cu files, so kernels must survive the static archive.
    add_values("cuda.build.devlink", true)

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_cuflags("-Xcompiler", "-fPIC", {force = true})
    add_culdflags("-Xcompiler", "-fPIC", {force = true})
    add_cuflags("-fmad=false")
    add_cugencodes("sm_80")
    add_values("cuda.build.devlink", true)

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()

-- Mirror the cpu pattern: aggregate targets gain the nvidia variants.
-- (Same-name target re-declaration merges in xmake.)
target("llaisys-device")
    add_deps("llaisys-device-nvidia")
target_end()

target("llaisys-ops")
    add_deps("llaisys-ops-nvidia")
target_end()

-- Shared library: pull in the CUDA runtime libs.
-- add_rules("cuda") brings cuda.env: SDK include/link/rpath dirs + cudadevrt syslink,
-- and avoids cudart_static because we link cudart explicitly.
target("llaisys")
    add_rules("cuda")
    add_links("cudart", "cublas")
    -- The cublas references come from libllaisys-ops-nvidia.a, which the
    -- linker scans AFTER the -lcublas flags above; with --as-needed the
    -- DT_NEEDED entry would be dropped and ctypes.CDLL fails at load time
    -- with "undefined symbol: cublasCreate_v2". Force the dependency in.
    -- (shared targets take shflags, not ldflags)
    add_shflags("-Wl,--no-as-needed", "-lcublas", "-Wl,--as-needed", {force = true})
target_end()
