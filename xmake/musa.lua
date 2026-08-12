-- xmake/musa.lua: Moore Threads MUSA implementations.
-- Only included from the top-level xmake.lua when --musa-gpu=y is configured.
--
-- MUSA is source-compatible with CUDA: kernels use __global__/threadIdx and
-- the runtime API is musaMalloc/musaMemcpy/... compiled with mcc (nvcc-like).

-- xmake has no built-in mcc toolchain, so we drive it directly with a custom
-- rule. mcc must be on PATH (MUSA SDK install); override with MCC=<path>.
rule("mcc")
    set_extensions(".cu")
    on_build_file(function (target, sourcefile, objectfile)
        os.mkdir(os.dirname(objectfile))
        local mcc = os.getenv("MCC") or "mcc"
        os.execv(mcc, {"-c", "-fPIC", "-O3", "-std=c++17",
                       "-I", os.projectdir() .. "/include",
                       "-I", os.projectdir() .. "/src",
                       "-o", objectfile, sourcefile})
    end)
rule_end()

target("llaisys-device-musa")
    set_kind("static")
    add_deps("llaisys-utils")
    add_rules("mcc")
    set_languages("cxx17")

    add_files("../src/device/musa/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-musa")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-musa")
    add_rules("mcc")
    set_languages("cxx17")

    add_files("../src/ops/*/musa/*.cu")

    on_install(function (target) end)
target_end()

-- Mirror the cpu pattern: aggregate targets gain the musa variants.
-- (Same-name target re-declaration merges in xmake.)
target("llaisys-device")
    add_deps("llaisys-device-musa")
target_end()

target("llaisys-ops")
    add_deps("llaisys-ops-musa")
target_end()

-- Shared library: pull in the MUSA runtime libs.
target("llaisys")
    add_links("musa", "mublas")
    -- Same --as-needed trick as nvidia.lua: the mublas references come from
    -- libllaisys-ops-musa.a, which the linker scans AFTER the -lmublas flags
    -- above; force the DT_NEEDED entry so ctypes.CDLL finds the symbol.
    -- (If mublas is not installed on the machine, drop "-lmublas" here —
    -- linear_musa.cu falls back to its naive kernel automatically.)
    add_shflags("-Wl,--no-as-needed", "-lmublas", "-Wl,--as-needed", {force = true})
target_end()
