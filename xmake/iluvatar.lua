-- xmake/iluvatar.lua: Iluvatar (Iluvatar CoreX) implementations.
-- Only included from the top-level xmake.lua when --iluvatar-gpu=y is configured.
-- The Iluvatar CoreX SDK ships an nvcc-compatible compiler driver and full
-- CUDA-compatible headers/runtime (cuda_runtime.h, cuda_bf16.h, libcudart),
-- so we reuse xmake's built-in CUDA toolchain, pointed at the CoreX SDK:
--     xmake f --iluvatar-gpu=y --cuda=/usr/local/corex-4.4.0
-- The kernels/runtime API are the same CUDA sources as the NVIDIA port,
-- compiled by the CoreX nvcc with an iluvatar namespace.

target("llaisys-device-iluvatar")
    set_kind("static")
    add_deps("llaisys-utils")

    set_languages("cxx17")        -- CUDA language + C++17 host standard
    set_warnings("all", "error")
    add_cuflags("-Xcompiler", "-fPIC", {force = true}) -- host side must be PIC for the final .so
    add_culdflags("-Xcompiler", "-fPIC", {force = true}) -- devlink step (nvcc -dlink) must be PIC too
    add_cuflags("-fmad=false")            -- no FMA fusion: bitwise parity with the CPU ops
    -- REQUIRED: static targets are not device-linked by default; the final
    -- shared lib has no .cu files, so kernels must survive the static archive.
    add_values("cuda.build.devlink", true)

    add_files("../src/device/iluvatar/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-iluvatar")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_cuflags("-Xcompiler", "-fPIC", {force = true})
    add_culdflags("-Xcompiler", "-fPIC", {force = true})
    add_cuflags("-fmad=false")
    add_values("cuda.build.devlink", true)

    add_files("../src/ops/*/iluvatar/*.cu")

    on_install(function (target) end)
target_end()

-- Mirror the cpu pattern: aggregate targets gain the iluvatar variants.
-- (Same-name target re-declaration merges in xmake.)
target("llaisys-device")
    add_deps("llaisys-device-iluvatar")
target_end()

target("llaisys-ops")
    add_deps("llaisys-ops-iluvatar")
target_end()

-- Shared library: pull in the CUDA runtime libs.
-- add_rules("cuda") brings cuda.env: SDK include/link/rpath dirs + cudadevrt syslink,
-- and avoids cudart_static because we link cudart explicitly.
target("llaisys")
    add_rules("cuda")
    add_links("cudart")
target_end()
