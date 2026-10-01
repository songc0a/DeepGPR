"""Consistency of the Python ABI table with ``lib/deepgpr.h`` and the call bridge.

These tests need neither PyTorch nor a GPU. When PyTorch is missing, the
``DeepGPR`` package ``__init__`` (which imports the solver) is bypassed and
only the torch-free ``config``/``utils``/``native`` sub-packages are loaded.
"""

from __future__ import annotations

import ctypes
import importlib
import re
import sys
import threading
import time
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "DeepGPR"
HEADER = PACKAGE_DIR / "lib" / "deepgpr.h"


def _import_native_modules():
    """Import DeepGPR.native.* without requiring PyTorch."""
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import torch  # noqa: F401
    except ImportError:
        if "DeepGPR" not in sys.modules:
            stub = types.ModuleType("DeepGPR")
            stub.__path__ = [str(PACKAGE_DIR)]
            sys.modules["DeepGPR"] = stub
    abi = importlib.import_module("DeepGPR.native.abi")
    bridge = importlib.import_module("DeepGPR.native.bridge")
    loader = importlib.import_module("DeepGPR.native.loader")
    constants = importlib.import_module("DeepGPR.config.constants")
    errors = importlib.import_module("DeepGPR.utils.exceptions")
    return abi, bridge, loader, constants, errors


abi, bridge, loader, constants, errors = _import_native_modules()

_C_KIND = {
    "float*": abi.FLOAT_PTR,
    "int*": abi.INT_PTR,
    "void*": abi.VOID_PTR,
    "float": abi.FLOAT,
    "int": abi.INT,
    "long long": abi.LONGLONG,
    "char*": abi.CHAR_PTR,
}


def parse_header(text: str) -> dict:
    """Return ``{function: (return_type, [(name, kind), ...])}`` from the header."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    functions = {}
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    pattern = r"DEEPGPR_API\s+((?:const\s+)?\w+(?:\s*\*\s*|\s+))(\w+)\s*\((.*?)\)\s*;"
    for match in re.finditer(pattern, text, flags=re.S):
        return_type, name, params = match.groups()
        return_type = " ".join(return_type.replace("const ", "").split()).replace(" *", "*").strip()
        parsed = []
        params = " ".join(params.split())
        if params and params != "void":
            for raw in params.split(","):
                raw = raw.strip().replace("const ", "")
                pieces = re.match(r"(.+?)\s*(\*?)\s*(\w+)$", raw)
                assert pieces, raw
                base, star, pname = pieces.groups()
                kind = " ".join(base.split()) + star
                parsed.append((pname, _C_KIND[kind]))
        functions[name] = (return_type, parsed)
    return functions


class HeaderConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = parse_header(HEADER.read_text(encoding="utf-8"))

    def test_every_header_function_is_described(self):
        self.assertEqual(set(self.header), set(abi.NATIVE_FUNCTIONS))

    def test_parameter_names_kinds_and_order_match(self):
        for name, (signature, return_kind) in abi.NATIVE_FUNCTIONS.items():
            with self.subTest(function=name):
                header_return, header_params = self.header[name]
                self.assertEqual(list(signature), header_params)
                expected_return = None if header_return == "void" else _C_KIND[header_return]
                self.assertEqual(return_kind, expected_return)

    def test_header_abi_version_matches_python(self):
        version = re.search(r"#define\s+DEEPGPR_ABI_VERSION\s+(\d+)", HEADER.read_text())
        self.assertEqual(int(version.group(1)), constants.NATIVE_ABI_VERSION)

    def test_physical_constants_match_python(self):
        text = HEADER.read_text()

        def define(name):
            match = re.search(rf"#define\s+{name}\s+([0-9.eE+-]+)f?\b", text)
            self.assertIsNotNone(match, name)
            return float(match.group(1))

        self.assertEqual(define("DEEPGPR_EPSILON0"), constants.VACUUM_PERMITTIVITY)
        self.assertEqual(define("DEEPGPR_MU0"), constants.VACUUM_PERMEABILITY)
        self.assertEqual(define("DEEPGPR_EPSILON0_F"), constants.VACUUM_PERMITTIVITY)
        self.assertEqual(define("DEEPGPR_MU0_F"), constants.VACUUM_PERMEABILITY)
        self.assertEqual(
            define("DEEPGPR_PEC_SIGMA_THRESHOLD"), constants.PEC_CONDUCTIVITY_THRESHOLD
        )
        self.assertEqual(
            define("DEEPGPR_PEC_SIGMA_THRESHOLD_F"), constants.PEC_CONDUCTIVITY_THRESHOLD
        )

    def test_backends_take_constants_from_the_header(self):
        for source in ("deepgpr_cpu.c", "deepgpr.cu"):
            code = re.sub(r"/\*.*?\*/", "", (HEADER.parent / source).read_text(), flags=re.S)
            with self.subTest(source=source):
                self.assertNotIn("8.8541878128e-12", code)
                self.assertNotIn("1.25663706212e-06", code)
                self.assertNotRegex(code, r"sigma_pad\[[^]]+\]\s*[<>]=?\s*100\.0f")

    def test_cpu_backend_has_no_atomics(self):
        code = (HEADER.parent / "deepgpr_cpu.c").read_text()
        self.assertNotIn("omp atomic", code)

    def test_parameter_counts(self):
        self.assertEqual(len(abi.FORWARD_SIGNATURE), 81)
        self.assertEqual(len(abi.BACKWARD_SIGNATURE), 86)
        self.assertEqual(len(abi.PHI_PARAMETER_NAMES), constants.CPML_PHI_TENSOR_COUNT)


class MarshallingTests(unittest.TestCase):
    def test_missing_and_unexpected_arguments_are_rejected(self):
        signature = (("a", abi.INT), ("b", abi.FLOAT))
        with self.assertRaisesRegex(KeyError, "missing=\\['b'\\]"):
            bridge.marshal_arguments(signature, {"a": 1})
        with self.assertRaisesRegex(KeyError, "unexpected=\\['c'\\]"):
            bridge.marshal_arguments(signature, {"a": 1, "b": 2.0, "c": 3})

    def test_conversion_follows_signature_order(self):
        buffer = (ctypes.c_float * 4)()
        values = bridge.marshal_arguments(
            (("p", abi.FLOAT_PTR), ("n", abi.INT), ("x", abi.FLOAT), ("v", abi.VOID_PTR)),
            {"x": 2.5, "v": ctypes.addressof(buffer), "n": 7.0, "p": ctypes.addressof(buffer)},
        )
        self.assertEqual(ctypes.cast(values[0], ctypes.c_void_p).value, ctypes.addressof(buffer))
        self.assertEqual(values[1:3], [7, 2.5])
        self.assertEqual(values[3].value, ctypes.addressof(buffer))


class OrderGateTests(unittest.TestCase):
    def test_different_orders_never_overlap_and_same_orders_run_concurrently(self):
        class FakeLibrary:
            def __init__(self):
                self.order = None
                self.active = {}
                self.max_same_order = 0
                self.lock = threading.Lock()
                self.violations = 0

            def set_fdtd_order(self, order):
                self.order = order

        lib = FakeLibrary()
        gate = bridge.FdtdOrderGate()

        def worker(order):
            with gate.hold(lib, order):
                with lib.lock:
                    lib.active[order] = lib.active.get(order, 0) + 1
                    if any(count for key, count in lib.active.items() if key != order):
                        lib.violations += 1
                    lib.max_same_order = max(lib.max_same_order, lib.active[order])
                    if lib.order != order:
                        lib.violations += 1
                time.sleep(0.01)
                with lib.lock:
                    lib.active[order] -= 1

        threads = [
            threading.Thread(target=worker, args=(order,)) for order in (2, 2, 4, 2, 8, 4, 4, 2) * 3
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(lib.violations, 0)
        self.assertGreater(lib.max_same_order, 1)


class LibraryLoadingTests(unittest.TestCase):
    def test_cpu_library_configuration(self):
        try:
            lib = loader.get_deepgpr_lib("cpu")
        except FileNotFoundError as exc:  # pragma: no cover - platform without a build
            self.skipTest(str(exc))
        self.assertEqual(int(lib.deepgpr_abi_version()), constants.NATIVE_ABI_VERSION)
        self.assertEqual(len(lib.forward.argtypes), len(abi.FORWARD_SIGNATURE))
        self.assertEqual(len(lib.backward.argtypes), len(abi.BACKWARD_SIGNATURE))
        self.assertTrue(loader.library_supports(lib, "deepgpr_supports_external_pml"))
        self.assertFalse(loader.library_supports(object(), "deepgpr_supports_external_pml"))
        self.assertIs(loader.get_deepgpr_lib("cpu"), lib)

    def _cpu_lib(self):
        try:
            return loader.get_deepgpr_lib("cpu")
        except FileNotFoundError as exc:  # pragma: no cover
            self.skipTest(str(exc))

    def test_cpu_backend_reports_errors_through_python(self):
        lib = self._cpu_lib()
        if not bridge.supports_error_reporting(lib):
            self.skipTest("CPU library predates native error reporting; rebuild it.")
        self.assertEqual(bridge.last_native_error(lib), "")
        # A field this large cannot be allocated on any 64-bit system: the
        # native backward must stop before touching any buffer and report it.
        arguments = {name: 0 for name, _ in abi.BACKWARD_SIGNATURE}
        arguments.update(
            dt=1.0e-12,
            nt=1,
            nshot=1,
            nreceiver=1,
            dx=0.01,
            dy=0.01,
            dz=0.01,
            nx_fields=1 << 15,
            ny_fields=1 << 15,
            nz_fields=1 << 15,
            ndata_source=1,
            nsource=1,
            sampling_interval=1,
            fwi_mode=2,
        )
        with self.assertRaisesRegex(errors.NativeLibraryError, "could not allocate|overflows"):
            bridge.invoke(lib, "backward", 2, arguments)
        self.assertEqual(bridge.last_native_error(lib), "")

    def test_cpu_backend_declares_deterministic_adjoint(self):
        lib = self._cpu_lib()
        if not hasattr(lib, "deepgpr_deterministic_adjoint"):
            self.skipTest("CPU library predates the deterministic adjoint; rebuild it.")
        self.assertTrue(loader.library_supports(lib, "deepgpr_deterministic_adjoint"))

    def test_unknown_device_and_kind(self):
        with self.assertRaises(ValueError):
            loader.get_deepgpr_lib("mps")
        with self.assertRaises(ValueError):
            loader.candidate_library_paths("tpu")


if __name__ == "__main__":
    unittest.main()
