"""Test B: streaming randomized Tucker sketch (DeepGPR.tucker.sketch / wavefield)."""

import math
import sys
import unittest
from pathlib import Path

import torch

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import verification_utils as vu

vu.configure_local_import()
for module_name in tuple(sys.modules):
    if module_name == "DeepGPR" or module_name.startswith("DeepGPR."):
        del sys.modules[module_name]
import DeepGPR  # noqa: E402,F401
from DeepGPR.tucker import (  # noqa: E402
    SketchedWavefieldHistory,
    StreamingTuckerSketch,
    TuckerSketchConfig,
    hosvd,
    relative_error,
    tucker_reconstruct,
)
from DeepGPR.tucker._tensor_ops import khatri_rao_contract, unfold  # noqa: E402
from DeepGPR.tucker.sketch import resolve_spatial_ranks  # noqa: E402

CUDA = torch.cuda.is_available()
F64 = torch.float64


def _low_rank(shape, ranks, seed=0, dtype=F64, batch=None):
    generator = torch.Generator().manual_seed(seed)
    tensors = []
    for _ in range(batch or 1):
        core = torch.randn(ranks, generator=generator, dtype=dtype)
        factors = [torch.randn((s, r), generator=generator, dtype=dtype) for s, r in zip(shape, ranks)]
        tensors.append(tucker_reconstruct(core, factors))
    stacked = torch.stack(tensors)
    return stacked if batch else stacked[0]


def _smooth_field(shape, centers, dtype=F64):
    """Sum of non-separable Gaussian shells: decaying, full multilinear rank."""
    grids = torch.meshgrid(*(torch.linspace(0, 1, n, dtype=dtype) for n in shape), indexing="ij")
    field = torch.zeros(shape, dtype=dtype)
    for center, radius in centers:
        distance = torch.sqrt(sum((g - c) ** 2 for g, c in zip(grids, center)))
        field = field + torch.exp(-((distance - radius) / 0.06) ** 2) * torch.cos(40 * distance)
    return field


def _sketch(x, ranks, oversampling=4, seed=0, dtype=F64, device="cpu", **kwargs):
    operator = StreamingTuckerSketch(x.shape[1:], ranks, oversampling, seed=seed, dtype=dtype, device=device, **kwargs)
    return operator, operator.recover(operator.sketch(x.to(device)))


class KhatriRaoTests(unittest.TestCase):
    def test_matches_explicit_khatri_rao_product(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 4, 3, dtype=F64)
        mats = {0: torch.randn(5, 6, dtype=F64), 2: torch.randn(3, 6, dtype=F64)}
        result = khatri_rao_contract(x, mats, keep=1, batch=1)
        # Explicit (prod I_others x k) Khatri-Rao matrix, rows ordered (i0, i2).
        kr = (mats[0][:, None, :] * mats[2][None, :, :]).reshape(15, 6)
        expected = torch.stack([unfold(x[b], 1) @ kr for b in range(2)])
        self.assertLess(relative_error(result, expected), 1e-14)


class SketchRecoveryTests(unittest.TestCase):
    def test_exact_recovery_of_random_low_rank_tensors(self):
        for shape, ranks in [((17, 13, 11), (3, 4, 2)), ((20, 9), (5, 2)), ((8, 7, 6, 5), (2, 3, 1, 4))]:
            with self.subTest(shape=shape):
                x = _low_rank(shape, ranks, batch=3)
                _, block = _sketch(x, ranks, oversampling=2)
                self.assertLess(relative_error(block.reconstruct(), x), 1e-11)
                for mode, factor in block.factors.items():
                    self.assertEqual(tuple(factor.shape), (3, shape[mode], ranks[mode]))
                self.assertEqual(tuple(block.core.shape), (3, *ranks))

    def test_wrong_rank_per_mode_is_not_exact(self):
        # Ranks permuted across modes (an axis-order bug) cannot be exact.
        x = _low_rank((16, 16, 16), (2, 6, 4), batch=1)
        _, good = _sketch(x, (2, 6, 4), oversampling=0)
        _, wrong = _sketch(x, (6, 4, 2), oversampling=0)
        self.assertLess(relative_error(good.reconstruct(), x), 1e-10)
        self.assertGreater(relative_error(wrong.reconstruct(), x), 1e-2)

    def test_known_rank_with_various_ranks_and_oversampling(self):
        x = _low_rank((24, 20, 18), (4, 3, 5), batch=2, seed=3)
        for ranks in [(4, 3, 5), (6, 6, 6), (10, 8, 12)]:
            for oversampling in (0, 3, 10):
                with self.subTest(ranks=ranks, oversampling=oversampling):
                    _, block = _sketch(x, ranks, oversampling)
                    self.assertLess(relative_error(block.reconstruct(), x), 1e-10)

    def test_error_decreases_with_rank_and_tracks_hosvd(self):
        x = _smooth_field((40, 36, 32), [((0.4, 0.5, 0.5), 0.25), ((0.7, 0.3, 0.6), 0.15)])[None]
        errors = []
        for rank in (4, 8, 16, 24):
            _, block = _sketch(x, (rank,) * 3, oversampling=8, seed=1)
            error = relative_error(block.reconstruct(), x)
            best = relative_error(tucker_reconstruct(*hosvd(x[0], rank)), x[0])
            errors.append(error)
            # Quasi-optimal: never better than HOSVD-truncation of the same rank
            # by more than rounding, and within a modest factor of it.
            self.assertGreaterEqual(error, 0.5 * best)
            self.assertLess(error, 4.0 * best + 1e-12)
        self.assertTrue(all(a > b for a, b in zip(errors, errors[1:])), errors)

    def test_oversampling_improves_accuracy_on_average(self):
        x = _smooth_field((36, 32, 30), [((0.5, 0.5, 0.5), 0.3)])[None]
        def mean_error(p):
            return sum(relative_error(_sketch(x, (8, 8, 8), p, seed=s)[1].reconstruct(), x) for s in range(4)) / 4
        self.assertLess(mean_error(10), mean_error(0))

    def test_high_rank_data(self):
        torch.manual_seed(0)
        x = torch.randn(1, 24, 24, 24, dtype=F64)
        _, block = _sketch(x, (8, 8, 8), oversampling=8)
        error = relative_error(block.reconstruct(), x)
        best = relative_error(tucker_reconstruct(*hosvd(x[0], 8)), x[0])
        self.assertTrue(math.isfinite(error))
        self.assertGreaterEqual(error, best - 1e-12)
        # Flat spectra: the one-pass error exceeds the optimum noticeably.
        self.assertLess(error, 2.0)

    def test_zero_tensor_reconstructs_zero(self):
        x = torch.zeros(2, 10, 9, 8)
        _, block = _sketch(x, (3, 3, 3), dtype=torch.float32)
        result = block.reconstruct()
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual(float(result.abs().max()), 0.0)

    def test_kept_modes_and_raw(self):
        x = _low_rank((12, 3, 10), (2, 3, 2), batch=2)
        operator, block = _sketch(x, (2, 5, 2))
        self.assertEqual(operator.kept, (False, True, False))
        self.assertNotIn(1, block.factors)
        self.assertLess(relative_error(block.reconstruct(), x), 1e-11)
        raw = StreamingTuckerSketch((4, 3), (4, 3), dtype=F64)
        self.assertTrue(raw.is_raw)
        y = torch.randn(1, 4, 3, dtype=F64)
        self.assertLess(relative_error(raw.recover(raw.sketch(y)).reconstruct(), y), 1e-15)

    def test_seed_reproducibility(self):
        x = _smooth_field((20, 18, 16), [((0.5, 0.4, 0.5), 0.2)])[None]
        a = _sketch(x, (5, 5, 5), seed=11)[1].reconstruct()
        b = _sketch(x, (5, 5, 5), seed=11)[1].reconstruct()
        c = _sketch(x, (5, 5, 5), seed=12)[1].reconstruct()
        self.assertEqual(float((a - b).abs().max()), 0.0)
        self.assertGreater(float((a - c).abs().max()), 0.0)

    @unittest.skipUnless(CUDA, "CUDA is not available")
    def test_cuda_matches_cpu(self):
        x = _smooth_field((26, 22, 20), [((0.5, 0.5, 0.4), 0.2)])[None].repeat(2, 1, 1, 1)
        for dtype, tolerance in ((F64, 1e-10), (torch.float32, 1e-4)):
            cpu = _sketch(x.to(dtype), (6, 6, 6), seed=3, dtype=dtype)[1].reconstruct()
            gpu_operator, gpu_block = _sketch(x.to(dtype), (6, 6, 6), seed=3, dtype=dtype, device="cuda")
            gpu = gpu_block.reconstruct()
            self.assertEqual(gpu.device.type, "cuda")
            self.assertLess(relative_error(gpu.cpu(), cpu), tolerance)
            self.assertLess(relative_error(gpu.cpu().double(), x), 0.5)
        # Exactness on CUDA too.
        y = _low_rank((18, 16, 14), (3, 2, 4), batch=2)
        _, block = _sketch(y, (3, 2, 4), oversampling=3, device="cuda")
        self.assertLess(relative_error(block.reconstruct().cpu(), y), 1e-10)

    def test_recovery_device_option_and_batched_recovery(self):
        from DeepGPR.tucker.sketch import recover_many

        devices = ["cpu"] + (["cuda"] if CUDA else [])
        x = _smooth_field((22, 20, 18), [((0.5, 0.5, 0.4), 0.2)])[None].repeat(3, 1, 1, 1)
        y = _low_rank((9, 14, 12, 10), (3, 2, 4, 3), batch=2)
        for device in devices:
            results = {}
            for where in ("host", "device"):
                a = StreamingTuckerSketch(x.shape[1:], (5, 6, 4), 4, seed=1, dtype=F64, device=device, recovery_device=where)
                b = StreamingTuckerSketch(y.shape[1:], (3, 2, 4, 3), 3, seed=2, dtype=F64, device=device, stream_mode=0, recovery_device=where)
                acc_b = b.new_accumulator(2)
                b.update(acc_b, y[:, :4].to(device), 0)
                b.update(acc_b, y[:, 4:].to(device), 4)
                blocks = recover_many([(a, a.sketch(x.to(device)), None), (b, acc_b, None)])
                single = a.recover(a.sketch(x.to(device)))
                self.assertLess(relative_error(blocks[0].reconstruct().cpu(), single.reconstruct().cpu()), 1e-12)
                self.assertLess(relative_error(blocks[1].reconstruct().cpu(), y), 1e-10)
                results[where] = blocks[0].reconstruct().cpu()
            with self.subTest(device=device):
                self.assertLess(relative_error(results["device"], results["host"]), 1e-9)

    @unittest.skipUnless(CUDA, "CUDA is not available")
    def test_tf32_is_disabled_inside_the_sketch(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            x = _smooth_field((30, 28, 26), [((0.5, 0.5, 0.5), 0.25)])[None].float()
            cpu = _sketch(x, (12, 12, 12), seed=2, dtype=torch.float32)[1].reconstruct()
            gpu = _sketch(x, (12, 12, 12), seed=2, dtype=torch.float32, device="cuda")[1].reconstruct().cpu()
            self.assertLess(relative_error(gpu, cpu), 1e-5)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)  # restored
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


class StreamingTests(unittest.TestCase):
    def test_chunked_stream_equals_one_shot(self):
        x = _low_rank((12, 9, 8, 7), (3, 2, 3, 2), batch=2)  # mode 0 = time
        operator = StreamingTuckerSketch(x.shape[1:], (3, 2, 3, 2), 3, seed=5, dtype=F64, stream_mode=0)
        whole = operator.sketch(x)
        streamed = operator.new_accumulator(2)
        for start, stop in [(0, 1), (1, 5), (5, 6), (6, 12)]:
            operator.update(streamed, x[:, start:stop], offset=start)
        for mode in operator.compressed_modes:
            self.assertLess(relative_error(streamed.factor_sketches[mode], whole.factor_sketches[mode]), 1e-13)
        self.assertLess(relative_error(streamed.core_sketch, whole.core_sketch), 1e-13)
        block = operator.recover(streamed)
        self.assertLess(relative_error(block.reconstruct(), x), 1e-11)
        self.assertLess(relative_error(block.reconstruct(rows=(3, 8)), x[:, 3:8]), 1e-11)

    def test_partial_block_and_order_checks(self):
        x = _low_rank((10, 9, 8), (3, 2, 3), batch=1)
        operator = StreamingTuckerSketch((16, 9, 8), (3, 2, 3), 3, dtype=F64, stream_mode=0)
        accumulator = operator.new_accumulator(1)
        operator.update(accumulator, x[:, :4], 0)
        with self.assertRaisesRegex(ValueError, "in order"):
            operator.update(accumulator, x[:, 6:8], 6)
        operator.update(accumulator, x[:, 4:10], 4)
        with self.assertRaisesRegex(ValueError, "were pushed"):
            operator.recover(accumulator, length=12)
        block = operator.recover(accumulator, length=10)
        self.assertLess(relative_error(block.reconstruct(), x), 1e-11)


class WavefieldHistoryTests(unittest.TestCase):
    def _space_time_low_rank(self, frames, shots, shape, ranks, time_rank, seed=0):
        """W[t, s] with spatial multilinear rank <= ranks and time rank <= time_rank."""
        generator = torch.Generator().manual_seed(seed)
        spatial = []
        for _ in range(shots):
            basis = [_low_rank(shape, ranks, seed=seed * 100 + 7 * s + j) for j, s in enumerate(range(time_rank))]
            coeff = torch.randn(frames, time_rank, generator=generator, dtype=F64)
            spatial.append(torch.einsum("tr,r...->t...", coeff, torch.stack(basis)))
        return torch.stack(spatial, dim=1)  # (F, S, *shape)

    def _history(self, frames, shape, config, shots=2):
        return SketchedWavefieldHistory(("Ez", "Rz"), shape, shots, frames, config, torch.device("cpu"), F64)

    def test_per_frame_history_exact_for_low_rank_frames(self):
        frames, shape = 23, (14, 12, 10)
        data = {name: _low_rank(shape, (3, 2, 4), seed=i, batch=frames * 2).reshape(frames, 2, *shape)
                for i, name in enumerate(("Ez", "Rz"))}
        config = TuckerSketchConfig(ranks=(3, 2, 4), oversampling=2, compute_dtype=F64)
        history = self._history(frames, shape, config)
        for start, stop in [(0, 7), (7, 8), (8, 23)]:
            history.push({k: v[start:stop] for k, v in data.items()}, start)
        for name in data:
            self.assertLess(relative_error(history.reconstruct(name), data[name]), 1e-10)
            self.assertLess(relative_error(history.reconstruct(name, 5, 9), data[name][5:14]), 1e-10)

    def test_time_blocks_streamed_across_segments(self):
        frames, shape = 29, (13, 11, 9)
        data = self._space_time_low_rank(frames, 2, shape, (2, 2, 2), time_rank=2)
        # Union of 2 time-basis tensors of rank 2 -> spatial rank <= 4.
        config = TuckerSketchConfig(ranks=4, oversampling=2, time_block=8, time_rank=2, compute_dtype=F64)
        history = SketchedWavefieldHistory(("Ez",), shape, 2, frames, config, torch.device("cpu"), F64)
        for start, stop in [(0, 3), (3, 11), (11, 12), (12, 29)]:  # blocks span pushes
            history.push({"Ez": data[start:stop]}, start)
        self.assertIsNone(history.stores["Ez"].open_block)
        self.assertLess(relative_error(history.reconstruct("Ez"), data), 1e-10)
        for start, count in [(0, 1), (6, 5), (24, 5), (7, 17)]:
            self.assertLess(relative_error(history.reconstruct("Ez", start, count), data[start : start + count]), 1e-10)

    def test_storage_bytes_and_no_raw_retention(self):
        frames, shape, shots = 10, (20, 18, 16), 3
        config = TuckerSketchConfig(ranks=(4, 5, 3), oversampling=3, compute_dtype=torch.float32)
        history = SketchedWavefieldHistory(("Ez", "Rz"), shape, shots, frames, config, torch.device("cpu"))
        buffer = torch.randn(frames, shots, *shape)
        history.push({"Ez": buffer, "Rz": buffer}, 0)
        report = history.size_report()
        per_frame = 4 * 5 * 3 + 20 * 4 + 18 * 5 + 16 * 3
        mask_bytes = 20 * 18 * 16 // 8  # packed support bits per component
        self.assertEqual(report.stored_bytes, 2 * (frames * shots * per_frame * 4 + mask_bytes))
        self.assertEqual(report.raw_bytes, 2 * frames * shots * 20 * 18 * 16 * 4)
        self.assertEqual(report.open_sketch_bytes, 0)
        # Every tensor held by the store is far smaller than one raw frame set.
        store = history.stores["Ez"]
        held = [store.core, *store.factors.values()]
        self.assertIsNone(store.raw)
        frame_bytes = shots * 20 * 18 * 16 * 4
        self.assertTrue(all(t.numel() * t.element_size() < frame_bytes for t in held))
        # Overwriting the caller's buffer does not change the stored history.
        before = history.reconstruct("Ez").clone()
        buffer.zero_()
        self.assertEqual(float((history.reconstruct("Ez") - before).abs().max()), 0.0)

    def test_support_mask_restores_exact_zeros(self):
        frames, shape = 5, (14, 12, 10)
        data = _smooth_field(shape, [((0.5, 0.5, 0.5), 0.2)]).expand(frames, 2, *shape).clone()
        data[..., 0, :, :] = 0.0  # a face that is never updated
        data[..., :, -1, :] = 0.0
        config = TuckerSketchConfig(ranks=3, oversampling=2, compute_dtype=F64)
        history = SketchedWavefieldHistory(("Ez",), shape, 2, frames, config, torch.device("cpu"), F64)
        history.push({"Ez": data}, 0)
        result = history.reconstruct("Ez")
        self.assertEqual(float(result[..., 0, :, :].abs().max()), 0.0)
        self.assertEqual(float(result[..., :, -1, :].abs().max()), 0.0)
        self.assertGreater(float(result[..., 5, 5, :].abs().max()), 0.0)
        packed = history.stores["Ez"].support_bits
        self.assertEqual(packed.numel(), -(-14 * 12 * 10 // 8))

    def test_raw_fallback_and_component_selection(self):
        config = TuckerSketchConfig(ranks={"E": (3, 3), "R": 64}, compress="all", compute_dtype=F64)
        history = SketchedWavefieldHistory(("Ez", "Rz"), (12, 10, 1), 1, 4, config, torch.device("cpu"), F64)
        self.assertTrue(history.stores["Ez"].compressed)
        self.assertFalse(history.stores["Rz"].compressed)
        self.assertEqual(history.stores["Rz"].reason, "ranks >= frame size")
        data = torch.randn(4, 1, 12, 10, 1, dtype=F64)
        history.push({"Ez": data, "Rz": data}, 0)
        self.assertEqual(float((history.reconstruct("Rz") - data).abs().max()), 0.0)
        only_e = TuckerSketchConfig(ranks=2, compress="E")
        h2 = SketchedWavefieldHistory(("Ex", "Ey", "Ez", "Rx", "Ry", "Rz"), (8, 8, 8), 1, 2, only_e, torch.device("cpu"))
        self.assertEqual([n for n, s in h2.stores.items() if s.compressed], ["Ex", "Ey", "Ez"])
        large = TuckerSketchConfig(ranks=(9, 9, 2), min_compression_ratio=4.0)
        h3 = SketchedWavefieldHistory(("Ez",), (10, 10, 3), 1, 2, large, torch.device("cpu"))
        self.assertFalse(h3.stores["Ez"].compressed)

    def test_2d_frames_and_shapes_per_component(self):
        # 2D frames (singleton z) are matrices; different shapes per history.
        for shape in [(30, 26, 1), (1, 20, 18), (25, 1, 1)]:
            with self.subTest(shape=shape):
                active = tuple(s for s in shape if s > 1)
                ranks = resolve_spatial_ranks(3, shape)
                self.assertEqual(sum(r > 1 for r in ranks), len(active))
                config = TuckerSketchConfig(ranks=3, oversampling=2, compute_dtype=F64)
                history = SketchedWavefieldHistory(("Ez",), shape, 2, 5, config, torch.device("cpu"), F64)
                if len(active) >= 2:
                    data = _low_rank(active, (3,) * len(active), batch=10).reshape(5, 2, *shape)
                    history.push({"Ez": data}, 0)
                    self.assertTrue(history.stores["Ez"].compressed)
                    self.assertLess(relative_error(history.reconstruct("Ez"), data), 1e-10)
                else:
                    self.assertFalse(history.stores["Ez"].compressed)

    def test_error_measurement_and_zero_frames(self):
        frames, shape = 6, (16, 14, 12)
        data = torch.zeros(frames, 1, *shape, dtype=F64)
        data[3:] = _smooth_field(shape, [((0.5, 0.5, 0.5), 0.2)])
        config = TuckerSketchConfig(ranks=4, oversampling=4, compute_dtype=F64, measure_error=True)
        history = SketchedWavefieldHistory(("Ez",), shape, 1, frames, config, torch.device("cpu"), F64)
        history.push({"Ez": data}, 0)
        stats = history.error_report()["Ez"]
        exact = relative_error(history.reconstruct("Ez"), data)
        self.assertAlmostEqual(stats.relative_l2, exact, places=10)
        self.assertEqual(stats.frames, frames)
        self.assertTrue(math.isfinite(stats.max_frame_relative))
        self.assertEqual(float(history.reconstruct("Ez")[:3].abs().max()), 0.0)

    def test_push_order_and_completeness(self):
        config = TuckerSketchConfig(ranks=2)
        history = SketchedWavefieldHistory(("Ez",), (6, 6, 6), 1, 4, config, torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "Expected frame 0"):
            history.push({"Ez": torch.zeros(1, 1, 6, 6, 6)}, 1)
        history.push({"Ez": torch.zeros(2, 1, 6, 6, 6)}, 0)
        with self.assertRaisesRegex(RuntimeError, "Only 2 of 4"):
            history.reconstruct_into({"Ez": torch.zeros(1, 1, 6, 6, 6)}, 0)


class ConfigTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            TuckerSketchConfig(oversampling=-1)
        with self.assertRaises(ValueError):
            TuckerSketchConfig(time_block=4, time_rank=5)
        with self.assertRaises(ValueError):
            TuckerSketchConfig(compute_dtype=torch.float16)
        with self.assertRaises(ValueError):
            TuckerSketchConfig(compress="H")
        with self.assertRaises(ValueError):
            TuckerSketchConfig(ranks={"Hx": 3})
        with self.assertRaises(TypeError):
            TuckerSketchConfig(segment_steps=2.5)
        config = TuckerSketchConfig(ranks={"E": 4, "Rz": (1, 2, 3), "default": 6})
        self.assertEqual(config.ranks_for("Ex"), 4)
        self.assertEqual(config.ranks_for("Rz"), (1, 2, 3))
        self.assertEqual(config.ranks_for("Rx"), 6)
        self.assertEqual(TuckerSketchConfig(time_block=10).resolved_time_rank, 4)
        with self.assertRaisesRegex(ValueError, "one per axis"):
            resolve_spatial_ranks((2, 3), (4, 5, 6))


if __name__ == "__main__":
    unittest.main()
