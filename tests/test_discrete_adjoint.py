import math
import sys
import unittest
from pathlib import Path

import torch


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import verification_utils as vu

REPO_ROOT = vu.configure_local_import()
for module_name in tuple(sys.modules):
    if module_name == "DeepGPR" or module_name.startswith("DeepGPR."):
        del sys.modules[module_name]
import DeepGPR


def _coefficients(radius):
    if radius == 4:
        return (1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0)
    if radius == 2:
        return (9.0 / 8.0, -1.0 / 24.0)
    return (1.0,)


def _usable_radius(coord, size, requested, forward):
    radius = requested
    while radius > 1:
        invalid = (
            coord - radius + 1 < 0 or coord + radius >= size
            if forward
            else coord < radius or coord + radius - 1 >= size
        )
        if not invalid:
            break
        radius -= 1
    return radius


def _difference_and_transpose(values, dual, order, forward):
    size = values.numel()
    requested = {2: 1, 4: 2, 8: 4}[order]
    derivative = torch.zeros_like(values)
    transpose = torch.zeros_like(values)
    coordinates = range(0, size - 1) if forward else range(1, size)
    for coord in coordinates:
        radius = _usable_radius(coord, size, requested, forward)
        for offset, coefficient in enumerate(_coefficients(radius), start=1):
            if forward:
                positive = coord + offset
                negative = coord - offset + 1
            else:
                positive = coord + offset - 1
                negative = coord - offset
            derivative[coord] += coefficient * (values[positive] - values[negative])
            transpose[positive] += coefficient * dual[coord]
            transpose[negative] -= coefficient * dual[coord]
    return derivative, transpose


def _locations(shape):
    center = [max(2, size // 2) for size in shape]
    return torch.tensor([[center]], dtype=torch.int32)


def _state_problem(order, pml):
    torch.manual_seed(1100 + order + pml)
    shape = (9, 10, 8)
    nx, ny, nz = shape
    x = torch.linspace(0.0, 1.0, nx)[:, None, None]
    y = torch.linspace(0.0, 1.0, ny)[None, :, None]
    z = torch.linspace(0.0, 1.0, nz)[None, None, :]
    eps_r = 3.5 + 0.7 * x + 0.2 * y + 0.1 * z
    sigma = 1.0e-4 + 1.0e-4 * (x + y + z) / 3.0
    mu_r = 1.0 + 0.15 * x + 0.0 * y + 0.05 * z
    location = _locations(shape)
    source = torch.zeros((1, 1, 1), dtype=torch.float32)
    spacing = (0.020, 0.017, 0.013)
    dt = 1.0e-11

    initial_e, initial_h, initial_cpml = DeepGPR.checkpoint_initial_field(
        device="cpu",
        dx=spacing,
        dt=dt,
        source_amplitudes=source,
        source_location=location,
        receiver_location=location,
        er=eps_r,
        se=sigma,
        mr=mu_r,
        pmlthick=pml,
        fdtd_order=order,
    )
    base_state = [
        *(torch.randn_like(tensor) * 1.0e-3 for tensor in initial_e),
        *(torch.randn_like(tensor) * 1.0e-3 for tensor in initial_h),
        *(torch.randn_like(tensor) * 1.0e-3 for tensor in initial_cpml),
    ]
    state = [tensor.clone().requires_grad_(True) for tensor in base_state]
    result = DeepGPR.compute(
        device="cpu",
        dx=spacing,
        dt=dt,
        source_amplitudes=source,
        source_location=location,
        receiver_location=location,
        er=eps_r,
        se=sigma,
        mr=mu_r,
        E=tuple(state[:3]),
        H=tuple(state[3:6]),
        PML=tuple(state[6:]),
        pmlthick=pml,
        source_direction=2,
        reciever_direction=2,
        fdtd_order=order,
        mode=3,
        model_gradient_sampling_interval=1,
        wavefield_storage_dtype=torch.float32,
        debug=True,
    )
    outputs = [*result[1], *result[2], *result[3], result[-1]]
    output_duals = [torch.randn_like(tensor) for tensor in outputs]
    output_dot = sum((value * dual).sum() for value, dual in zip(outputs, output_duals))
    output_dot.backward()
    input_dot = sum(
        (base * variable.grad).sum() for base, variable in zip(base_state, state)
    )
    output_value = float(output_dot.detach())
    input_value = float(input_dot.detach())
    scale = max(abs(output_value), abs(input_value), 1.0e-20)
    return abs(output_value - input_value) / scale


def _material_case(pml):
    torch.manual_seed(2100 + pml)
    nx, ny, nt = 14, 17, 100
    x = torch.linspace(0.0, 1.0, nx)[:, None]
    y = torch.linspace(0.0, 1.0, ny)[None, :]
    eps_base = 4.0 + 0.4 * x + 0.2 * y
    sigma_base = 2.5e-4 + 0.5e-4 * x + 0.3e-4 * y
    mu_r = 1.0 + 0.1 * x + 0.05 * y
    source = DeepGPR.wavelet.ricker(3.5e8, nt, 2.5e-11, 2.0e-9).reshape(1, nt, 1)
    source_location = torch.tensor([[[6, 6, 0]]], dtype=torch.int32)
    receiver_location = torch.tensor([[[6, 10, 0], [7, 12, 0]]], dtype=torch.int32)
    arguments = dict(
        device="cpu",
        dx=(0.020, 0.016, 0.012),
        dt=2.5e-11,
        source_amplitudes=source,
        source_location=source_location,
        receiver_location=receiver_location,
        mr=mu_r,
        pmlthick=pml,
        fdtd_order=4,
        mode=2,
        model_gradient_sampling_interval=1,
        wavefield_storage_dtype=torch.float32,
    )

    with torch.no_grad():
        reference = DeepGPR.compute(er=eps_base, se=sigma_base, **arguments)[-1]
    data_scale = reference.abs().max().clamp_min(1.0e-12)
    data_target = 0.85 * reference

    def objective(eps_r, sigma):
        residual = (DeepGPR.compute(er=eps_r, se=sigma, **arguments)[-1] - data_target) / data_scale
        return 0.5 * residual.square().sum()

    eps_r = eps_base.clone().requires_grad_(True)
    sigma = sigma_base.clone().requires_grad_(True)
    loss = objective(eps_r, sigma)
    loss.backward()

    mask = torch.ones_like(eps_base)
    if pml:
        mask[: pml + 1] = 0.0
        mask[-pml:] = 0.0
        mask[:, : pml + 1] = 0.0
        mask[:, -pml:] = 0.0
    direction_eps = torch.randn_like(eps_base) * mask
    direction_eps /= direction_eps.abs().max().clamp_min(1.0e-12)
    direction_eps *= 0.15
    direction_sigma = torch.randn_like(sigma_base) * mask
    direction_sigma /= direction_sigma.abs().max().clamp_min(1.0e-12)
    direction_sigma *= 2.0e-4

    rows = {}
    directions = {
        "eps_r": (direction_eps, torch.zeros_like(direction_sigma)),
        "sigma": (torch.zeros_like(direction_eps), direction_sigma),
        "joint": (direction_eps, direction_sigma),
    }
    for name, (d_eps, d_sigma) in directions.items():
        directional_adjoint = float((eps_r.grad * d_eps).sum() + (sigma.grad * d_sigma).sum())
        case_rows = []
        for step in (1.0, 0.5, 0.25):
            with torch.no_grad():
                plus = float(objective(eps_base + step * d_eps, sigma_base + step * d_sigma))
                minus = float(objective(eps_base - step * d_eps, sigma_base - step * d_sigma))
            finite_difference = (plus - minus) / (2.0 * step)
            relative_error = abs(directional_adjoint - finite_difference) / max(
                abs(directional_adjoint), abs(finite_difference), 1.0e-20
            )
            first_order_remainder = abs(
                plus - float(loss.detach()) - step * directional_adjoint
            )
            case_rows.append((step, relative_error, first_order_remainder))
        rows[name] = case_rows
    return rows


class DiscreteAdjointTests(unittest.TestCase):
    def test_cpml_coefficients_are_separate_from_trainable_materials(self):
        eps_r = torch.full((10, 12, 8), 4.0, requires_grad=True)
        mu_r_pad = torch.ones((11, 13, 9), requires_grad=True)
        pml = torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32)
        coefficients = DeepGPR.build_pml_coeffs(
            eps_r, mu_r_pad, 2.0e-11, (0.02, 0.017, 0.013),
            10, 12, 8, pml, torch.device("cpu"), torch.float32,
        )[6:]
        self.assertTrue(all(not tensor.requires_grad for tensor in coefficients))

    def test_canonical_and_deprecated_compute_names_match(self):
        eps_r = torch.full((10, 12), 4.0)
        sigma = torch.zeros_like(eps_r)
        source = torch.linspace(-0.5, 0.5, 20).reshape(1, 20, 1)
        location = torch.tensor([[[5, 6, 0]]], dtype=torch.int32)
        common = dict(
            device="cpu", dx=0.02, dt=3.0e-11,
            source_amplitudes=source,
            source_location=location,
            receiver_location=location,
            pmlthick=0, fdtd_order=2, mode=2,
        )
        old = DeepGPR.compute(
            er=eps_r, se=sigma, reciever_direction=2, **common
        )[-1]
        new = DeepGPR.compute(
            eps_r=eps_r, sigma=sigma, receiver_component=2, **common
        )[-1]
        torch.testing.assert_close(old, new, rtol=0.0, atol=0.0)

    def test_weighted_curl_dot_products_orders_2_4_8(self):
        torch.manual_seed(7)
        values = torch.randn(17, dtype=torch.float64)
        dual = torch.randn(17, dtype=torch.float64)
        material_weight = torch.linspace(0.4, 1.7, 17, dtype=torch.float64)
        for order in (2, 4, 8):
            for forward in (False, True):
                weighted_dual = material_weight * dual
                derivative, transpose = _difference_and_transpose(
                    values, weighted_dual, order, forward
                )
                lhs = (material_weight * derivative * dual).sum()
                rhs = (values * transpose).sum()
                with self.subTest(order=order, forward=forward):
                    torch.testing.assert_close(lhs, rhs, rtol=1.0e-13, atol=1.0e-13)

    def test_native_field_update_dot_products_orders_2_4_8(self):
        for order in (2, 4, 8):
            with self.subTest(order=order):
                self.assertLess(_state_problem(order, pml=0), 2.0e-5)

    def test_native_cpml_single_step_dot_product_all_faces(self):
        self.assertLess(_state_problem(order=4, pml=2), 3.0e-5)

    def test_source_waveform_gradient_matches_finite_difference(self):
        nx, ny, nt = 12, 14, 60
        eps_r = torch.full((nx, ny), 4.0)
        sigma = torch.zeros_like(eps_r)
        source_location = torch.tensor([[[5, 5, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[5, 8, 0]]], dtype=torch.int32)
        base_source = DeepGPR.wavelet.ricker(4.0e8, nt, 3.0e-11, 2.0e-9).reshape(1, nt, 1)
        arguments = dict(
            device="cpu", dx=0.02, dt=3.0e-11,
            source_location=source_location, receiver_location=receiver_location,
            er=eps_r, se=sigma, pmlthick=0, fdtd_order=2, mode=2,
        )

        def objective(source):
            receiver = DeepGPR.compute(source_amplitudes=source, **arguments)[-1]
            return 0.5 * receiver.square().sum()

        source = base_source.clone().requires_grad_(True)
        objective(source).backward()
        direction = torch.randn_like(source)
        direction /= direction.norm()
        adjoint = float((source.grad * direction).sum())
        best_error = math.inf
        for step in (2.0e-3, 1.0e-3, 5.0e-4):
            with torch.no_grad():
                finite_difference = float(
                    (objective(base_source + step * direction) - objective(base_source - step * direction))
                    / (2.0 * step)
                )
            best_error = min(
                best_error,
                abs(adjoint - finite_difference) / max(abs(adjoint), abs(finite_difference), 1.0e-20),
            )
        self.assertLess(best_error, 8.0e-3)

    def test_material_taylor_checks_without_and_with_cpml(self):
        for pml in (0, 3):
            rows = _material_case(pml)
            for name, case_rows in rows.items():
                errors = [row[1] for row in case_rows]
                remainders = [row[2] for row in case_rows]
                with self.subTest(pml=pml, parameter=name):
                    self.assertLess(min(errors), 1.5e-2)
                    if name == "sigma":
                        self.assertLess(min(remainders), 2.0e-6)
                    else:
                        self.assertLess(remainders[-1], remainders[0] * 0.2)

    def test_incomplete_sampling_block_uses_actual_length(self):
        nx, ny, nt = 12, 14, 11
        eps_base = torch.full((nx, ny), 4.0)
        sigma = torch.full_like(eps_base, 2.0e-4)
        source = torch.linspace(-0.5, 0.8, nt).reshape(1, nt, 1)
        source_location = torch.tensor([[[5, 5, 0]]], dtype=torch.int32)
        receiver_location = torch.tensor([[[5, 7, 0]]], dtype=torch.int32)

        def gradient(interval):
            eps_r = eps_base.clone().requires_grad_(True)
            receiver = DeepGPR.compute(
                device="cpu", dx=0.02, dt=3.0e-11,
                source_amplitudes=source,
                source_location=source_location,
                receiver_location=receiver_location,
                er=eps_r, se=sigma, pmlthick=0,
                fdtd_order=2, mode=2,
                model_gradient_sampling_interval=interval,
                wavefield_storage_dtype=torch.float32,
            )[-1]
            receiver.square().sum().backward()
            return eps_r.grad

        torch.testing.assert_close(gradient(nt), gradient(nt + 5), rtol=0.0, atol=0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_cuda_forward_and_gradient_consistency(self):
        def run(device, use_async_offload=False):
            eps_r = torch.full((14, 17), 4.0, device=device, requires_grad=True)
            sigma = torch.full((14, 17), 2.0e-4, device=device, requires_grad=True)
            source = DeepGPR.wavelet.ricker(3.5e8, 100, 2.5e-11, 2.0e-9).reshape(1, 100, 1).to(device)
            source_location = torch.tensor([[[6, 6, 0]]], dtype=torch.int32, device=device)
            receiver_location = torch.tensor([[[6, 10, 0]]], dtype=torch.int32, device=device)
            receiver = DeepGPR.compute(
                device=device, dx=(0.020, 0.016, 0.012), dt=2.5e-11,
                source_amplitudes=source,
                source_location=source_location,
                receiver_location=receiver_location,
                er=eps_r, se=sigma, pmlthick=3,
                fdtd_order=4, mode=2,
                model_gradient_sampling_interval=1,
                wavefield_storage_dtype=torch.float32,
                use_async_offload=use_async_offload,
            )[-1]
            receiver.square().mean().backward()
            return receiver.detach().cpu(), eps_r.grad.cpu(), sigma.grad.cpu()

        cpu = run(torch.device("cpu"))
        cuda = run(torch.device("cuda"))
        cuda_offload = run(torch.device("cuda"), use_async_offload=True)
        for reference, candidate in (*zip(cpu, cuda), *zip(cuda, cuda_offload)):
            torch.testing.assert_close(reference, candidate, rtol=5.0e-4, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
