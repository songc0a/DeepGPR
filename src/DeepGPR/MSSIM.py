import torch
import torch.nn as nn
import torch.nn.functional as F


class Misfit_M_SSIM(nn.Module):
    """Differentiable MS-SSIM-style misfit used by older DeepGPR notebooks."""

    def __init__(self, is_3d=False, window_size=7, levels=3, data_range=None):
        super().__init__()
        self.is_3d = bool(is_3d)
        self.window_size = int(window_size)
        self.levels = int(levels)
        self.data_range = data_range

    def _as_image_batch(self, x):
        if self.is_3d:
            if x.dim() == 3:
                return x.unsqueeze(0).unsqueeze(0)
            if x.dim() == 4:
                return x.unsqueeze(1)
            if x.dim() == 5:
                return x
            raise ValueError("3D MS-SSIM expects a 3D, 4D, or 5D tensor.")

        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() == 4:
            return x
        raise ValueError("2D MS-SSIM expects a 2D, 3D, or 4D tensor.")

    def _pool(self, x):
        if self.is_3d:
            return F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)
        return F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

    def _local_average(self, x):
        dims = x.dim() - 2
        kernel = min(self.window_size, *(int(v) for v in x.shape[-dims:]))
        if kernel < 1:
            return x
        if kernel % 2 == 0:
            kernel -= 1
        if kernel <= 1:
            return x

        padding = kernel // 2
        if self.is_3d:
            return F.avg_pool3d(x, kernel_size=kernel, stride=1, padding=padding)
        return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=padding)

    def _ssim(self, x, y):
        if self.data_range is None:
            dynamic_range = torch.max(torch.stack([x.detach().amax(), y.detach().amax()])) - torch.min(
                torch.stack([x.detach().amin(), y.detach().amin()])
            )
            dynamic_range = dynamic_range.clamp_min(1.0)
        else:
            dynamic_range = torch.as_tensor(self.data_range, dtype=x.dtype, device=x.device)

        c1 = (0.01 * dynamic_range) ** 2
        c2 = (0.03 * dynamic_range) ** 2

        mu_x = self._local_average(x)
        mu_y = self._local_average(y)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x = self._local_average(x * x) - mu_x2
        sigma_y = self._local_average(y * y) - mu_y2
        sigma_xy = self._local_average(x * y) - mu_xy

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
        return (numerator / denominator.clamp_min(1e-12)).mean()

    def forward(self, prediction, target):
        x = self._as_image_batch(prediction.float())
        y = self._as_image_batch(target.float())

        if x.shape != y.shape:
            raise ValueError(f"MS-SSIM inputs must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}.")

        scores = []
        levels = max(1, self.levels)
        for _ in range(levels):
            scores.append(self._ssim(x, y))
            if min(x.shape[-2:]) < 2 or (self.is_3d and min(x.shape[-3:]) < 2):
                break
            x = self._pool(x)
            y = self._pool(y)

        score = torch.stack(scores).mean()
        return 1.0 - score.clamp(-1.0, 1.0)
