import torch
import random
import numpy as np


class ECGAugmentation:
    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate

    def __call__(self, x):
        if random.random() < 0.8:
            x = self.random_scale(x)
        if random.random() < 0.5:
            x = self.random_shift(x)
        if random.random() < 0.3:
            x = self.random_noise(x)
        if random.random() < 0.3:
            x = self.random_filter(x)
        return x

    def random_scale(self, x, scale_range=(0.8, 1.2)):
        scale = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return x * scale

    def random_shift(self, x, max_shift=0.1):
        max_shift_samples = int(max_shift * self.sample_rate)
        shift = random.randint(-max_shift_samples, max_shift_samples)
        return torch.roll(x, shifts=shift, dims=-1)

    def random_noise(self, x, noise_level=0.05):
        noise = torch.randn_like(x) * noise_level
        return x + noise

    def random_filter(self, x):
        # Simulate bandpass filter effects
        b, a = self._butterworth_coeffs()
        x_np = x.numpy()
        x_filt = torch.from_numpy(self._apply_filter(x_np, b, a))
        return x_filt.float()

    def _butterworth_coeffs(self):
        # Simplified filter coefficients
        return [0.1, 0.2, 0.1], [1.0, -0.5, 0.2]

    def _apply_filter(self, x, b, a):
        # Apply IIR filter
        return (x - a[1] * np.roll(x, 1) - a[2] * np.roll(x, 2)) / b[0]