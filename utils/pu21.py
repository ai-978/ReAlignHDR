"""PU21 metrics for absolute HDR and the SCTNet benchmark protocol."""

import math

import numpy as np
import torch
import torch.nn.functional as F


PU21_BANDING_GLARE = np.array(
    [
        0.353487901,
        0.3734658629,
        8.277049286e-05,
        0.9062562627,
        0.09150303166,
        0.9099517204,
        596.3148142,
    ],
    dtype=np.float64,
)
PU21_L_MIN = 0.005
PU21_L_MAX = 10000.0
PU21_DYNAMIC_RANGE = 256.0
RGB_TO_LUMINANCE = np.array([0.212656, 0.715158, 0.072186], dtype=np.float64)


def _as_hwc_rgb(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3:
        raise ValueError("Expected a 3D RGB image.")
    if image.shape[-1] == 3:
        return image
    if image.shape[0] == 3:
        return image.transpose(1, 2, 0)
    raise ValueError("Expected RGB channels in the first or last dimension.")


def pu21_encode(luminance):
    """Encode absolute linear luminance using official PU21 banding_glare."""
    luminance = np.clip(np.asarray(luminance, dtype=np.float64), PU21_L_MIN, PU21_L_MAX)
    p = PU21_BANDING_GLARE
    encoded = p[6] * (
        ((p[0] + p[1] * luminance ** p[3]) / (1.0 + p[2] * luminance ** p[3]))
        ** p[4]
        - p[5]
    )
    return np.maximum(encoded, 0.0)


def map_relative_to_absolute(prediction, reference, peak_luminance=1000.0):
    """Map a relative HDR pair to nits using one reference-derived scale."""
    if peak_luminance <= 0 or peak_luminance > PU21_L_MAX:
        raise ValueError(f"peak_luminance must be in (0, {PU21_L_MAX}].")
    prediction = _as_hwc_rgb(prediction)
    reference = _as_hwc_rgb(reference)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must have the same shape.")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(reference)):
        raise ValueError("Prediction and reference must contain only finite values.")
    reference_peak = float(np.max(reference))
    if reference_peak <= 0:
        raise ValueError("Reference image must contain a positive value.")
    scale = peak_luminance / reference_peak
    return prediction * scale, reference * scale


def display_model_gog(
    encoded_image,
    peak_luminance=55.0,
    contrast=1000.0,
    gamma=2.2,
    ambient_light=10.0,
    reflectivity=0.005,
):
    """Apply the gain-gamma-offset display model used by SCTNet."""
    encoded_image = np.asarray(encoded_image, dtype=np.float64)
    black_luminance = ambient_light / math.pi * reflectivity + peak_luminance / contrast
    return (peak_luminance - black_luminance) * encoded_image ** gamma + black_luminance


def _psnr(test, reference):
    mse = float(np.mean((test - reference) ** 2, dtype=np.float64))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(PU21_DYNAMIC_RANGE / math.sqrt(mse))


def _gaussian_kernel(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float64) - size // 2
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return torch.outer(kernel, kernel).view(1, 1, size, size)


def _ssim(test, reference):
    test = np.asarray(test, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if test.shape != reference.shape or test.ndim != 2:
        raise ValueError("SSIM inputs must be same-sized 2D images.")

    window_size = min(11, min(test.shape))
    if window_size % 2 == 0:
        window_size -= 1
    sigma = 1.5 * window_size / 11.0
    window = _gaussian_kernel(window_size, sigma)
    test = torch.from_numpy(np.ascontiguousarray(test)).view(1, 1, *test.shape)
    reference = torch.from_numpy(np.ascontiguousarray(reference)).view(
        1, 1, *reference.shape
    )

    mu_test = F.conv2d(test, window)
    mu_reference = F.conv2d(reference, window)
    mu_test_sq = mu_test.square()
    mu_reference_sq = mu_reference.square()
    mu_product = mu_test * mu_reference
    sigma_test_sq = F.conv2d(test.square(), window) - mu_test_sq
    sigma_reference_sq = F.conv2d(reference.square(), window) - mu_reference_sq
    sigma_product = F.conv2d(test * reference, window) - mu_product

    c1 = (0.01 * PU21_DYNAMIC_RANGE) ** 2
    c2 = (0.03 * PU21_DYNAMIC_RANGE) ** 2
    ssim_map = ((2.0 * mu_product + c1) * (2.0 * sigma_product + c2)) / (
        (mu_test_sq + mu_reference_sq + c1)
        * (sigma_test_sq + sigma_reference_sq + c2)
    )
    return float(ssim_map.mean())


def pu21_metrics_absolute(prediction_absolute, reference_absolute):
    """Compute PU21-PSNR on RGB and PU21-SSIM on luminance."""
    prediction_absolute = _as_hwc_rgb(prediction_absolute)
    reference_absolute = _as_hwc_rgb(reference_absolute)
    if prediction_absolute.shape != reference_absolute.shape:
        raise ValueError("Prediction and reference must have the same shape.")

    pu_psnr = _psnr(pu21_encode(prediction_absolute), pu21_encode(reference_absolute))
    prediction_y = prediction_absolute @ RGB_TO_LUMINANCE
    reference_y = reference_absolute @ RGB_TO_LUMINANCE
    pu_ssim = _ssim(pu21_encode(prediction_y), pu21_encode(reference_y))
    return pu_psnr, pu_ssim


def pu21_metrics(prediction, reference, peak_luminance=1000.0):
    """Compute strict PU21 metrics after mapping relative HDR to absolute nits."""
    prediction_abs, reference_abs = map_relative_to_absolute(
        prediction, reference, peak_luminance
    )
    return pu21_metrics_absolute(prediction_abs, reference_abs)


def pu21_metrics_sctnet(
    prediction,
    reference,
    peak_luminance=55.0,
    contrast=1000.0,
    ambient_light=10.0,
    mu=5000.0,
):
    """Reproduce SCTNet's mu-law PNG and 55-nit PU21 benchmark protocol."""
    prediction = _as_hwc_rgb(prediction)
    reference = _as_hwc_rgb(reference)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must have the same shape.")

    denominator = math.log1p(mu)
    prediction_mu = np.log1p(mu * np.clip(prediction, 0.0, None)) / denominator
    reference_mu = np.log1p(mu * np.clip(reference, 0.0, None)) / denominator

    prediction_png = np.rint(np.clip(prediction_mu, 0.0, 1.0) * 255.0) / 255.0
    reference_png = np.rint(np.clip(reference_mu, 0.0, 1.0) * 255.0) / 255.0
    prediction_png = prediction_png[..., ::-1]
    reference_png = reference_png[..., ::-1]

    prediction_abs = display_model_gog(
        prediction_png, peak_luminance, contrast, ambient_light=ambient_light
    )
    reference_abs = display_model_gog(
        reference_png, peak_luminance, contrast, ambient_light=ambient_light
    )
    return pu21_metrics_absolute(prediction_abs, reference_abs)
