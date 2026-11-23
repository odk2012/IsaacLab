"""Per-foot load targets in newtons. Column order: left, right. Frame: world Z."""
import torch


def measured_foot_fz(env):
    """Return [env, foot] forces and the sensor's per-environment validity."""
    sensor = env.scene.sensors["tactile_pad_gpu"]
    data = sensor.data
    if tuple(data.force_w.shape[1:]) != (2, 102, 3):
        raise ValueError(f"Unexpected pad force shape: {tuple(data.force_w.shape)}")
    return data.force_w[..., 2].sum(dim=-1), data.valid


def foot_fz_error(measured, target, sample_valid):
    """Return signed error and its mask. NaN target means unspecified; zero is valid."""
    if measured.ndim != 2 or measured.shape[1] != 2 or target.shape != measured.shape:
        raise ValueError("measured and target must both have shape [num_envs, 2]")
    if sample_valid.shape != measured.shape[:1] or sample_valid.dtype != torch.bool:
        raise ValueError("sample_valid must be a boolean tensor of shape [num_envs]")
    valid = sample_valid[:, None] & torch.isfinite(target) & torch.isfinite(measured)
    error = torch.where(valid, measured - target, float("nan"))
    return error, valid
