"""
Language-action computation utilities.

Ports LAP's language-action pipeline to pure numpy (no TensorFlow dependency).
Computes per-step motion deltas from EEF states and converts them to natural language.
"""

import re

import numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# Rotation Utilities
# =============================================================================


def axis_angle_to_euler(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to extrinsic XYZ Euler angles.

    Args:
        axis_angle: (..., 3) axis-angle vectors (rotation vector format).

    Returns:
        (..., 3) Euler angles [roll, pitch, yaw] in extrinsic XYZ convention.
    """
    orig_shape = axis_angle.shape
    flat = axis_angle.reshape(-1, 3)
    euler = Rotation.from_rotvec(flat).as_euler("xyz")
    return euler.reshape(orig_shape).astype(np.float64)


def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """Convert extrinsic XYZ Euler angles to rotation matrices.

    Args:
        euler: (..., 3) Euler angles [roll, pitch, yaw].

    Returns:
        (..., 3, 3) rotation matrices.
    """
    orig_shape = euler.shape[:-1]
    flat = euler.reshape(-1, 3)
    mats = Rotation.from_euler("xyz", flat).as_matrix()
    return mats.reshape(*orig_shape, 3, 3).astype(np.float64)


def rotation_matrix_to_euler(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to extrinsic XYZ Euler angles.

    Args:
        rot_matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) Euler angles [roll, pitch, yaw].
    """
    orig_shape = rot_matrix.shape[:-2]
    flat = rot_matrix.reshape(-1, 3, 3)
    euler = Rotation.from_matrix(flat).as_euler("xyz")
    return euler.reshape(*orig_shape, 3).astype(np.float64)


def euler_diff(euler1: np.ndarray, euler2: np.ndarray) -> np.ndarray:
    """Compute relative Euler angle difference: R_rel = R2^T @ R1.

    Computes euler_rel such that R(euler2) @ R(euler_rel) = R(euler1).

    Args:
        euler1: (..., 3) target Euler angles.
        euler2: (..., 3) source Euler angles.

    Returns:
        (..., 3) relative Euler angles.
    """
    R1 = euler_to_rotation_matrix(euler1)
    R2 = euler_to_rotation_matrix(euler2)
    # R_rel = R2^T @ R1
    R_rel = np.einsum("...ji,...jk->...ik", R2, R1)
    return rotation_matrix_to_euler(R_rel)


# =============================================================================
# Movement Action Computation
# =============================================================================


def compute_padded_movement_actions(eef_state: np.ndarray) -> np.ndarray:
    """Compute movement actions as deltas between consecutive EEF states.

    Follows DROID convention: action[t] = state[t+1] - state[t],
    with action[T-1] = 0 (zero-padded).

    Args:
        eef_state: (T, 6) array containing [xyz(3), rpy(3)].

    Returns:
        (T, 6) padded movement actions [delta_xyz(3), delta_rpy(3)].
    """
    T = eef_state.shape[0]
    # Position deltas
    pos_deltas = eef_state[1:, :3] - eef_state[:-1, :3]
    # Rotation deltas via euler_diff
    rot_deltas = euler_diff(eef_state[1:, 3:6], eef_state[:-1, 3:6])
    # Concatenate
    movement = np.concatenate([pos_deltas, rot_deltas], axis=-1)
    # Zero-pad last timestep
    padded = np.concatenate([movement, np.zeros((1, 6))], axis=0)
    return padded


def compute_language_actions(
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Compute per-step 7D language-action vectors for a LIBERO trajectory.

    Args:
        states: (T, 8) raw observation states [eef_pos(3), axis_angle(3), gripper_qpos(2)].
        actions: (T, 7) raw actions [delta_eef(6), gripper(1)].

    Returns:
        (T, 7) language-action vectors [delta_xyz(3), delta_rpy(3), gripper(1)].
    """
    # Convert axis-angle to Euler
    euler_rpy = axis_angle_to_euler(states[:, 3:6])
    eef_state = np.concatenate([states[:, :3], euler_rpy], axis=-1)  # (T, 6)

    # Compute per-step movement deltas
    padded_movement = compute_padded_movement_actions(eef_state)  # (T, 6)

    # Process gripper: clip to [0,1], invert (1=open, 0=close)
    gripper_action = np.clip(actions[:, -1:], 0, 1)
    gripper_action = 1.0 - gripper_action

    # Concatenate: [delta_xyz(3), delta_rpy(3), gripper(1)]
    language_action = np.concatenate([padded_movement, gripper_action], axis=-1)
    return language_action


def aggregate_language_actions(
    language_actions: np.ndarray,
    window_size: int = 8,
) -> np.ndarray:
    """Sum language-action deltas over a temporal window.

    For position and rotation: sum deltas over the window.
    For gripper: take the last step's value in the window.

    Args:
        language_actions: (T, 7) per-step language-action vectors.
        window_size: Number of future steps to aggregate.

    Returns:
        (T, 7) aggregated language-action vectors.
    """
    T = language_actions.shape[0]
    aggregated = np.zeros_like(language_actions)

    for t in range(T):
        end = min(t + window_size, T)
        window = language_actions[t:end]
        # Sum position and rotation deltas
        aggregated[t, :6] = window[:, :6].sum(axis=0)
        # Gripper: use last step in window
        aggregated[t, 6] = window[-1, 6]

    return aggregated


# =============================================================================
# Text Summarization (ported from LAP action_text.py)
# =============================================================================


def _round_to_nearest_n(value: float, n: int = 5) -> int:
    return int(round(value / n) * n)


def _format_numeric(val: float, sum_decimal: str) -> str:
    decimals = 0
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            return ""
        if sum_decimal == "nearest_10":
            return str(int(round(val / 10) * 10))
        m = re.fullmatch(r"(\d+)f", sum_decimal)
        if m:
            decimals = int(m.group(1))
    return f"{val:.{decimals}f}"


def summarize_numeric_actions(
    arr_like,
    sum_decimal: str,
    include_rotation: bool = False,
    rotation_precision: int = 10,
) -> str | None:
    """Convert a 7D language-action vector to human-readable text.

    Args:
        arr_like: (7,) or (N, 7) array of language-action values.
            [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
            Position deltas in meters, rotation deltas in radians, gripper 0-1.
        sum_decimal: Format string. "1f" = 1 decimal, "0f" = integer, "no_number" = no magnitudes.
        include_rotation: Whether to include rotation descriptions.
        rotation_precision: Round rotation degrees to nearest N.

    Returns:
        Text description, e.g. "move forward 3.0 cm, move up 1.0 cm, close gripper".
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 7:
        return None

    if sum_decimal in {"no_number", "nearest_10"}:
        decimals = 0
    else:
        decimals = int(re.fullmatch(r"(\d+)f", sum_decimal).group(1))

    dx_m = float(arr[..., 0].sum())
    dy_m = float(arr[..., 1].sum())
    dz_m = float(arr[..., 2].sum())
    dx = round(abs(dx_m * 100.0), decimals)
    dy = round(abs(dy_m * 100.0), decimals)
    dz = round(abs(dz_m * 100.0), decimals)

    if include_rotation:
        droll_rad = float(arr[..., 3].sum())
        dpitch_rad = float(arr[..., 4].sum())
        dyaw_rad = float(arr[..., 5].sum())
        droll = _round_to_nearest_n(abs(droll_rad * 180.0 / np.pi), rotation_precision)
        dpitch = _round_to_nearest_n(abs(dpitch_rad * 180.0 / np.pi), rotation_precision)
        dyaw = _round_to_nearest_n(abs(dyaw_rad * 180.0 / np.pi), rotation_precision)

    parts: list[str] = []
    if sum_decimal == "no_number":
        if dx_m > 0 and dx != 0:
            parts.append("move forward")
        elif dx_m < 0 and dx != 0:
            parts.append("move back")
        if dy_m > 0 and dy != 0:
            parts.append("move left")
        if dy_m < 0 and dy != 0:
            parts.append("move right")
        if dz_m > 0 and dz != 0:
            parts.append("move up")
        elif dz_m < 0 and dz != 0:
            parts.append("move down")
        if include_rotation:
            if droll_rad > 0:
                parts.append("tilt left")
            elif droll_rad < 0:
                parts.append("tilt right")
            if dpitch_rad > 0:
                parts.append("tilt back")
            elif dpitch_rad < 0:
                parts.append("tilt forward")
            if dyaw_rad > 0:
                parts.append("rotate counterclockwise")
            elif dyaw_rad < 0:
                parts.append("rotate clockwise")
    else:
        fmt_dx = _format_numeric(dx, sum_decimal)
        fmt_dy = _format_numeric(dy, sum_decimal)
        fmt_dz = _format_numeric(dz, sum_decimal)
        if dx_m > 0 and dx != 0:
            parts.append(f"move forward {fmt_dx} cm")
        elif dx_m < 0 and dx != 0:
            parts.append(f"move back {fmt_dx} cm")
        if dz_m > 0 and dz != 0:
            parts.append(f"move up {fmt_dz} cm")
        elif dz_m < 0 and dz != 0:
            parts.append(f"move down {fmt_dz} cm")
        if dy_m > 0 and dy != 0:
            parts.append(f"move left {fmt_dy} cm")
        elif dy_m < 0 and dy != 0:
            parts.append(f"move right {fmt_dy} cm")
        if include_rotation:
            if droll_rad > 0 and droll != 0:
                parts.append(f"tilt left {droll} degrees")
            elif droll_rad < 0 and droll != 0:
                parts.append(f"tilt right {droll} degrees")
            if dpitch_rad > 0 and dpitch != 0:
                parts.append(f"tilt back {dpitch} degrees")
            elif dpitch_rad < 0 and dpitch != 0:
                parts.append(f"tilt forward {dpitch} degrees")
            if dyaw_rad > 0 and dyaw != 0:
                parts.append(f"rotate counterclockwise {dyaw} degrees")
            elif dyaw_rad < 0 and dyaw != 0:
                parts.append(f"rotate clockwise {dyaw} degrees")

    g_last = float(arr[-1, 6])
    if g_last >= 0.5:
        parts.append("open gripper")
    else:
        parts.append("close gripper")
    return ", ".join(parts)


def language_actions_to_text(
    language_actions: np.ndarray,
    window_size: int = 8,
    include_rotation: bool = True,
) -> tuple[list[str], list[str]]:
    """Convert per-step language-action vectors to text for an entire trajectory.

    Aggregates over a temporal window, then produces both verbose and no-number formats.

    Args:
        language_actions: (T, 7) per-step language-action vectors.
        window_size: Temporal aggregation window.
        include_rotation: Whether to include rotation in the text.

    Returns:
        Tuple of (verbose_texts, nonum_texts), each a list of T strings.
    """
    aggregated = aggregate_language_actions(language_actions, window_size)

    verbose_texts = []
    nonum_texts = []
    for t in range(len(aggregated)):
        verbose = summarize_numeric_actions(
            aggregated[t], sum_decimal="1f", include_rotation=include_rotation
        )
        nonum = summarize_numeric_actions(
            aggregated[t], sum_decimal="no_number", include_rotation=include_rotation
        )
        verbose_texts.append(verbose or "close gripper")
        nonum_texts.append(nonum or "close gripper")

    return verbose_texts, nonum_texts
