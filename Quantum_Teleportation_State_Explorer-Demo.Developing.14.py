from __future__ import annotations

import queue
import threading
import warnings
import os
import json
import csv
import sys
import subprocess
from datetime import datetime
from dataclasses import dataclass, replace, asdict, is_dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from itertools import product

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

# Do not force matplotlib.use("TkAgg") in Spyder.
# Spyder usually starts a Qt backend first; forcing TkAgg then causes an ImportError.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scipy.optimize import minimize, brentq
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial.hermite import hermgauss

try:
    import qutip as qt
except ImportError as exc:
    raise ImportError("QuTiP is required. Install it with: pip install qutip") from exc


APP_INFORMATION_TEXT = r"""
Quantum Teleportation State Explorer & Optimizer
================================================

What this app does
------------------
This app is an inverse-design tool for noisy continuous-variable quantum teleportation. Instead of only simulating a predefined state, it searches over arbitrary Fock-basis states and asks which state is optimal for the experimental setup selected by the user.

The optimized single-mode state is written as the LaTeX expression

    \[
    |\psi\rangle = \sum_{n=0}^{N_{\rm cut}-1} c_n |n\rangle .
    \]

The app chooses the complex coefficients \(c_n\) subject to normalization, energy, and any optional experimental constraints.

Core optimization problem
-------------------------
For a pure input state \(|\psi\rangle\) and a teleportation channel \(\mathcal E\), the basic objective is

    \[
    F(\psi)=\langle\psi|\mathcal E(|\psi\rangle\langle\psi|)|\psi\rangle .
    \]

The always-active constraints are

    \[
    \langle\psi|\psi\rangle=1,
    \qquad
    \langle\psi|\hat n|\psi\rangle=N .
    \]

Here \(N\) is the target mean photon number. Optional constraints can restrict parity, displacement, Fock support, photon-number statistics, quadrature variances, Wigner negativity, high-Fock tail population, or overlap with reference states.

Wigner-space channel model
--------------------------
The teleportation channel is evaluated in phase space. For the gain/noise part, the Wigner function transforms approximately as

    \[
    W_{\rm out}(x,p)=
    \int G_Y(x-gx',p-gp') W_{\rm in}(x',p')\,dx'\,dp' .
    \]

The matrix \(Y\) is the Gaussian noise covariance. It can include finite squeezing, anisotropy, detector noise, thermal noise, correlated EPR noise, rotated EPR noise, loss-like noise, phase diffusion, and deterministic drift.

Fidelity from Wigner functions
------------------------------
For a pure input state, the Wigner overlap formula is

    \[
    F = 2\pi\int W_{\rm in}(x,p)W_{\rm out}(x,p)\,dx\,dp .
    \]

This is the main diagnostic plotted by the app. In fast modes, Wigner functions are not necessarily recomputed at every optimizer step. They are produced only when needed for visualization or when the objective explicitly depends on Wigner negativity.

Wigner negativity and survival
------------------------------
The Wigner negativity is

    \[
    \mathcal N(W)=\frac12\int \left(|W(x,p)|-W(x,p)\right)\,dx\,dp .
    \]

When the negativity-survival objective is enabled, the app computes

    \[
    R_{\rm surv}=\frac{\mathcal N(W_{\rm out})}{\mathcal N(W_{\rm in})} .
    \]

If \(\mathcal N(W_{\rm in})\) is numerically zero, the ratio is undefined. This is why coherent states should be reported as "N/A" for negativity survival rather than assigned a fake ratio. Values below the numerical display tolerance are treated as zero in user-facing plots.

Computation profiles
--------------------
Fast Search is designed for speed. It reduces live plotting and usually shows Wigner diagnostics only at the final result.

Balanced is the recommended default. It uses moderate numerical settings and low-resolution live Wigner preview.

Presentation mode prioritizes visual feedback. It updates Wigner plots more often and is useful for demonstrations. It is slower, so use moderate cutoffs and fewer random starts.

High Accuracy is for final verification. It uses larger grids and more iterations, disables live Wigner preview, and focuses on a reliable final answer.

Live Wigner preview
-------------------
Live Wigner preview is a visualization option. It does not change the mathematical objective. The fastest approach is to optimize without Wigner plots inside the objective and draw Wigner functions only periodically or at the end.

If the objective itself involves Wigner negativity, then Wigner calculations are mathematically necessary during optimization.

Manual experiment configuration
-------------------------------
All experimental settings are controlled directly by the visible noise and constraint fields. The old experimental-preset system has been removed, so nothing automatically overwrites these values. Computation profiles only change runtime/numerical settings when Apply Profile is pressed.

For an ideal sanity check, disable all extra imperfection switches, use unit gain, and choose whether finite-squeezing noise should be included.

For a finite-squeezing-only test, enable finite-squeezing noise, set the squeezing parameter r, and leave detector, thermal, phase, loss, correlated, rotated, and drift noises off.

Detector inefficiency is controlled by the detector-noise switch, detector efficiency eta, and detector-noise strength.

Thermal and loss-like effects are controlled by the thermal-noise and loss/mode-mismatch settings.

Phase-noisy experiment adds phase diffusion, which strongly affects displaced states such as coherent states.

Low-energy lab realistic combines moderate squeezing, detector inefficiency, weak thermal noise, and phase uncertainty.

Comparison & Validation Tests
-----------------------------
After an optimized state is found, open the Comparison & Validation tab and press Run Comparison. The app evaluates the optimized state and reference states under the same configured teleportation channel. It reports fidelity, input/output Wigner negativity, negativity survival, Wigner-shape overlap, energy, parity, and whether each reference state satisfies the active constraints.

The comparison section first generates results inside the app without writing files automatically. After checking the plots, press Save Comparison Results to export the figures, CSV table, JSON manifest, and large Wigner-pair images into a timestamped folder. Wigner plots are kept large and separate, not compressed into tiny panels.

How to interpret the final result
---------------------------------
A state with high fidelity is not automatically the most experimentally useful state. Check mean photon number, high-Fock population, number of dominant coefficients, Wigner negativity, and convergence warnings.

If a non-coherent state beats coherent under a noisy channel, that means the selected noise model and objective favor that state. It does not invalidate the coherent-state benchmark for the ideal unit-gain isotropic channel.
"""


# =============================================================================
# Export settings
# =============================================================================

APP_VERSION = "Demo.Developing.14"


def _default_export_root(subfolder: str) -> str:
    """Return a portable output folder.

    The older development versions used a hard-coded OneDrive path.  That is
    convenient on one machine but breaks on every other computer.  The new
    default is user-home based and can still be overridden through the
    QTSE_OUTPUT_DIR environment variable.
    """
    base = os.environ.get("QTSE_OUTPUT_DIR")
    if not base:
        documents = os.path.join(os.path.expanduser("~"), "Documents")
        base = documents if os.path.isdir(documents) else os.path.expanduser("~")
        base = os.path.join(base, "QTSE_Outputs")
    return os.path.join(base, subfolder)


DEFAULT_STATE_EXPORT_ROOT = _default_export_root("Optimized_States")
DEFAULT_COMPARISON_EXPORT_ROOT = _default_export_root("Comparison_Runs")
STATE_EXPORT_FORMAT_VERSION = "cv_optimized_state_v3_extended_noise"
COMPARISON_EXPORT_FORMAT_VERSION = "cv_app_comparison_v1"
NEGATIVITY_ZERO_TOL = 1e-4
# Values below this display tolerance are treated as numerical zero in reports/plots.
# This prevents coherent or other Gaussian states from receiving artificial
# Wigner-negativity survival ratios due to finite grid and Fock-cutoff artifacts.
NEGATIVITY_DISPLAY_ZERO_TOL = 1e-4

# Numerical safety for full-covariance Gaussian noise.
# A covariance matrix with |rho_xp| extremely close to one is mathematically
# close to singular.  On a finite Wigner grid that can collapse the convolution
# kernel into a line-like object and produce artificial jumps in comparison
# sweeps.  The cap below keeps correlated/rotated noise strongly correlated
# but numerically well-conditioned.
MAX_EFFECTIVE_NOISE_CORRELATION = 0.98
MIN_NOISE_DET_RELATIVE = 1e-8
COHERENT_LIKE_OVERLAP_TOL = 1.0 - 1e-6
COHERENT_NUMERICAL_NEGATIVITY_TOL = 5e-2


def _json_safe(value):
    """Convert app objects, numpy values, dataclasses, and complex numbers into JSON-safe data."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def save_optimized_state_package(best: Dict[str, object], root_dir: str = DEFAULT_STATE_EXPORT_ROOT) -> str:
    """Save the optimized Fock coefficients and metadata to a timestamped folder."""
    os.makedirs(root_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    folder = os.path.join(root_dir, f"optimized_state_{stamp}")
    os.makedirs(folder, exist_ok=False)

    c = normalize_coeffs(np.asarray(best["coeffs"], dtype=complex))
    cfg = best.get("config")
    params = best.get("params")

    np.savez_compressed(
        os.path.join(folder, "optimized_state.npz"),
        coefficients=c,
        real=c.real,
        imag=c.imag,
        probabilities=np.abs(c) ** 2,
        Ncut=len(c),
        target_energy=getattr(cfg, "target_energy", np.nan),
        fidelity=float(best.get("fidelity", np.nan)),
        objective_fidelity=float(best.get("objective_fidelity", np.nan)),
        selection_score=float(best.get("selection_score", np.nan)),
        input_wigner_negativity=float(best.get("wigner_negativity", np.nan)),
        output_wigner_negativity=float(best.get("output_wigner_negativity", np.nan)),
        negativity_survival_ratio=float(best.get("negativity_survival_ratio", np.nan)),
    )

    metadata = {
        "format_version": STATE_EXPORT_FORMAT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "state_file": "optimized_state.npz",
        "summary": {
            "fidelity": best.get("fidelity"),
            "objective_fidelity": best.get("objective_fidelity"),
            "selection_score": best.get("selection_score"),
            "energy": best.get("energy"),
            "a": best.get("a"),
            "parity": best.get("parity"),
            "input_wigner_negativity": best.get("wigner_negativity"),
            "output_wigner_negativity": best.get("output_wigner_negativity"),
            "negativity_survival_ratio": best.get("negativity_survival_ratio"),
            "objective_negativity_survival_ratio": best.get("objective_negativity_survival_ratio"),
        },
        "config": asdict(cfg) if cfg is not None else None,
        "channel": asdict(params) if params is not None else None,
        "environment": {
            "app_version": APP_VERSION,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "qutip_version": getattr(qt, "__version__", "unknown"),
            "quadrature_convention": "x=(a+a_dag)/sqrt(2), p=(a-a_dag)/(i*sqrt(2)), Var(vacuum)=0.5",
            "wigner_normalization": "integral W(x,p) dx dp = 1; fidelity for pure input is 2*pi*integral W_in W_out dx dp",
            "fock_ordering": "coefficients[n] is the amplitude of |n>, with n=0 the vacuum",
        },
    }

    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2, ensure_ascii=False)

    with open(os.path.join(folder, "coefficients.csv"), "w", encoding="utf-8") as f:
        f.write("n,Re(c_n),Im(c_n),probability\n")
        for n, cn in enumerate(c):
            f.write(f"{n},{cn.real:.16e},{cn.imag:.16e},{abs(cn)**2:.16e}\n")

    with open(os.path.join(folder, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Optimized CV teleportation state export\n")
        f.write("=======================================\n\n")
        f.write("Load optimized_state.npz and use the complex array named 'coefficients'.\n")
        f.write("metadata.json contains the channel, constraints, and objective settings.\n")

    return folder


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UserConfig:
    # Hilbert space and numerical grid
    Ncut: int = 10
    target_energy: float = 1.0
    x_max: float = 6.0
    grid_points: int = 81

    # Optimization
    n_starts: int = 6
    maxiter: int = 80
    ftol: float = 1e-7
    random_seed: int = 1234
    live_update_every: int = 3

    # Numerical stability / convergence controls
    # Broad random seeds sample the full allowed Fock subspace with exact mean energy.
    use_broad_random_seeds: bool = True

    # Tail control prevents the optimizer from exploiting the artificial Fock cutoff.
    # It is one of the most important controls for Ncut convergence.
    use_tail_probability_constraint: bool = False
    tail_levels: int = 3
    max_tail_probability: float = 1e-3

    # Optional soft penalty. It changes the optimized score to
    # score = fidelity - tail_penalty_strength * P_tail.
    use_tail_penalty: bool = False
    tail_penalty_strength: float = 0.02

    # After all starts, rerun SLSQP from the best state.
    use_best_polishing: bool = True
    polish_maxiter: int = 80

    # Base channel
    use_finite_squeezing_noise: bool = True
    r: float = 0.9

    use_gain_mismatch: bool = False
    gain_if_enabled: float = 0.75
    gain_if_disabled: float = 1.0

    use_anisotropic_noise: bool = False
    anisotropic_noise_x2: float = 0.04
    anisotropic_noise_p2: float = 0.50

    use_thermal_noise: bool = False
    thermal_nbar: float = 0.5
    thermal_noise_strength: float = 0.02

    use_detector_noise: bool = False
    detector_efficiency_eta: float = 0.90
    detector_noise_strength: float = 0.05

    use_extra_additive_noise: bool = False
    extra_noise_x2: float = 0.0
    extra_noise_p2: float = 0.0

    # More realistic EPR-resource and calibration noise models.
    # These parameters extend the additive Gaussian noise from a diagonal
    # covariance diag(noise_x2, noise_p2) to a full covariance matrix
    # [[noise_x2, noise_xp], [noise_xp, noise_p2]], optionally with loss and drift.
    use_correlated_epr_noise: bool = False
    correlated_epr_rho: float = 0.30

    use_rotated_asymmetric_epr_noise: bool = False
    rotated_epr_noise_major2: float = 0.25
    rotated_epr_noise_minor2: float = 0.02
    rotated_epr_angle_degrees: float = 30.0

    use_loss_channel_noise: bool = False
    loss_transmissivity: float = 0.95
    loss_thermal_nbar: float = 0.0

    use_displacement_drift: bool = False
    drift_x: float = 0.0
    drift_p: float = 0.0

    use_phase_diffusion: bool = False
    phase_sigma: float = 0.40
    n_phase_quad: int = 11

    auto_enforce_cp_noise: bool = True

    # Basic state constraints
    use_zero_displacement_constraint: bool = False
    use_even_parity_constraint: bool = False
    use_odd_parity_constraint: bool = False
    use_real_coefficients: bool = False
    use_real_displacement_axis: bool = False

    # Fock-space subspace constraints
    use_fock_range_constraint: bool = False
    fock_min: int = 0
    fock_max: int = 9

    use_modular_fock_constraint: bool = False
    modulus_m: int = 3
    residue_k: int = 0

    # Displacement constraints
    use_bounded_displacement: bool = False
    max_displacement_squared: float = 0.25

    use_fixed_displacement: bool = False
    fixed_a_real: float = 0.0
    fixed_a_imag: float = 0.0

    # Non-displacement energy constraints
    use_fluctuation_energy_min: bool = False
    min_fluctuation_energy: float = 0.10

    use_fluctuation_energy_max: bool = False
    max_fluctuation_energy: float = 0.50

    # Photon-number statistics constraints
    use_photon_variance_min: bool = False
    min_photon_variance: float = 0.0

    use_photon_variance_max: bool = False
    max_photon_variance: float = 1.0

    use_mandel_q_min: bool = False
    min_mandel_q: float = -1.0

    use_mandel_q_max: bool = False
    max_mandel_q: float = 0.0

    # Quadrature moment constraints
    use_x_variance_min: bool = False
    min_x_variance: float = 0.0

    use_x_variance_max: bool = False
    max_x_variance: float = 0.5

    use_p_variance_min: bool = False
    min_p_variance: float = 0.0

    use_p_variance_max: bool = False
    max_p_variance: float = 0.5

    use_covariance_min: bool = False
    min_xp_covariance: float = -0.5

    use_covariance_max: bool = False
    max_xp_covariance: float = 0.5

    # Overlap/non-Gaussianity constraints
    use_coherent_overlap_max: bool = False
    max_coherent_overlap: float = 0.80

    use_squeezed_overlap_max: bool = False
    max_squeezed_overlap: float = 0.80

    use_cat_overlap_min: bool = False
    cat_alpha: float = 1.0
    cat_phase: float = 0.0
    min_cat_overlap: float = 0.50

    # Wigner negativity constraints. These are expensive because Wigner functions
    # must be computed inside the optimizer constraints.
    use_wigner_negativity_min: bool = False
    min_wigner_negativity: float = 0.01

    use_wigner_negativity_max: bool = False
    max_wigner_negativity: float = 0.50

    # Joint objective: fidelity plus Wigner-negativity survival.
    # Input negativity must be above min_input_negativity_for_survival when enabled,
    # so the survival ratio N_out/N_in is well-defined and nontrivial.
    use_negativity_survival_objective: bool = False
    negativity_survival_fidelity_weight: float = 0.50
    negativity_survival_ratio_weight: float = 0.50
    min_input_negativity_for_survival: float = 1e-4
    survival_ratio_clip: float = 1.0

    # Robust optimization objective
    use_robust_average_objective: bool = False
    use_robust_worstcase_objective: bool = False
    robust_r_min: float = 0.75
    robust_r_max: float = 1.05
    robust_r_samples: int = 3
    robust_gain_min: float = 0.90
    robust_gain_max: float = 1.10
    robust_gain_samples: int = 3
    robust_phase_max: float = 0.20
    robust_phase_samples: int = 1

    # Time-dependent field plots for the optimized state.
    # Single-mode oscillator convention:
    # E(t) = E_scale * X_theta, B(t) = B_scale * X_{theta + pi/2}.
    field_omega: float = 1.0
    field_periods: float = 3.0
    field_time_points: int = 400
    field_E_scale: float = 1.0
    field_B_scale: float = 1.0
    field_show_uncertainty: bool = True

    # Workflow and performance/profile settings.
    # Experimental/noise parameters are always edited manually. This compatibility
    # field is kept only so older saved reports do not break.
    computation_profile: str = "Balanced"
    live_wigner_preview_mode: str = "Low-resolution"
    live_wigner_preview_grid_points: int = 61

    # Presentation settings. These only affect plotting, not the optimization.
    wigner_colormap: str = "RdBu_r"
    wigner_color_scale: str = "symmetric"

    # Comparison/validation plotting finesse.  Higher values give smoother
    # parameter sweeps but increase comparison runtime.
    comparison_sweep_points: int = 31

    # Convergence diagnostics. These do not bias the optimization unless a
    # user explicitly enables regularizing constraints above. They only test
    # whether the finite-cutoff result looks stable or cutoff-dependent.
    run_analytic_convergence_diagnostics: bool = True

    # Local probe: reruns short optimizations from small random perturbations
    # around the final state. This tests whether the returned point is at least
    # locally stable for the finite-dimensional problem. It is optional because
    # it can be slow.
    run_local_optimality_probe: bool = False
    local_probe_trials: int = 5
    local_probe_maxiter: int = 25
    local_probe_perturbation: float = 1e-2

    # Cutoff scan: repeats the optimization for a list of Ncut values. This is
    # the strongest numerical convergence test, but it is expensive.
    run_cutoff_scan: bool = False
    cutoff_scan_values: str = "8,10,12,14,16,20,24,30"
    cutoff_scan_n_starts: int = 3
    cutoff_scan_maxiter: int = 50


@dataclass
class ChannelParams:
    gain: float
    noise_x2: float
    noise_p2: float
    phase_sigma: float
    n_phase_quad: int
    noise_xp: float = 0.0
    displacement_x: float = 0.0
    displacement_p: float = 0.0


def channel_kernel_key(params: ChannelParams, xvec: np.ndarray, pvec: np.ndarray) -> Tuple[float, float, float, int, int, float, float]:
    """Hashable key for the Gaussian convolution kernel cache."""
    return (
        round(float(params.noise_x2), 14),
        round(float(params.noise_p2), 14),
        round(float(getattr(params, "noise_xp", 0.0)), 14),
        int(len(xvec)),
        int(len(pvec)),
        round(float(xvec[-1] - xvec[0]), 14),
        round(float(pvec[-1] - pvec[0]), 14),
    )


class StopOptimization(Exception):
    pass


# =============================================================================
# State utilities
# =============================================================================

def normalize_coeffs(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=complex)
    norm = float(np.sqrt(np.vdot(c, c).real))
    if norm < 1e-14:
        out = np.zeros_like(c, dtype=complex)
        out[0] = 1.0
        return out
    return c / norm


def active_fock_indices(cfg: UserConfig) -> np.ndarray:
    indices = np.arange(cfg.Ncut, dtype=int)

    if cfg.use_fock_range_constraint:
        lo = max(0, int(cfg.fock_min))
        hi = min(cfg.Ncut - 1, int(cfg.fock_max))
        indices = indices[(indices >= lo) & (indices <= hi)]

    if cfg.use_even_parity_constraint and cfg.use_odd_parity_constraint:
        raise ValueError("Even parity and odd parity cannot both be enabled.")

    if cfg.use_even_parity_constraint:
        indices = indices[indices % 2 == 0]

    if cfg.use_odd_parity_constraint:
        indices = indices[indices % 2 == 1]

    if cfg.use_modular_fock_constraint:
        m = int(cfg.modulus_m)
        if m < 2:
            raise ValueError("Modular Fock constraint requires modulus m >= 2.")
        k = int(cfg.residue_k) % m
        indices = indices[indices % m == k]

    if len(indices) == 0:
        raise ValueError("No Fock states remain after applying subspace constraints.")

    return indices.astype(int)


def active_to_full(c_active: np.ndarray, active: np.ndarray, Ncut: int) -> np.ndarray:
    c = np.zeros(Ncut, dtype=complex)
    c[active] = c_active
    return c


def y_to_active(
    y: np.ndarray,
    n_active: int,
    normalize: bool = True,
    real_only: bool = False,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if real_only:
        c = y[:n_active].astype(complex)
    else:
        c = y[:n_active] + 1j * y[n_active:2 * n_active]
    return normalize_coeffs(c) if normalize else c


def active_to_y(c_active: np.ndarray, real_only: bool = False) -> np.ndarray:
    c_active = np.asarray(c_active, dtype=complex)
    if real_only:
        return c_active.real.copy()
    return np.concatenate([c_active.real, c_active.imag])


def coeffs_to_ket(c: np.ndarray) -> "qt.Qobj":
    c = normalize_coeffs(c)
    return qt.Qobj(c.reshape((-1, 1)), dims=[[len(c)], [1]])


def a_expectation(c: np.ndarray) -> complex:
    c = normalize_coeffs(c)
    value = 0.0j
    for n in range(1, len(c)):
        value += np.sqrt(n) * np.conjugate(c[n - 1]) * c[n]
    return complex(value)


def a2_expectation(c: np.ndarray) -> complex:
    c = normalize_coeffs(c)
    value = 0.0j
    for n in range(2, len(c)):
        value += np.sqrt(n * (n - 1)) * np.conjugate(c[n - 2]) * c[n]
    return complex(value)


def mean_energy(c: np.ndarray) -> float:
    c = normalize_coeffs(c)
    n = np.arange(len(c))
    return float(np.sum(n * np.abs(c) ** 2).real)


def n2_expectation(c: np.ndarray) -> float:
    c = normalize_coeffs(c)
    n = np.arange(len(c))
    return float(np.sum((n ** 2) * np.abs(c) ** 2).real)


def photon_variance(c: np.ndarray) -> float:
    n = mean_energy(c)
    return max(0.0, n2_expectation(c) - n ** 2)


def mandel_q(c: np.ndarray) -> float:
    n = mean_energy(c)
    if n <= 1e-12:
        return 0.0
    return (photon_variance(c) - n) / n


def parity_expectation(c: np.ndarray) -> float:
    c = normalize_coeffs(c)
    n = np.arange(len(c))
    return float(np.sum(((-1.0) ** n) * np.abs(c) ** 2).real)


def tail_probability(c: np.ndarray, tail_levels: int) -> float:
    """Probability contained in the highest tail_levels Fock states.

    A large value means the result is touching the artificial cutoff. Such states
    often change strongly when Ncut is increased.
    """
    c = normalize_coeffs(c)
    tail_levels = int(max(1, tail_levels))
    start = max(0, len(c) - tail_levels)
    return float(np.sum(np.abs(c[start:]) ** 2).real)


def selection_score_from_components(
    F_raw: float,
    c: np.ndarray,
    cfg: UserConfig,
    survival_ratio: Optional[float] = None,
    input_negativity: Optional[float] = None,
) -> float:
    """Return the scalar objective score used by SLSQP.

    Default: score = fidelity.

    If the negativity-survival objective is enabled, the score becomes a
    normalized weighted sum

        score = (w_F F + w_R R_surv) / (w_F + w_R),

    where R_surv = N(W_out)/N(W_in). States with input Wigner negativity below
    min_input_negativity_for_survival are assigned a large negative score, so
    non-negative states cannot be selected and division by zero is avoided.
    """
    c = normalize_coeffs(c)
    F_raw = float(F_raw)

    if cfg.use_negativity_survival_objective:
        nin = 0.0 if input_negativity is None else float(input_negativity)
        if nin < float(cfg.min_input_negativity_for_survival):
            # Large but finite penalty keeps SLSQP numerically stable while
            # preventing non-negative states from being selected.
            score = -1.0e6 - (float(cfg.min_input_negativity_for_survival) - nin)
        else:
            ratio = 0.0 if survival_ratio is None else float(survival_ratio)
            if not np.isfinite(ratio):
                ratio = 0.0
            ratio = float(np.clip(ratio, 0.0, max(1e-12, float(cfg.survival_ratio_clip))))

            wF = max(0.0, float(cfg.negativity_survival_fidelity_weight))
            wR = max(0.0, float(cfg.negativity_survival_ratio_weight))
            denom = wF + wR
            if denom <= 0.0:
                wF, wR, denom = 1.0, 0.0, 1.0
            score = (wF * F_raw + wR * ratio) / denom
    else:
        score = F_raw

    if cfg.use_tail_penalty:
        score -= float(cfg.tail_penalty_strength) * tail_probability(c, cfg.tail_levels)

    return float(score)


def selection_score_from_raw_fidelity(F_raw: float, c: np.ndarray, cfg: UserConfig) -> float:
    """Backward-compatible score helper for pure-fidelity contexts."""
    return selection_score_from_components(F_raw, c, cfg)


def quadrature_moments(c: np.ndarray) -> Dict[str, float]:
    c = normalize_coeffs(c)
    a = a_expectation(c)
    a2 = a2_expectation(c)
    n = mean_energy(c)

    x_mean = np.sqrt(2.0) * a.real
    p_mean = np.sqrt(2.0) * a.imag

    x2 = n + 0.5 + a2.real
    p2 = n + 0.5 - a2.real

    x_var = float(max(0.0, x2 - x_mean ** 2))
    p_var = float(max(0.0, p2 - p_mean ** 2))

    # 1/2 <xp + px> - <x><p> = Im(<a^2>) - <x><p>
    xp_cov = float(a2.imag - x_mean * p_mean)

    return {
        "x_mean": float(x_mean),
        "p_mean": float(p_mean),
        "x_var": x_var,
        "p_var": p_var,
        "xp_cov": xp_cov,
    }


def state_diagnostics(c: np.ndarray) -> Dict[str, object]:
    c = normalize_coeffs(c)
    a = a_expectation(c)
    n = mean_energy(c)
    var_n = photon_variance(c)
    q = mandel_q(c)
    quad = quadrature_moments(c)
    return {
        "energy": n,
        "n_variance": var_n,
        "mandel_q": q,
        "a": a,
        "displacement_squared": abs(a) ** 2,
        "fluctuation_energy": n - abs(a) ** 2,
        "parity": parity_expectation(c),
        **quad,
    }


def quadrature_theta_stats(c: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean and variance of the rotated quadrature

        X_theta = (a exp(-i theta) + a^dagger exp(i theta)) / sqrt(2).

    The electric-field plot uses X_theta. The magnetic-field plot uses the
    conjugate quadrature X_{theta + pi/2}. This is the single-mode oscillator
    convention, where electric and magnetic energies exchange in time.
    """
    c = normalize_coeffs(c)
    theta = np.asarray(theta, dtype=float)

    a = a_expectation(c)
    a2 = a2_expectation(c)
    n = mean_energy(c)

    mean = np.sqrt(2.0) * np.real(a * np.exp(-1j * theta))
    second = n + 0.5 + np.real(a2 * np.exp(-2j * theta))
    variance = np.maximum(0.0, second - mean ** 2)
    return mean.astype(float), variance.astype(float)


def field_time_series(c: np.ndarray, cfg: UserConfig) -> Dict[str, np.ndarray]:
    """
    Time-dependent electric and magnetic field expectation values and
    uncertainty bands for the optimized single-mode state.

    Units are dimensionless unless E_scale and B_scale are changed. A physical
    SI amplitude can be inserted through E_scale and B_scale if a mode volume,
    angular frequency, and geometry are known.
    """
    omega = float(cfg.field_omega)
    if omega <= 0.0:
        omega = 1.0

    periods = max(0.1, float(cfg.field_periods))
    n_points = max(20, int(cfg.field_time_points))

    t_max = 2.0 * np.pi * periods / omega
    t = np.linspace(0.0, t_max, n_points)
    theta = omega * t

    E_mean_q, E_var_q = quadrature_theta_stats(c, theta)
    B_mean_q, B_var_q = quadrature_theta_stats(c, theta + np.pi / 2.0)

    E_scale = float(cfg.field_E_scale)
    B_scale = float(cfg.field_B_scale)

    return {
        "t": t,
        "E_mean": E_scale * E_mean_q,
        "E_std": abs(E_scale) * np.sqrt(E_var_q),
        "B_mean": B_scale * B_mean_q,
        "B_std": abs(B_scale) * np.sqrt(B_var_q),
    }



def coherent_overlap(c: np.ndarray, coh: np.ndarray) -> float:
    c = normalize_coeffs(c)
    coh = normalize_coeffs(coh)
    return float(abs(np.vdot(coh, c)) ** 2)


def squeezed_overlap(c: np.ndarray, sq: np.ndarray) -> float:
    c = normalize_coeffs(c)
    sq = normalize_coeffs(sq)
    return float(abs(np.vdot(sq, c)) ** 2)


def cat_coeffs(Ncut: int, alpha: float, phase: float) -> np.ndarray:
    ket_plus = qt.coherent(Ncut, alpha)
    ket_minus = qt.coherent(Ncut, -alpha)
    ket = ket_plus + np.exp(1j * phase) * ket_minus
    c = np.asarray(ket.full()).flatten()
    return normalize_coeffs(c)


def cat_overlap(c: np.ndarray, cat: np.ndarray) -> float:
    c = normalize_coeffs(c)
    cat = normalize_coeffs(cat)
    return float(abs(np.vdot(cat, c)) ** 2)


def state_allowed(c: np.ndarray, cfg: UserConfig, tol: float = 5e-4) -> bool:
    c = normalize_coeffs(c)
    d = state_diagnostics(c)

    if abs(float(d["energy"]) - cfg.target_energy) > tol:
        return False

    if cfg.use_zero_displacement_constraint and abs(complex(d["a"])) > 5e-5:
        return False

    if cfg.use_even_parity_constraint and np.sum(np.abs(c[1::2]) ** 2) > 5e-5:
        return False

    if cfg.use_odd_parity_constraint and np.sum(np.abs(c[0::2]) ** 2) > 5e-5:
        return False

    if cfg.use_real_coefficients and np.sum(np.abs(c.imag) ** 2) > 5e-5:
        return False

    if cfg.use_bounded_displacement and float(d["displacement_squared"]) > cfg.max_displacement_squared + tol:
        return False

    return True


# =============================================================================
# Channel and Wigner functions
# =============================================================================

def _enforce_full_gaussian_cp(
    gain: float,
    noise_x2: float,
    noise_p2: float,
    noise_xp: float,
    auto_enforce: bool,
) -> Tuple[float, float, float]:
    """Return a physical and numerically stable full covariance matrix.

    For the teleportation-style one-mode Gaussian channel with X = g I and

        Y = [[noise_x2, noise_xp], [noise_xp, noise_p2]],

    complete positivity requires

        det(Y) >= (1 - g^2)^2 / 4.

    In addition to the formal CP condition, the app keeps the effective
    correlation |noise_xp|/sqrt(noise_x2 noise_p2) below a numerical cap.
    This prevents nearly singular kernels from producing artificial jumps in
    Wigner-space comparison sweeps when rotated and correlated noise are used
    together.
    """
    nx = float(noise_x2)
    np_ = float(noise_p2)
    n_xp = float(noise_xp)

    if nx < 0.0 or np_ < 0.0:
        raise ValueError("Noise variances must be non-negative.")

    if nx <= 0.0 or np_ <= 0.0:
        if abs(n_xp) > 0.0:
            if not auto_enforce:
                warnings.warn(
                    "Noise covariance has nonzero xp covariance but a zero diagonal variance.",
                    RuntimeWarning,
                )
                return nx, np_, n_xp
            diag = abs(n_xp) / MAX_EFFECTIVE_NOISE_CORRELATION + 1e-14
            nx = max(nx, diag)
            np_ = max(np_, diag)

    product = max(nx * np_, 0.0)

    if product > 0.0:
        max_abs_xp = MAX_EFFECTIVE_NOISE_CORRELATION * np.sqrt(product)
        if abs(n_xp) > max_abs_xp:
            if not auto_enforce:
                warnings.warn(
                    "Selected full Gaussian noise covariance is nearly singular or non-positive. "
                    "Enable Auto-enforce CP/noise stability or reduce xp-correlation.",
                    RuntimeWarning,
                )
            else:
                n_xp = float(np.sign(n_xp) * max_abs_xp)

    bound = ((1.0 - float(gain) ** 2) ** 2) / 4.0
    product = max(nx * np_, 0.0)
    det = product - n_xp ** 2

    # A relative determinant floor avoids numerically line-like Gaussian kernels.
    # It is zero only when the covariance itself is zero.
    det_floor = MIN_NOISE_DET_RELATIVE * max(product, n_xp ** 2, 1.0 if (product > 0.0 or abs(n_xp) > 0.0) else 0.0)
    required_det = max(bound, det_floor)

    if det + 1e-15 >= required_det:
        return float(nx), float(np_), float(n_xp)

    msg = (
        "Selected full Gaussian noise covariance violates positivity/CP or is numerically singular: "
        f"det(Y)={det:.6g}, required det(Y)>={required_det:.6g}."
    )

    if not auto_enforce:
        warnings.warn(msg, RuntimeWarning)
        return float(nx), float(np_), float(n_xp)

    if nx <= 0.0 or np_ <= 0.0:
        diag = np.sqrt(required_det + n_xp ** 2) + 1e-12
        nx = max(nx, diag)
        np_ = max(np_, diag)
    else:
        scale = np.sqrt((required_det + n_xp ** 2 + 1e-15) / max(nx * np_, 1e-300))
        scale = max(1.0, float(scale))
        nx *= scale
        np_ *= scale

    # After scaling, clip once more to keep the kernel well-conditioned.
    product = max(nx * np_, 0.0)
    if product > 0.0:
        max_abs_xp = MAX_EFFECTIVE_NOISE_CORRELATION * np.sqrt(product)
        if abs(n_xp) > max_abs_xp:
            n_xp = float(np.sign(n_xp) * max_abs_xp)

    return float(nx), float(np_), float(n_xp)

def build_channel(cfg: UserConfig) -> ChannelParams:
    gain = cfg.gain_if_enabled if cfg.use_gain_mismatch else cfg.gain_if_disabled

    # Finite squeezing is the baseline teleportation noise.  Additional
    # experimental noise switches are added on top of this baseline.  Earlier
    # development versions treated anisotropic noise as a replacement for the
    # finite-squeezing noise; that made r-sweeps flat whenever anisotropic
    # noise was enabled.  The additive interpretation is more useful for
    # experimental configurations and keeps the squeezing parameter active.
    base = float(np.exp(-2.0 * cfg.r)) if cfg.use_finite_squeezing_noise else 0.0
    nx = base
    np_ = base

    n_xp = 0.0

    if cfg.use_anisotropic_noise:
        nx += float(cfg.anisotropic_noise_x2)
        np_ += float(cfg.anisotropic_noise_p2)

    if cfg.use_thermal_noise:
        added = cfg.thermal_noise_strength * (2.0 * cfg.thermal_nbar + 1.0)
        nx += added
        np_ += added

    if cfg.use_detector_noise:
        eta = float(cfg.detector_efficiency_eta)
        if eta <= 0.0 or eta > 1.0:
            raise ValueError("Detector efficiency eta must satisfy 0 < eta <= 1.")
        added = cfg.detector_noise_strength * (1.0 - eta) / eta
        nx += added
        np_ += added

    if cfg.use_extra_additive_noise:
        nx += float(cfg.extra_noise_x2)
        np_ += float(cfg.extra_noise_p2)

    # Loss or mode mismatch to a vacuum/thermal environment.
    # In the convention Var(vacuum quadrature)=1/2,
    # x -> sqrt(T) x + sqrt(1-T) x_env.
    if cfg.use_loss_channel_noise:
        T = float(cfg.loss_transmissivity)
        if T <= 0.0 or T > 1.0:
            raise ValueError("Loss transmissivity must satisfy 0 < T <= 1.")
        n_env = max(0.0, float(cfg.loss_thermal_nbar))
        gain *= np.sqrt(T)
        loss_added = (1.0 - T) * (2.0 * n_env + 1.0) / 2.0
        nx += loss_added
        np_ += loss_added

    # Rotated asymmetric EPR noise: add a covariance ellipse whose principal
    # axes are rotated by theta relative to the x,p grid.
    if cfg.use_rotated_asymmetric_epr_noise:
        v_major = float(cfg.rotated_epr_noise_major2)
        v_minor = float(cfg.rotated_epr_noise_minor2)
        if v_major < 0.0 or v_minor < 0.0:
            raise ValueError("Rotated EPR noise variances must be non-negative.")
        theta = np.deg2rad(float(cfg.rotated_epr_angle_degrees))
        cth = np.cos(theta)
        sth = np.sin(theta)
        nx += v_major * cth ** 2 + v_minor * sth ** 2
        np_ += v_major * sth ** 2 + v_minor * cth ** 2
        n_xp += (v_major - v_minor) * sth * cth

    # Correlated non-ideal EPR noise: add xp covariance proportional to the
    # geometric mean of the diagonal noise.
    if cfg.use_correlated_epr_noise:
        rho = float(np.clip(cfg.correlated_epr_rho, -0.999999, 0.999999))
        n_xp += rho * np.sqrt(max(nx * np_, 0.0))

    if nx < 0.0 or np_ < 0.0:
        raise ValueError("Noise variances must be non-negative.")

    nx, np_, n_xp = _enforce_full_gaussian_cp(gain, nx, np_, n_xp, cfg.auto_enforce_cp_noise)

    phase_sigma = cfg.phase_sigma if cfg.use_phase_diffusion else 0.0

    dx = float(cfg.drift_x) if cfg.use_displacement_drift else 0.0
    dp = float(cfg.drift_p) if cfg.use_displacement_drift else 0.0

    return ChannelParams(
        gain=float(gain),
        noise_x2=float(nx),
        noise_p2=float(np_),
        phase_sigma=float(phase_sigma),
        n_phase_quad=int(cfg.n_phase_quad),
        noise_xp=float(n_xp),
        displacement_x=dx,
        displacement_p=dp,
    )

def channel_from_values(gain: float, r: float, phase_sigma: float, cfg: UserConfig, base: ChannelParams) -> ChannelParams:
    # Vary only the finite-squeezing contribution while preserving the other
    # configured noise contributions.  This keeps robustness scans and
    # comparison sweeps meaningful even when anisotropic, thermal, detector,
    # phase, or extra noise is also active.
    if cfg.use_finite_squeezing_noise:
        base_clean = float(np.exp(-2.0 * cfg.r))
        varied_clean = float(np.exp(-2.0 * r))
        extra_x = max(0.0, float(base.noise_x2) - base_clean)
        extra_p = max(0.0, float(base.noise_p2) - base_clean)
        nx = varied_clean + extra_x
        np_ = varied_clean + extra_p
    else:
        nx = float(base.noise_x2)
        np_ = float(base.noise_p2)

    n_xp = getattr(base, "noise_xp", 0.0)
    nx, np_, n_xp = _enforce_full_gaussian_cp(gain, nx, np_, n_xp, cfg.auto_enforce_cp_noise)

    return ChannelParams(
        gain=float(gain),
        noise_x2=float(nx),
        noise_p2=float(np_),
        phase_sigma=float(phase_sigma),
        n_phase_quad=base.n_phase_quad,
        noise_xp=float(n_xp),
        displacement_x=float(getattr(base, "displacement_x", 0.0)),
        displacement_p=float(getattr(base, "displacement_p", 0.0)),
    )

def robust_channels(cfg: UserConfig, base: ChannelParams) -> List[ChannelParams]:
    if not (cfg.use_robust_average_objective or cfg.use_robust_worstcase_objective):
        return [base]

    r_samples = max(1, int(cfg.robust_r_samples))
    g_samples = max(1, int(cfg.robust_gain_samples))
    p_samples = max(1, int(cfg.robust_phase_samples))

    if cfg.use_finite_squeezing_noise:
        r_values = np.linspace(cfg.robust_r_min, cfg.robust_r_max, r_samples)
    else:
        r_values = np.array([cfg.r])

    g_values = np.linspace(cfg.robust_gain_min, cfg.robust_gain_max, g_samples)

    if p_samples == 1:
        phase_values = np.array([base.phase_sigma])
    else:
        phase_values = np.linspace(0.0, cfg.robust_phase_max, p_samples)

    channels = []
    for r, g, ph in product(r_values, g_values, phase_values):
        channels.append(channel_from_values(float(g), float(r), float(ph), cfg, base))

    return channels


def make_grid(cfg: UserConfig) -> Tuple[np.ndarray, np.ndarray]:
    points = int(cfg.grid_points)
    if points % 2 == 0:
        points += 1
    xvec = np.linspace(-cfg.x_max, cfg.x_max, points)
    pvec = np.linspace(-cfg.x_max, cfg.x_max, points)
    return xvec, pvec


def wigner_from_coeffs(c: np.ndarray, xvec: np.ndarray, pvec: np.ndarray) -> np.ndarray:
    ket = coeffs_to_ket(c)
    rho = qt.ket2dm(ket)
    return np.asarray(qt.wigner(rho, xvec, pvec), dtype=float)


def gaussian_kernel(
    xvec: np.ndarray,
    pvec: np.ndarray,
    noise_x2: float,
    noise_p2: float,
    noise_xp: float = 0.0,
) -> np.ndarray:
    """Gaussian convolution kernel for a full 2D noise covariance matrix.

    The covariance matrix is

        Y = [[noise_x2, noise_xp], [noise_xp, noise_p2]].

    Setting noise_xp=0 recovers the original diagonal-noise kernel.
    """
    dx = xvec[1] - xvec[0]
    dp = pvec[1] - pvec[0]
    X, P = np.meshgrid(xvec, pvec, indexing="xy")

    noise_x2 = float(noise_x2)
    noise_p2 = float(noise_p2)
    noise_xp = float(noise_xp)

    if abs(noise_x2) < 1e-14 and abs(noise_p2) < 1e-14 and abs(noise_xp) < 1e-14:
        G = np.zeros_like(X)
        G[len(pvec) // 2, len(xvec) // 2] = 1.0 / (dx * dp)
        return G

    noise_x2 = max(noise_x2, 1e-14)
    noise_p2 = max(noise_p2, 1e-14)
    det = noise_x2 * noise_p2 - noise_xp ** 2

    if det <= 1e-18:
        # Numerical safety fallback. The CP guard should normally prevent this.
        noise_xp = 0.0
        det = noise_x2 * noise_p2

    exponent = (noise_p2 * X ** 2 - 2.0 * noise_xp * X * P + noise_x2 * P ** 2) / det
    G = np.exp(-0.5 * exponent)

    # Discrete normalization is the normalization that actually matters on the
    # finite phase-space grid.  Avoiding the analytic prefactor is numerically
    # safer for very broad or very narrow kernels.
    norm = float(np.sum(G) * dx * dp)
    if norm <= 1e-300 or not np.isfinite(norm):
        G = np.zeros_like(X)
        G[len(pvec) // 2, len(xvec) // 2] = 1.0 / (dx * dp)
        return G
    G /= norm
    return G

def apply_gain(W: np.ndarray, xvec: np.ndarray, pvec: np.ndarray, gain: float) -> np.ndarray:
    if abs(gain) < 1e-14:
        raise ValueError("Gain must be nonzero.")
    if abs(gain - 1.0) < 1e-14:
        return W.copy()

    X, P = np.meshgrid(xvec, pvec, indexing="xy")
    interp = RegularGridInterpolator((pvec, xvec), W, bounds_error=False, fill_value=0.0)
    points = np.column_stack([(P / gain).ravel(), (X / gain).ravel()])
    return interp(points).reshape(W.shape) / (gain ** 2)


def rotate_wigner(W: np.ndarray, xvec: np.ndarray, pvec: np.ndarray, phi: float) -> np.ndarray:
    X, P = np.meshgrid(xvec, pvec, indexing="xy")
    c = np.cos(phi)
    s = np.sin(phi)

    X_old = X * c + P * s
    P_old = -X * s + P * c

    interp = RegularGridInterpolator((pvec, xvec), W, bounds_error=False, fill_value=0.0)
    points = np.column_stack([P_old.ravel(), X_old.ravel()])
    return interp(points).reshape(W.shape)


def shift_wigner(W: np.ndarray, xvec: np.ndarray, pvec: np.ndarray, dx: float, dp: float) -> np.ndarray:
    """Displace a Wigner function by (dx, dp) in phase space."""
    if abs(dx) < 1e-14 and abs(dp) < 1e-14:
        return W

    X, P = np.meshgrid(xvec, pvec, indexing="xy")
    interp = RegularGridInterpolator((pvec, xvec), W, bounds_error=False, fill_value=0.0)
    points = np.column_stack([(P - dp).ravel(), (X - dx).ravel()])
    return interp(points).reshape(W.shape)


def apply_phase_diffusion(
    W: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    sigma: float,
    n_quad: int,
) -> np.ndarray:
    if sigma <= 0.0:
        return W
    if n_quad % 2 == 0:
        n_quad += 1

    nodes, weights = hermgauss(n_quad)
    phis = np.sqrt(2.0) * sigma * nodes
    weights = weights / np.sqrt(np.pi)

    X, P = np.meshgrid(xvec, pvec, indexing="xy")
    interp = RegularGridInterpolator((pvec, xvec), W, bounds_error=False, fill_value=0.0)
    out = np.zeros_like(W)
    for phi, weight in zip(phis, weights):
        cphi = np.cos(float(phi))
        sphi = np.sin(float(phi))
        X_old = X * cphi + P * sphi
        P_old = -X * sphi + P * cphi
        points = np.column_stack([P_old.ravel(), X_old.ravel()])
        out += weight * interp(points).reshape(W.shape)
    return out


def apply_channel(
    W: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    params: ChannelParams,
    precomputed_kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    dx = xvec[1] - xvec[0]
    dp = pvec[1] - pvec[0]

    Wout = apply_gain(W, xvec, pvec, params.gain)

    n_xp = float(getattr(params, "noise_xp", 0.0))
    if params.noise_x2 > 1e-14 or params.noise_p2 > 1e-14 or abs(n_xp) > 1e-14:
        G = precomputed_kernel
        if G is None:
            G = gaussian_kernel(xvec, pvec, params.noise_x2, params.noise_p2, n_xp)
        Wout = fftconvolve(Wout, G, mode="same") * dx * dp

    Wout = apply_phase_diffusion(Wout, xvec, pvec, params.phase_sigma, params.n_phase_quad)

    Wout = shift_wigner(
        Wout,
        xvec,
        pvec,
        float(getattr(params, "displacement_x", 0.0)),
        float(getattr(params, "displacement_p", 0.0)),
    )

    norm = float(np.sum(Wout) * dx * dp)
    if abs(norm) > 1e-13 and abs(norm - 1.0) > 0.05:
        warnings.warn(
            f"Output Wigner norm={norm:.4g} deviates by more than 5% from 1. "
            "The phase-space window may be too small for the selected gain/noise/drift. "
            "Increase x_max or grid_points for a safer fidelity estimate.",
            RuntimeWarning,
            stacklevel=2,
        )
    if abs(norm) > 1e-13:
        Wout = Wout / norm

    return Wout


def wigner_fidelity(W_in: np.ndarray, W_out: np.ndarray, xvec: np.ndarray, pvec: np.ndarray) -> float:
    dx = xvec[1] - xvec[0]
    dp = pvec[1] - pvec[0]
    return float((2.0 * np.pi * np.sum(W_in * W_out) * dx * dp).real)


def wigner_negativity(W: np.ndarray, xvec: np.ndarray, pvec: np.ndarray) -> float:
    dx = xvec[1] - xvec[0]
    dp = pvec[1] - pvec[0]
    return float(0.5 * np.sum(np.abs(W) - W) * dx * dp)


def is_coherent_like_state(c: np.ndarray, cfg: Optional[UserConfig] = None, threshold: float = COHERENT_LIKE_OVERLAP_TOL) -> bool:
    """Return True when a state is numerically indistinguishable from the energy-matched coherent reference."""
    if cfg is None:
        return False
    try:
        coh = coherent_coeffs(int(cfg.Ncut), float(cfg.target_energy))
        return coherent_overlap(normalize_coeffs(c), coh) >= float(threshold)
    except Exception:
        return False


def clean_wigner_negativity_value(
    raw_value: float,
    c: Optional[np.ndarray] = None,
    cfg: Optional[UserConfig] = None,
    state_name: str = "",
) -> float:
    """Convert numerical Wigner-negativity artifacts into physical zero for user-facing values."""
    try:
        val = float(raw_value)
    except Exception:
        return 0.0
    if not np.isfinite(val) or val < 0.0:
        return 0.0
    lname = str(state_name).lower()
    coherent_by_name = ("coherent" in lname)
    coherent_by_overlap = c is not None and is_coherent_like_state(c, cfg)
    if coherent_by_name or coherent_by_overlap:
        return 0.0
    if val <= NEGATIVITY_DISPLAY_ZERO_TOL:
        return 0.0
    return val


def clean_wigner_negativity(
    W: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    c: Optional[np.ndarray] = None,
    cfg: Optional[UserConfig] = None,
    state_name: str = "",
) -> float:
    return clean_wigner_negativity_value(wigner_negativity(W, xvec, pvec), c=c, cfg=cfg, state_name=state_name)


def evaluate_state(
    c: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    params: ChannelParams,
) -> Tuple[float, np.ndarray, np.ndarray]:
    c = normalize_coeffs(c)
    W_in = wigner_from_coeffs(c, xvec, pvec)
    W_out = apply_channel(W_in, xvec, pvec, params)
    F = wigner_fidelity(W_in, W_out, xvec, pvec)
    return F, W_in, W_out


def _aggregate_metric(values: Sequence[float], mode: str = "single") -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    if mode == "worst":
        return float(np.min(arr))
    if mode == "average":
        return float(np.mean(arr))
    return float(arr[0])


def evaluate_state_for_channels(
    c: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    channels: Sequence[ChannelParams],
    mode: str = "single",
) -> float:
    c = normalize_coeffs(c)
    W_in = wigner_from_coeffs(c, xvec, pvec)
    fidelities = []
    kernel_cache: Dict[Tuple[float, float, float, int, int, float, float], np.ndarray] = {}
    for params in channels:
        n_xp = float(getattr(params, "noise_xp", 0.0))
        kernel = None
        if params.noise_x2 > 1e-14 or params.noise_p2 > 1e-14 or abs(n_xp) > 1e-14:
            key = channel_kernel_key(params, xvec, pvec)
            kernel = kernel_cache.get(key)
            if kernel is None:
                kernel = gaussian_kernel(xvec, pvec, params.noise_x2, params.noise_p2, n_xp)
                kernel_cache[key] = kernel
        W_out = apply_channel(W_in, xvec, pvec, params, precomputed_kernel=kernel)
        fidelities.append(wigner_fidelity(W_in, W_out, xvec, pvec))
    return _aggregate_metric(fidelities, mode)


def evaluate_objective_components(
    c: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    channels: Sequence[ChannelParams],
    mode: str,
    cfg: UserConfig,
) -> Dict[str, float]:
    """Compute fidelity and Wigner-negativity survival information.

    The input Wigner negativity is

        N_in = 1/2 integral (|W_in| - W_in) dx dp.

    The output negativity is computed after the selected teleportation channel.
    The survival ratio is R = N_out/N_in, but only if N_in is above the user
    threshold. Otherwise R is set to zero and the joint objective penalizes the
    state.
    """
    c = normalize_coeffs(c)
    W_in = wigner_from_coeffs(c, xvec, pvec)
    input_neg = wigner_negativity(W_in, xvec, pvec)

    fidelities: List[float] = []
    output_negs: List[float] = []
    ratios: List[float] = []

    threshold = max(float(cfg.min_input_negativity_for_survival), 1e-300)
    kernel_cache: Dict[Tuple[float, float, float, int, int, float, float], np.ndarray] = {}
    for params in channels:
        n_xp = float(getattr(params, "noise_xp", 0.0))
        kernel = None
        if params.noise_x2 > 1e-14 or params.noise_p2 > 1e-14 or abs(n_xp) > 1e-14:
            key = channel_kernel_key(params, xvec, pvec)
            kernel = kernel_cache.get(key)
            if kernel is None:
                kernel = gaussian_kernel(xvec, pvec, params.noise_x2, params.noise_p2, n_xp)
                kernel_cache[key] = kernel
        W_out = apply_channel(W_in, xvec, pvec, params, precomputed_kernel=kernel)
        F = wigner_fidelity(W_in, W_out, xvec, pvec)
        output_neg = wigner_negativity(W_out, xvec, pvec)
        if input_neg >= threshold:
            ratio = output_neg / input_neg
        else:
            ratio = 0.0
        if not np.isfinite(ratio):
            ratio = 0.0
        fidelities.append(float(F))
        output_negs.append(float(output_neg))
        ratios.append(float(ratio))

    return {
        "objective_fidelity": _aggregate_metric(fidelities, mode),
        "input_negativity": float(input_neg),
        "output_negativity": _aggregate_metric(output_negs, mode),
        "negativity_survival_ratio": _aggregate_metric(ratios, mode),
    }


def objective_score_for_state(
    c: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    channels: Sequence[ChannelParams],
    mode: str,
    cfg: UserConfig,
) -> Tuple[float, Dict[str, float]]:
    # Keep the normal fidelity-only objective fast. Negativity survival requires
    # extra Wigner negativity evaluations, so we only compute them when the
    # user explicitly enables the joint objective.
    if not cfg.use_negativity_survival_objective:
        F = evaluate_state_for_channels(c, xvec, pvec, channels, mode=mode)
        score = selection_score_from_components(F, c, cfg)
        return float(score), {
            "objective_fidelity": float(F),
            "input_negativity": float("nan"),
            "output_negativity": float("nan"),
            "negativity_survival_ratio": float("nan"),
        }

    comps = evaluate_objective_components(c, xvec, pvec, channels, mode, cfg)
    score = selection_score_from_components(
        comps["objective_fidelity"],
        c,
        cfg,
        survival_ratio=comps["negativity_survival_ratio"],
        input_negativity=comps["input_negativity"],
    )
    return float(score), comps


def parse_cutoff_values(text: str, current_ncut: int) -> List[int]:
    """Parse a comma/space separated Ncut list and keep valid unique values."""
    values: List[int] = []
    for part in str(text).replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(float(part))
        except Exception:
            continue
        if n >= 2 and n not in values:
            values.append(n)
    if current_ncut not in values:
        values.append(int(current_ncut))
    return sorted(values)


def fock_support_metrics(c: np.ndarray, cfg: UserConfig) -> Dict[str, float]:
    """Cutoff-sensitivity indicators based only on the photon distribution."""
    c = normalize_coeffs(c)
    probs = np.abs(c) ** 2
    n = np.arange(len(c))

    tail_levels = int(max(1, min(len(c), cfg.tail_levels)))
    tail_start = max(0, len(c) - tail_levels)
    upper_quarter_start = int(np.floor(0.75 * (len(c) - 1))) if len(c) > 1 else 0
    upper_half_start = int(np.floor(0.50 * (len(c) - 1))) if len(c) > 1 else 0

    significant_1e6 = np.where(probs > 1e-6)[0]
    significant_1e4 = np.where(probs > 1e-4)[0]

    return {
        "p_top1": float(probs[-1]) if len(probs) else 0.0,
        "p_tail": float(np.sum(probs[tail_start:])),
        "e_tail": float(np.sum(n[tail_start:] * probs[tail_start:])),
        "p_upper_quarter": float(np.sum(probs[upper_quarter_start:])),
        "e_upper_quarter": float(np.sum(n[upper_quarter_start:] * probs[upper_quarter_start:])),
        "p_upper_half": float(np.sum(probs[upper_half_start:])),
        "e_upper_half": float(np.sum(n[upper_half_start:] * probs[upper_half_start:])),
        "max_prob_n": float(int(np.argmax(probs))) if len(probs) else 0.0,
        "max_n_prob_gt_1e_minus_6": float(int(significant_1e6[-1])) if len(significant_1e6) else 0.0,
        "max_n_prob_gt_1e_minus_4": float(int(significant_1e4[-1])) if len(significant_1e4) else 0.0,
        "inverse_participation_dim": float(1.0 / max(np.sum(probs ** 2), 1e-300)),
        "n2": n2_expectation(c),
        "var_n": photon_variance(c),
    }


def two_level_state(Ncut: int, low: int, high: int, target_energy: float, phase: float) -> Optional[np.ndarray]:
    """Return sqrt(1-w)|low> + exp(i phase)sqrt(w)|high> with fixed energy."""
    if high <= low:
        return None
    if not (low <= target_energy <= high):
        return None
    w_high = (target_energy - low) / (high - low)
    if w_high < -1e-12 or w_high > 1.0 + 1e-12:
        return None
    w_high = float(np.clip(w_high, 0.0, 1.0))
    c = np.zeros(Ncut, dtype=complex)
    c[low] = np.sqrt(1.0 - w_high)
    c[high] = np.exp(1j * phase) * np.sqrt(w_high)
    return normalize_coeffs(c)


def escape_sequence_scan(
    cfg: UserConfig,
    xvec: np.ndarray,
    pvec: np.ndarray,
    channels: Sequence[ChannelParams],
    objective_mode: str,
) -> Dict[str, object]:
    """Scan simple two-level energy-hiding states.

    This is an analytical/numerical test for the noncompact fixed-energy issue:
        sqrt(1-N/M)|0> + sqrt(N/M)|M>.
    If the best member occurs near the cutoff and approaches the optimized
    fidelity, the result may be a cutoff-dependent maximizing sequence.
    """
    active = active_fock_indices(cfg)
    lows = [int(x) for x in active if x <= cfg.target_energy + 1e-12]
    highs = [int(x) for x in active if x >= cfg.target_energy - 1e-12]
    if not lows or not highs:
        return {"available": False, "reason": "No two-level pair brackets the target energy."}

    # Prefer the lowest allowed level because that is the canonical escape sequence.
    # Also include the closest lower level, which matters for parity/modular sectors.
    low_candidates = sorted(set([min(lows), max(lows)]))
    phases = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]

    series: List[Dict[str, float]] = []
    best: Optional[Dict[str, object]] = None
    for low in low_candidates:
        for high in highs:
            if high <= low:
                continue
            best_for_high: Optional[Dict[str, object]] = None
            for phase in phases:
                c = two_level_state(cfg.Ncut, low, high, cfg.target_energy, phase)
                if c is None:
                    continue
                try:
                    score, comps = objective_score_for_state(c, xvec, pvec, channels, objective_mode, cfg)
                    F_obj = comps["objective_fidelity"]
                except Exception:
                    continue
                candidate = {
                    "low": float(low),
                    "high": float(high),
                    "phase": float(phase),
                    "objective_fidelity": float(F_obj),
                    "selection_score": float(score),
                    "tail_probability": tail_probability(c, cfg.tail_levels),
                    "var_n": photon_variance(c),
                }
                if best_for_high is None or candidate["selection_score"] > best_for_high["selection_score"]:
                    best_for_high = candidate
                if best is None or candidate["selection_score"] > float(best["selection_score"]):
                    best = {**candidate, "coeffs": c}
            if best_for_high is not None:
                series.append(best_for_high)

    if best is None:
        return {"available": False, "reason": "No two-level state could be evaluated."}

    # Estimate whether the escape family improves toward the cutoff.
    series_sorted = sorted(series, key=lambda row: row["high"])
    last = series_sorted[-min(5, len(series_sorted)):]
    if len(last) >= 2:
        xh = np.array([row["high"] for row in last], dtype=float)
        yh = np.array([row["selection_score"] for row in last], dtype=float)
        slope = float(np.polyfit(xh, yh, 1)[0]) if np.std(xh) > 0 else 0.0
    else:
        slope = 0.0

    coeffs = best.pop("coeffs")
    return {
        "available": True,
        "best": best,
        "best_coeffs": coeffs,
        "series": series_sorted,
        "last_window_score_slope": slope,
        "best_high_fraction": float(best["high"] / max(1, cfg.Ncut - 1)),
    }


def has_compactifying_constraint(cfg: UserConfig) -> bool:
    """Whether active constraints give a reasonable infinite-dimensional compactness proxy."""
    return bool(
        cfg.use_fock_range_constraint
        or cfg.use_photon_variance_max
        or cfg.use_mandel_q_max
        or cfg.use_fluctuation_energy_max
        or cfg.use_x_variance_max
        or cfg.use_p_variance_max
        or cfg.use_bounded_displacement
        or cfg.use_fixed_displacement
    )


def headless_optimize_once(cfg: UserConfig, stop_event: Optional[threading.Event] = None) -> Dict[str, object]:
    """Small non-GUI optimizer used by the cutoff scan and local probe."""
    active = active_fock_indices(cfg)
    xvec, pvec = make_grid(cfg)
    base_params = build_channel(cfg)
    channels = robust_channels(cfg, base_params)
    if cfg.use_robust_worstcase_objective:
        objective_mode = "worst"
    elif cfg.use_robust_average_objective:
        objective_mode = "average"
    else:
        objective_mode = "single"

    constraints, unpack = build_constraints(cfg, active, xvec, pvec)
    seeds = make_seeds(cfg, active)
    n_active = len(active)
    n_vars = n_active if cfg.use_real_coefficients else 2 * n_active
    cache: Dict[bytes, float] = {}
    best: Optional[Dict[str, object]] = None

    def objective(y: np.ndarray) -> float:
        if stop_event is not None and stop_event.is_set():
            raise StopOptimization()
        key = np.asarray(y).round(11).tobytes()
        if key in cache:
            return cache[key]
        c = unpack(y, normalize=True)
        score, _ = objective_score_for_state(c, xvec, pvec, channels, objective_mode, cfg)
        cache[key] = -score
        return -score

    for c0 in seeds:
        y0 = active_to_y(c0[active], real_only=cfg.use_real_coefficients)
        try:
            result = minimize(
                objective,
                y0,
                method="SLSQP",
                constraints=constraints,
                bounds=[(-2.0, 2.0)] * n_vars,
                options={"maxiter": cfg.maxiter, "ftol": cfg.ftol, "disp": False},
            )
        except StopOptimization:
            raise
        except Exception:
            continue

        c = unpack(result.x, normalize=True)
        feas = feasibility_diagnostics(c, cfg, xvec, pvec)
        F_base, W_in, W_out = evaluate_state(c, xvec, pvec, base_params)
        score, comps = objective_score_for_state(c, xvec, pvec, channels, objective_mode, cfg)
        F_obj = comps["objective_fidelity"]
        d = state_diagnostics(c)
        item = {
            "coeffs": c,
            "fidelity": F_base,
            "objective_fidelity": F_obj,
            "selection_score": score,
            "energy": d["energy"],
            "a": d["a"],
            "parity": d["parity"],
            "n_variance": d["n_variance"],
            "tail_probability": tail_probability(c, cfg.tail_levels),
            "W_in": W_in,
            "W_out": W_out,
            "success": bool(result.success),
            "message": str(result.message),
            "feasible": bool(feas["is_feasible"]),
            "constraint_violation": float(feas["max_violation"]),
            "violated_constraints": list(feas.get("violated", [])),
        }
        if not item["feasible"]:
            continue
        if best is None or float(item["selection_score"]) > float(best["selection_score"]):
            best = item

    if best is None:
        raise RuntimeError("Headless optimization did not produce a feasible result.")
    best["xvec"] = xvec
    best["pvec"] = pvec
    best["params"] = base_params
    best["active"] = active
    best["objective_mode"] = objective_mode
    return best


def local_optimality_probe(
    best: Dict[str, object],
    cfg: UserConfig,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, object]:
    """Short local restarts from perturbed final state.

    This does not prove global optimality. It checks whether SLSQP can easily
    improve the result in the same finite-dimensional constrained problem.
    """
    rng = np.random.default_rng(cfg.random_seed + 2027)
    active = np.asarray(best.get("active", active_fock_indices(cfg)), dtype=int)
    c_best = normalize_coeffs(np.asarray(best["coeffs"]))
    xvec = np.asarray(best["xvec"])
    pvec = np.asarray(best["pvec"])
    base_params = best["params"]
    channels = best.get("objective_channels", robust_channels(cfg, base_params))
    objective_mode = str(best.get("objective_mode", "single"))
    constraints, unpack = build_constraints(cfg, active, xvec, pvec)
    n_active = len(active)
    n_vars = n_active if cfg.use_real_coefficients else 2 * n_active

    base_score = float(best.get("selection_score", best.get("objective_fidelity", best["fidelity"])))
    improvements: List[float] = []
    successes = 0
    best_probe_score = base_score

    def objective(y: np.ndarray) -> float:
        if stop_event is not None and stop_event.is_set():
            raise StopOptimization()
        c = unpack(y, normalize=True)
        score, _ = objective_score_for_state(c, xvec, pvec, channels, objective_mode, cfg)
        return -score

    for _ in range(max(0, int(cfg.local_probe_trials))):
        if cfg.use_real_coefficients:
            noise = rng.normal(size=n_active)
            y0 = active_to_y(c_best[active], real_only=True) + cfg.local_probe_perturbation * noise
        else:
            noise = rng.normal(size=2 * n_active)
            y0 = active_to_y(c_best[active], real_only=False) + cfg.local_probe_perturbation * noise
        try:
            result = minimize(
                objective,
                y0,
                method="SLSQP",
                constraints=constraints,
                bounds=[(-2.0, 2.0)] * n_vars,
                options={"maxiter": cfg.local_probe_maxiter, "ftol": cfg.ftol, "disp": False},
            )
            c_probe = unpack(result.x, normalize=True)
            score_probe, comps_probe = objective_score_for_state(c_probe, xvec, pvec, channels, objective_mode, cfg)
            F_probe = comps_probe["objective_fidelity"]
            improvement = float(score_probe - base_score)
            improvements.append(improvement)
            if result.success:
                successes += 1
            if score_probe > best_probe_score:
                best_probe_score = float(score_probe)
        except StopOptimization:
            raise
        except Exception:
            continue

    if improvements:
        max_improvement = float(np.max(improvements))
        median_improvement = float(np.median(improvements))
    else:
        max_improvement = 0.0
        median_improvement = 0.0

    return {
        "trials": int(cfg.local_probe_trials),
        "completed": len(improvements),
        "successes": successes,
        "base_score": base_score,
        "best_probe_score": best_probe_score,
        "max_improvement": max_improvement,
        "median_improvement": median_improvement,
        "locally_stable": bool(max_improvement < max(1e-6, 10.0 * cfg.ftol)),
    }


def cutoff_convergence_scan(
    cfg: UserConfig,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, object]:
    """Run a reduced optimization over several Ncut values."""
    values = parse_cutoff_values(cfg.cutoff_scan_values, cfg.Ncut)
    rows: List[Dict[str, object]] = []
    for ncut in values:
        if stop_event is not None and stop_event.is_set():
            raise StopOptimization()
        if cfg.target_energy > ncut - 1:
            continue
        scan_cfg = replace(
            cfg,
            Ncut=int(ncut),
            n_starts=max(1, int(cfg.cutoff_scan_n_starts)),
            maxiter=max(1, int(cfg.cutoff_scan_maxiter)),
            use_best_polishing=False,
            run_cutoff_scan=False,
            run_local_optimality_probe=False,
        )
        try:
            result = headless_optimize_once(scan_cfg, stop_event=stop_event)
            support = fock_support_metrics(result["coeffs"], scan_cfg)
            rows.append({
                "Ncut": int(ncut),
                "fidelity": float(result["fidelity"]),
                "objective_fidelity": float(result["objective_fidelity"]),
                "selection_score": float(result["selection_score"]),
                "energy": float(result["energy"]),
                "var_n": float(result["n_variance"]),
                "tail_probability": float(result["tail_probability"]),
                "max_n_gt_1e_minus_4": support["max_n_prob_gt_1e_minus_4"],
                "max_n_gt_1e_minus_6": support["max_n_prob_gt_1e_minus_6"],
                "success": bool(result["success"]),
                "feasible": bool(result.get("feasible", True)),
                "constraint_violation": float(result.get("constraint_violation", 0.0)),
                "message": str(result["message"]),
            })
        except Exception as exc:
            rows.append({"Ncut": int(ncut), "error": str(exc)})

    valid = [row for row in rows if "selection_score" in row]
    verdict = "not enough valid cutoff-scan points"
    last_range = None
    if len(valid) >= 3:
        last = valid[-3:]
        vals = np.array([float(row["selection_score"]) for row in last])
        last_range = float(np.max(vals) - np.min(vals))
        maxn_fracs = [float(row["max_n_gt_1e_minus_4"]) / max(1.0, float(row["Ncut"] - 1)) for row in last]
        if last_range < 1e-3 and max(maxn_fracs) < 0.75:
            verdict = "likely converged over the scanned cutoffs"
        elif last_range < 1e-3 and max(maxn_fracs) >= 0.75:
            verdict = "fidelity nearly converged, but state still lives near the cutoff"
        else:
            verdict = "not converged over the scanned cutoffs"

    return {"rows": rows, "valid_count": len(valid), "last_three_score_range": last_range, "verdict": verdict}


def convergence_diagnostics(
    best: Dict[str, object],
    cfg: UserConfig,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, object]:
    """Collect analytical and numerical convergence diagnostics for final report."""
    c = normalize_coeffs(np.asarray(best["coeffs"]))
    xvec = np.asarray(best["xvec"])
    pvec = np.asarray(best["pvec"])
    channels = best.get("objective_channels", [best["params"]])
    objective_mode = str(best.get("objective_mode", "single"))
    support = fock_support_metrics(c, cfg)
    compact_proxy = has_compactifying_constraint(cfg)

    escape = {"available": False, "reason": "disabled"}
    if cfg.run_analytic_convergence_diagnostics:
        escape = escape_sequence_scan(cfg, xvec, pvec, channels, objective_mode)

    local_probe = None
    if cfg.run_local_optimality_probe:
        local_probe = local_optimality_probe(best, cfg, stop_event=stop_event)

    cutoff_scan = None
    if cfg.run_cutoff_scan:
        cutoff_scan = cutoff_convergence_scan(cfg, stop_event=stop_event)

    score = float(best.get("selection_score", best.get("objective_fidelity", best["fidelity"])))
    warnings_list: List[str] = []
    if not compact_proxy:
        warnings_list.append(
            "Only fixed mean energy does not make the infinite-dimensional pure-state search compact; a cutoff-dependent maximizing sequence is possible."
        )
    if support["var_n"] > 10.0 * max(1.0, cfg.target_energy):
        warnings_list.append("Photon-number variance is very large compared with the fixed mean energy.")
    if support["max_n_prob_gt_1e_minus_4"] > 0.70 * max(1, cfg.Ncut - 1):
        warnings_list.append("Significant photon-number weight extends close to the artificial cutoff.")
    if cfg.use_tail_probability_constraint and support["p_tail"] > 0.80 * max(cfg.max_tail_probability, 1e-300):
        warnings_list.append("The high-Fock tail constraint is active or nearly active.")
    if support["e_upper_quarter"] > 0.10 * max(1.0, cfg.target_energy):
        warnings_list.append("A non-negligible part of the energy is stored in the upper quarter of the cutoff.")

    if escape.get("available"):
        esc_best = escape["best"]
        esc_gap = score - float(esc_best["selection_score"])
        if esc_gap < 1e-3 and float(escape.get("best_high_fraction", 0.0)) > 0.70:
            warnings_list.append("A simple two-level escape state nearly matches the optimized score near the cutoff.")
        if float(escape.get("last_window_score_slope", 0.0)) > 1e-5:
            warnings_list.append("The two-level escape-family score is still increasing near the largest scanned Fock level.")

    if cutoff_scan is not None and cutoff_scan.get("valid_count", 0) >= 3:
        if "not converged" in str(cutoff_scan.get("verdict", "")):
            warnings_list.append("The automatic Ncut scan did not converge.")
        elif "near the cutoff" in str(cutoff_scan.get("verdict", "")):
            warnings_list.append("The Ncut scan shows near-converged fidelity but cutoff-sensitive states.")

    if not warnings_list:
        verdict = "PASS: no strong cutoff-instability indicator was found."
    elif compact_proxy and not any("cutoff" in w.lower() for w in warnings_list):
        verdict = "CAUTION: finite-dimensional result is locally plausible, but some moment diagnostics are large."
    else:
        verdict = "CAUTION: result may be a cutoff-dependent maximizing sequence rather than a stable infinite-dimensional optimum."

    return {
        "support": support,
        "has_compactifying_constraint": compact_proxy,
        "escape_scan": escape,
        "local_probe": local_probe,
        "cutoff_scan": cutoff_scan,
        "warnings": warnings_list,
        "verdict": verdict,
    }


# =============================================================================
# Seeds and reference states
# =============================================================================

def coherent_coeffs(Ncut: int, target_energy: float) -> np.ndarray:
    if target_energy <= 0.0:
        c = np.zeros(Ncut, dtype=complex)
        c[0] = 1.0
        return c

    def truncated_energy(alpha: float) -> float:
        ket = qt.coherent(Ncut, alpha)
        c = np.asarray(ket.full()).flatten()
        return mean_energy(c)

    try:
        alpha = brentq(
            lambda a: truncated_energy(a) - target_energy,
            0.0,
            max(3.0 * np.sqrt(Ncut), 3.0 * np.sqrt(target_energy + 1.0)),
            maxiter=100,
        )
    except Exception:
        alpha = np.sqrt(target_energy)

    ket = qt.coherent(Ncut, alpha)
    return normalize_coeffs(np.asarray(ket.full()).flatten())


def squeezed_vacuum_coeffs(Ncut: int, target_energy: float) -> np.ndarray:
    if target_energy <= 0.0:
        c = np.zeros(Ncut, dtype=complex)
        c[0] = 1.0
        return c

    def squeezed_energy(s: float) -> float:
        ket = qt.squeeze(Ncut, s) * qt.basis(Ncut, 0)
        c = np.asarray(ket.full()).flatten()
        return mean_energy(c)

    try:
        s = brentq(
            lambda x: squeezed_energy(x) - target_energy,
            0.0,
            max(3.0, 3.0 * np.arcsinh(np.sqrt(target_energy)) + 1.0),
            maxiter=100,
        )
    except Exception:
        s = np.arcsinh(np.sqrt(target_energy))

    ket = qt.squeeze(Ncut, s) * qt.basis(Ncut, 0)
    return normalize_coeffs(np.asarray(ket.full()).flatten())


def fock_coeffs(Ncut: int, n: int) -> np.ndarray:
    c = np.zeros(Ncut, dtype=complex)
    c[int(np.clip(n, 0, Ncut - 1))] = 1.0
    return c


def two_fock_seed(
    Ncut: int,
    target_energy: float,
    allowed: Sequence[int],
    rng: np.random.Generator,
    zero_displacement: bool,
    deterministic: bool,
) -> np.ndarray:
    allowed = np.array(sorted(set(int(x) for x in allowed)), dtype=int)

    # Important: only deterministic seeds collapse to exact |N> for integer energy.
    # Random starts should still explore nontrivial two-Fock superpositions.
    if deterministic:
        for n in allowed:
            if abs(target_energy - n) < 1e-12:
                return fock_coeffs(Ncut, int(n))

    pairs = []
    for low in allowed:
        for high in allowed:
            if high <= low:
                continue
            if low <= target_energy <= high:
                if zero_displacement and abs(high - low) == 1:
                    continue
                pairs.append((int(low), int(high)))

    if not pairs:
        # Last fallback: exact Fock state if possible.
        for n in allowed:
            if abs(target_energy - n) < 1e-12:
                return fock_coeffs(Ncut, int(n))
        raise ValueError("Could not construct a feasible two-Fock seed.")

    if deterministic:
        low, high = sorted(pairs, key=lambda pair: pair[1] - pair[0])[0]
        phi = 0.0
    else:
        low, high = pairs[int(rng.integers(0, len(pairs)))]
        phi = float(rng.uniform(0.0, 2.0 * np.pi))

    w_high = (target_energy - low) / (high - low)
    w_high = float(np.clip(w_high, 0.0, 1.0))

    c = np.zeros(Ncut, dtype=complex)
    c[low] = np.sqrt(1.0 - w_high)
    c[high] = np.exp(1j * phi) * np.sqrt(w_high)
    return normalize_coeffs(c)


def random_fixed_energy_seed(
    Ncut: int,
    target_energy: float,
    allowed: Sequence[int],
    rng: np.random.Generator,
    real_only: bool = False,
) -> np.ndarray:
    """Random full-subspace seed with exact normalization and exact mean energy.

    We sample random positive weights and then exponentially tilt them,

        p_n ∝ w_n exp(beta n),

    with beta chosen so that sum_n n p_n = target_energy. This gives much more
    diverse starts than two-Fock seeds and greatly improves high-dimensional
    searches.
    """
    allowed = np.array(sorted(set(int(x) for x in allowed)), dtype=int)

    if len(allowed) == 0:
        raise ValueError("No allowed Fock states are available.")

    if target_energy < allowed[0] - 1e-12 or target_energy > allowed[-1] + 1e-12:
        raise ValueError("Target energy is outside the allowed Fock sector.")

    weights = rng.exponential(scale=1.0, size=len(allowed)) + 1e-12

    def probabilities(beta: float) -> np.ndarray:
        z = np.log(weights) + beta * allowed
        z = z - np.max(z)
        p = np.exp(z)
        return p / np.sum(p)

    def mean_for_beta(beta: float) -> float:
        p = probabilities(beta)
        return float(np.sum(allowed * p))

    try:
        beta_star = brentq(lambda beta: mean_for_beta(beta) - target_energy, -100.0, 100.0, maxiter=100)
        probs = probabilities(beta_star)
    except Exception:
        return two_fock_seed(Ncut, target_energy, allowed, rng, zero_displacement=False, deterministic=False)

    phases = np.zeros(len(allowed)) if real_only else rng.uniform(0.0, 2.0 * np.pi, size=len(allowed))
    c = np.zeros(Ncut, dtype=complex)
    c[allowed] = np.sqrt(probs) * np.exp(1j * phases)
    return normalize_coeffs(c)


def apply_subspace_to_seed(c: np.ndarray, active: np.ndarray, Ncut: int) -> Optional[np.ndarray]:
    out = np.zeros(Ncut, dtype=complex)
    out[active] = c[active]
    if np.vdot(out, out).real < 1e-14:
        return None
    return normalize_coeffs(out)


def make_seeds(cfg: UserConfig, active: np.ndarray) -> List[np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    seeds: List[np.ndarray] = []

    coherent_allowed = (
        not cfg.use_zero_displacement_constraint
        and not cfg.use_even_parity_constraint
        and not cfg.use_odd_parity_constraint
        and not cfg.use_modular_fock_constraint
        and not cfg.use_real_coefficients
    )
    if coherent_allowed:
        c = coherent_coeffs(cfg.Ncut, cfg.target_energy)
        c = apply_subspace_to_seed(c, active, cfg.Ncut)
        if c is not None:
            seeds.append(c)

    if not cfg.use_odd_parity_constraint and not cfg.use_modular_fock_constraint:
        try:
            c = squeezed_vacuum_coeffs(cfg.Ncut, cfg.target_energy)
            c = apply_subspace_to_seed(c, active, cfg.Ncut)
            if c is not None:
                seeds.append(c)
        except Exception:
            pass

    try:
        seeds.append(
            two_fock_seed(
                cfg.Ncut,
                cfg.target_energy,
                active,
                rng,
                zero_displacement=cfg.use_zero_displacement_constraint,
                deterministic=True,
            )
        )
    except Exception:
        pass

    # Add broad random seeds with exact <n>=target_energy. These are much better
    # than only two-Fock seeds when Ncut is large. For a pure zero-displacement
    # constraint without parity, broad random seeds are usually infeasible, so we
    # skip them there.
    broad_allowed = (
        cfg.use_broad_random_seeds
        and not cfg.use_fixed_displacement
        and not (
            cfg.use_zero_displacement_constraint
            and not (cfg.use_even_parity_constraint or cfg.use_odd_parity_constraint or cfg.use_modular_fock_constraint)
        )
    )

    attempts = 0
    while len(seeds) < cfg.n_starts and attempts < 10 * max(1, cfg.n_starts):
        attempts += 1
        try:
            if broad_allowed and attempts % 2 == 1:
                seeds.append(
                    random_fixed_energy_seed(
                        cfg.Ncut,
                        cfg.target_energy,
                        active,
                        rng,
                        real_only=cfg.use_real_coefficients,
                    )
                )
            else:
                seeds.append(
                    two_fock_seed(
                        cfg.Ncut,
                        cfg.target_energy,
                        active,
                        rng,
                        zero_displacement=cfg.use_zero_displacement_constraint,
                        deterministic=False,
                    )
                )
        except Exception:
            continue

    unique: List[np.ndarray] = []
    for c in seeds:
        c = normalize_coeffs(c)
        if cfg.use_real_coefficients:
            c = normalize_coeffs(c.real.astype(complex))
        if not any(abs(np.vdot(c, u)) > 1.0 - 1e-10 for u in unique):
            unique.append(c)

    if not unique:
        raise RuntimeError("No feasible initial states were created. Try relaxing constraints or increasing Ncut.")

    return unique


def reference_states(cfg: UserConfig) -> Dict[str, np.ndarray]:
    refs: Dict[str, np.ndarray] = {}
    refs["coherent"] = coherent_coeffs(cfg.Ncut, cfg.target_energy)

    try:
        refs["squeezed vacuum"] = squeezed_vacuum_coeffs(cfg.Ncut, cfg.target_energy)
    except Exception:
        pass

    nearest = int(round(cfg.target_energy))
    if 0 <= nearest < cfg.Ncut:
        refs[f"nearest Fock |{nearest}>"] = fock_coeffs(cfg.Ncut, nearest)

    try:
        refs["cat reference"] = cat_coeffs(cfg.Ncut, cfg.cat_alpha, cfg.cat_phase)
    except Exception:
        pass

    try:
        refs["feasible two-Fock"] = two_fock_seed(
            cfg.Ncut,
            cfg.target_energy,
            active_fock_indices(cfg),
            np.random.default_rng(cfg.random_seed + 99),
            zero_displacement=cfg.use_zero_displacement_constraint,
            deterministic=True,
        )
    except Exception:
        pass

    return refs


def evaluate_references(
    cfg: UserConfig,
    xvec: np.ndarray,
    pvec: np.ndarray,
    params: ChannelParams,
) -> List[Dict[str, object]]:
    rows = []
    for name, c in reference_states(cfg).items():
        F, _, _ = evaluate_state(c, xvec, pvec, params)
        d = state_diagnostics(c)
        # Use the same complete constraint checker used for selecting the final best state.
        # This is important because state_allowed() only checks the older/basic constraints
        # and does not know about the high-Fock tail constraint, Wigner negativity, etc.
        try:
            feas = feasibility_diagnostics(c, cfg, xvec=xvec, pvec=pvec)
            allowed = bool(feas["is_feasible"])
            max_violation = float(feas["max_violation"])
            violated = list(feas["violated"])
        except Exception:
            allowed = state_allowed(c, cfg)
            max_violation = float("nan")
            violated = []
        rows.append(
            {
                "state": name,
                "fidelity": F,
                "energy": d["energy"],
                "a": d["a"],
                "parity": d["parity"],
                "n_variance": d["n_variance"],
                "mandel_q": d["mandel_q"],
                "x_var": d["x_var"],
                "p_var": d["p_var"],
                "displacement_squared": d["displacement_squared"],
                "tail_probability": tail_probability(c, cfg.tail_levels),
                "allowed": allowed,
                "max_violation": max_violation,
                "violated_constraints": violated,
            }
        )
    rows.sort(key=lambda row: float(row["fidelity"]), reverse=True)
    return rows


# =============================================================================
# Constraint builder
# =============================================================================

def build_constraints(
    cfg: UserConfig,
    active: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
):
    n_active = len(active)
    n_full = np.arange(cfg.Ncut)

    coh_ref = coherent_coeffs(cfg.Ncut, cfg.target_energy)
    try:
        sq_ref = squeezed_vacuum_coeffs(cfg.Ncut, cfg.target_energy)
    except Exception:
        sq_ref = None

    try:
        cat_ref = cat_coeffs(cfg.Ncut, cfg.cat_alpha, cfg.cat_phase)
    except Exception:
        cat_ref = None

    def unpack(y: np.ndarray, normalize: bool = True) -> np.ndarray:
        return active_to_full(
            y_to_active(y, n_active, normalize=normalize, real_only=cfg.use_real_coefficients),
            active,
            cfg.Ncut,
        )

    def norm_constraint(y: np.ndarray) -> float:
        c = unpack(y, normalize=False)
        return float(np.vdot(c, c).real - 1.0)

    def energy_constraint(y: np.ndarray) -> float:
        c = unpack(y, normalize=False)
        return float(np.sum(n_full * np.abs(c) ** 2).real - cfg.target_energy)

    constraints = [
        {"type": "eq", "fun": norm_constraint},
        {"type": "eq", "fun": energy_constraint},
    ]

    # Equality constraints
    if cfg.use_fixed_displacement:
        constraints.append({
            "type": "eq",
            "fun": lambda y: float(a_expectation(unpack(y)).real - cfg.fixed_a_real),
        })
        constraints.append({
            "type": "eq",
            "fun": lambda y: float(a_expectation(unpack(y)).imag - cfg.fixed_a_imag),
        })
    elif cfg.use_zero_displacement_constraint and not (
        cfg.use_even_parity_constraint or cfg.use_odd_parity_constraint
    ):
        constraints.append({"type": "eq", "fun": lambda y: float(a_expectation(unpack(y)).real)})
        constraints.append({"type": "eq", "fun": lambda y: float(a_expectation(unpack(y)).imag)})

    if cfg.use_real_displacement_axis:
        constraints.append({"type": "eq", "fun": lambda y: float(a_expectation(unpack(y)).imag)})

    # Inequality constraints. SLSQP expects fun(y) >= 0.
    if cfg.use_bounded_displacement:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_displacement_squared - abs(a_expectation(unpack(y))) ** 2),
        })

    if cfg.use_fluctuation_energy_min:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(
                mean_energy(unpack(y)) - abs(a_expectation(unpack(y))) ** 2 - cfg.min_fluctuation_energy
            ),
        })

    if cfg.use_fluctuation_energy_max:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(
                cfg.max_fluctuation_energy - (mean_energy(unpack(y)) - abs(a_expectation(unpack(y))) ** 2)
            ),
        })

    if cfg.use_photon_variance_min:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(photon_variance(unpack(y)) - cfg.min_photon_variance),
        })

    if cfg.use_photon_variance_max:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_photon_variance - photon_variance(unpack(y))),
        })

    if cfg.use_mandel_q_min:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(mandel_q(unpack(y)) - cfg.min_mandel_q),
        })

    if cfg.use_mandel_q_max:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_mandel_q - mandel_q(unpack(y))),
        })

    if cfg.use_x_variance_min:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(quadrature_moments(unpack(y))["x_var"] - cfg.min_x_variance),
        })

    if cfg.use_x_variance_max:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_x_variance - quadrature_moments(unpack(y))["x_var"]),
        })

    if cfg.use_p_variance_min:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(quadrature_moments(unpack(y))["p_var"] - cfg.min_p_variance),
        })

    if cfg.use_p_variance_max:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_p_variance - quadrature_moments(unpack(y))["p_var"]),
        })

    if cfg.use_covariance_min:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(quadrature_moments(unpack(y))["xp_cov"] - cfg.min_xp_covariance),
        })

    if cfg.use_covariance_max:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_xp_covariance - quadrature_moments(unpack(y))["xp_cov"]),
        })

    if cfg.use_coherent_overlap_max:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_coherent_overlap - coherent_overlap(unpack(y), coh_ref)),
        })

    if cfg.use_squeezed_overlap_max and sq_ref is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_squeezed_overlap - squeezed_overlap(unpack(y), sq_ref)),
        })

    if cfg.use_cat_overlap_min and cat_ref is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cat_overlap(unpack(y), cat_ref) - cfg.min_cat_overlap),
        })

    if cfg.use_tail_probability_constraint:
        constraints.append({
            "type": "ineq",
            "fun": lambda y: float(cfg.max_tail_probability - tail_probability(unpack(y), cfg.tail_levels)),
        })

    if cfg.use_wigner_negativity_min:
        def negativity_min_constraint(y: np.ndarray) -> float:
            c = unpack(y)
            W = wigner_from_coeffs(c, xvec, pvec)
            return float(wigner_negativity(W, xvec, pvec) - cfg.min_wigner_negativity)
        constraints.append({"type": "ineq", "fun": negativity_min_constraint})

    if cfg.use_negativity_survival_objective:
        def survival_input_negativity_constraint(y: np.ndarray) -> float:
            c = unpack(y)
            W = wigner_from_coeffs(c, xvec, pvec)
            return float(wigner_negativity(W, xvec, pvec) - cfg.min_input_negativity_for_survival)
        constraints.append({"type": "ineq", "fun": survival_input_negativity_constraint})

    if cfg.use_wigner_negativity_max:
        def negativity_max_constraint(y: np.ndarray) -> float:
            c = unpack(y)
            W = wigner_from_coeffs(c, xvec, pvec)
            return float(cfg.max_wigner_negativity - wigner_negativity(W, xvec, pvec))
        constraints.append({"type": "ineq", "fun": negativity_max_constraint})

    return constraints, unpack


def tail_energy_feasibility_bounds(cfg: UserConfig, active: np.ndarray) -> Tuple[float, float]:
    """Analytical energy bounds implied by the active Fock subspace and tail constraint."""
    active = np.asarray(active, dtype=int)
    if len(active) == 0:
        raise ValueError("No active Fock states are available.")
    if not cfg.use_tail_probability_constraint:
        return float(np.min(active)), float(np.max(active))
    tail_start = max(0, cfg.Ncut - int(max(1, cfg.tail_levels)))
    tail = active[active >= tail_start]
    non_tail = active[active < tail_start]
    pmax = float(np.clip(cfg.max_tail_probability, 0.0, 1.0))
    if len(non_tail) == 0:
        if pmax < 1.0 - 1e-12:
            raise ValueError("The active Fock subspace lies entirely inside the high-tail window, but P_tail is constrained below 1. This makes normalization impossible.")
        return float(np.min(active)), float(np.max(active))
    min_energy = float(np.min(non_tail))
    max_non_tail = float(np.max(non_tail))
    max_energy = max_non_tail if len(tail) == 0 else (1.0 - pmax) * max_non_tail + pmax * float(np.max(tail))
    return min_energy, max_energy


def feasibility_diagnostics(c: np.ndarray, cfg: UserConfig, xvec: Optional[np.ndarray] = None, pvec: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Check active constraints for a candidate state; used to decide the true final best state."""
    c = normalize_coeffs(c)
    d = state_diagnostics(c)
    violations: List[Tuple[str, float]] = []
    energy_tol = max(1e-6, 1e-6 * max(1.0, abs(float(cfg.target_energy))))
    def eq(name, value, target, tol=1e-6):
        violations.append((name, max(0.0, abs(float(value) - float(target)) - tol)))
    def le(name, value, upper, tol=1e-8):
        violations.append((name, max(0.0, float(value) - float(upper) - tol)))
    def ge(name, value, lower, tol=1e-8):
        violations.append((name, max(0.0, float(lower) - float(value) - tol)))
    eq("energy", float(d["energy"]), float(cfg.target_energy), energy_tol)
    if cfg.use_tail_probability_constraint:
        le("high_fock_tail", tail_probability(c, cfg.tail_levels), cfg.max_tail_probability)
    if cfg.use_even_parity_constraint:
        le("even_parity_subspace", float(np.sum(np.abs(c[1::2]) ** 2)), 0.0)
    if cfg.use_odd_parity_constraint:
        le("odd_parity_subspace", float(np.sum(np.abs(c[0::2]) ** 2)), 0.0)
    if cfg.use_real_coefficients:
        le("real_coefficients", float(np.max(np.abs(c.imag))), 0.0)
    if cfg.use_fock_range_constraint:
        mask = np.ones(len(c), dtype=bool); lo = max(0, int(cfg.fock_min)); hi = min(len(c)-1, int(cfg.fock_max)); mask[lo:hi+1] = False
        le("fock_range", float(np.sum(np.abs(c[mask]) ** 2)), 0.0)
    if cfg.use_modular_fock_constraint:
        m = int(cfg.modulus_m); k = int(cfg.residue_k) % m; n = np.arange(len(c))
        le("modular_fock_sector", float(np.sum(np.abs(c[n % m != k]) ** 2)), 0.0)
    a = complex(d["a"])
    if cfg.use_fixed_displacement:
        eq("fixed_Re_a", a.real, cfg.fixed_a_real); eq("fixed_Im_a", a.imag, cfg.fixed_a_imag)
    elif cfg.use_zero_displacement_constraint:
        le("zero_displacement", abs(a), 0.0, 1e-6)
    if cfg.use_real_displacement_axis:
        le("real_displacement_axis", abs(a.imag), 0.0, 1e-6)
    if cfg.use_bounded_displacement:
        le("bounded_displacement", float(d["displacement_squared"]), cfg.max_displacement_squared)
    if cfg.use_fluctuation_energy_min:
        ge("fluctuation_energy_min", float(d["fluctuation_energy"]), cfg.min_fluctuation_energy)
    if cfg.use_fluctuation_energy_max:
        le("fluctuation_energy_max", float(d["fluctuation_energy"]), cfg.max_fluctuation_energy)
    if cfg.use_photon_variance_min:
        ge("photon_variance_min", float(d["n_variance"]), cfg.min_photon_variance)
    if cfg.use_photon_variance_max:
        le("photon_variance_max", float(d["n_variance"]), cfg.max_photon_variance)
    if cfg.use_mandel_q_min:
        ge("mandel_q_min", float(d["mandel_q"]), cfg.min_mandel_q)
    if cfg.use_mandel_q_max:
        le("mandel_q_max", float(d["mandel_q"]), cfg.max_mandel_q)
    if cfg.use_x_variance_min:
        ge("x_variance_min", float(d["x_var"]), cfg.min_x_variance)
    if cfg.use_x_variance_max:
        le("x_variance_max", float(d["x_var"]), cfg.max_x_variance)
    if cfg.use_p_variance_min:
        ge("p_variance_min", float(d["p_var"]), cfg.min_p_variance)
    if cfg.use_p_variance_max:
        le("p_variance_max", float(d["p_var"]), cfg.max_p_variance)
    if cfg.use_covariance_min:
        ge("covariance_min", float(d["xp_cov"]), cfg.min_xp_covariance)
    if cfg.use_covariance_max:
        le("covariance_max", float(d["xp_cov"]), cfg.max_xp_covariance)
    if cfg.use_coherent_overlap_max:
        le("coherent_overlap_max", coherent_overlap(c, coherent_coeffs(cfg.Ncut, cfg.target_energy)), cfg.max_coherent_overlap)
    if cfg.use_squeezed_overlap_max:
        try: le("squeezed_overlap_max", squeezed_overlap(c, squeezed_vacuum_coeffs(cfg.Ncut, cfg.target_energy)), cfg.max_squeezed_overlap)
        except Exception: pass
    if cfg.use_cat_overlap_min:
        try: ge("cat_overlap_min", cat_overlap(c, cat_coeffs(cfg.Ncut, cfg.cat_alpha, cfg.cat_phase)), cfg.min_cat_overlap)
        except Exception: pass
    if (cfg.use_wigner_negativity_min or cfg.use_wigner_negativity_max or cfg.use_negativity_survival_objective) and xvec is not None and pvec is not None:
        W = wigner_from_coeffs(c, xvec, pvec); neg = wigner_negativity(W, xvec, pvec)
        if cfg.use_wigner_negativity_min: ge("wigner_negativity_min", neg, cfg.min_wigner_negativity)
        if cfg.use_wigner_negativity_max: le("wigner_negativity_max", neg, cfg.max_wigner_negativity)
        if cfg.use_negativity_survival_objective: ge("survival_input_negativity", neg, cfg.min_input_negativity_for_survival)
    max_violation = max((v for _, v in violations), default=0.0)
    violated = [(name, value) for name, value in violations if value > 0.0]
    return {"is_feasible": bool(max_violation <= 0.0), "max_violation": float(max_violation), "total_violation": float(sum(v for _, v in violations)), "violated": violated}


# =============================================================================
# Optimizer worker
# =============================================================================

def optimize_worker(cfg: UserConfig, out_queue: queue.Queue, stop_event: threading.Event) -> None:
    try:
        active = active_fock_indices(cfg)

        if cfg.target_energy < 0.0 or cfg.target_energy > cfg.Ncut - 1:
            raise ValueError("Target energy must satisfy 0 <= N <= Ncut - 1.")

        if cfg.target_energy < active[0] - 1e-12 or cfg.target_energy > active[-1] + 1e-12:
            raise ValueError(
                f"With the selected Fock subspace, target energy must lie between {active[0]} and {active[-1]}."
            )

        energy_min_possible, energy_max_possible = tail_energy_feasibility_bounds(cfg, active)
        if cfg.target_energy < energy_min_possible - 1e-10 or cfg.target_energy > energy_max_possible + 1e-10:
            raise ValueError(
                "The selected constraints are analytically infeasible. "
                f"With the active Fock subspace and high-tail constraint, <n> must lie in [{energy_min_possible:.6g}, {energy_max_possible:.6g}], "
                f"but target_energy={cfg.target_energy:.6g}. Increase Ncut, reduce target energy, or relax/disable the high-Fock tail probability constraint."
            )

        if cfg.use_fixed_displacement and cfg.use_zero_displacement_constraint:
            raise ValueError("Use either fixed displacement or zero displacement, not both.")

        if cfg.use_robust_average_objective and cfg.use_robust_worstcase_objective:
            raise ValueError("Use robust average objective or robust worst-case objective, not both.")

        xvec, pvec = make_grid(cfg)
        preview_mode = str(getattr(cfg, "live_wigner_preview_mode", "Low-resolution"))
        preview_xvec: Optional[np.ndarray] = None
        preview_pvec: Optional[np.ndarray] = None
        if preview_mode == "Low-resolution":
            preview_points = max(21, int(getattr(cfg, "live_wigner_preview_grid_points", 61)))
            if preview_points % 2 == 0:
                preview_points += 1
            preview_xvec = np.linspace(-cfg.x_max, cfg.x_max, preview_points)
            preview_pvec = np.linspace(-cfg.x_max, cfg.x_max, preview_points)
        elif preview_mode == "Full-resolution":
            preview_xvec, preview_pvec = xvec, pvec

        base_params = build_channel(cfg)
        objective_channels = robust_channels(cfg, base_params)
        seeds = make_seeds(cfg, active)

        constraints, unpack = build_constraints(cfg, active, xvec, pvec)

        n_active = len(active)
        n_vars = n_active if cfg.use_real_coefficients else 2 * n_active

        out_queue.put(
            {
                "type": "status",
                "text": (
                    f"Started: profile={cfg.computation_profile}, "
                    f"live_wigner={cfg.live_wigner_preview_mode}, starts={len(seeds)}, active_dim={n_active}, vars={n_vars}, "
                    f"constraints={len(constraints)}, tail_levels={cfg.tail_levels}, "
                    f"max_tail={cfg.max_tail_probability:.3g}, gain={base_params.gain:.5g}, "
                    f"noise_x2={base_params.noise_x2:.5g}, noise_p2={base_params.noise_p2:.5g}, "
                    f"noise_xp={getattr(base_params, 'noise_xp', 0.0):.5g}, "
                    f"phase_sigma={base_params.phase_sigma:.5g}"
                ),
            }
        )

        if cfg.use_tail_probability_constraint:
            try:
                coh_tmp = coherent_coeffs(cfg.Ncut, cfg.target_energy)
                coh_tail = tail_probability(coh_tmp, cfg.tail_levels)
                if coh_tail > cfg.max_tail_probability + 1e-8:
                    out_queue.put(
                        {
                            "type": "status",
                            "text": (
                                f"Note: the coherent reference violates the active tail constraint: "
                                f"P_tail(coherent)={coh_tail:.3e} > {cfg.max_tail_probability:.3e}. "
                                "For the unconstrained coherent benchmark, turn the tail constraint off."
                            ),
                        }
                    )
            except Exception:
                pass

        if cfg.use_wigner_negativity_min or cfg.use_wigner_negativity_max:
            out_queue.put(
                {
                    "type": "status",
                    "text": "Wigner-negativity constraints are enabled. Optimization may be much slower.",
                }
            )

        if cfg.use_negativity_survival_objective:
            out_queue.put(
                {
                    "type": "status",
                    "text": (
                        "Joint objective enabled: score combines fidelity and Wigner-negativity survival. "
                        f"Input negativity must be >= {cfg.min_input_negativity_for_survival:.3e}."
                    ),
                }
            )

        if cfg.use_robust_average_objective or cfg.use_robust_worstcase_objective:
            out_queue.put(
                {
                    "type": "status",
                    "text": f"Robust objective uses {len(objective_channels)} channel samples.",
                }
            )

        cache: Dict[bytes, float] = {}
        history: List[float] = []
        feasible_history: List[float] = []
        all_results: List[Dict[str, object]] = []
        best: Optional[Dict[str, object]] = None
        best_feasible_score: Optional[float] = None
        callback_count = 0

        if cfg.use_robust_worstcase_objective:
            objective_mode = "worst"
        elif cfg.use_robust_average_objective:
            objective_mode = "average"
        else:
            objective_mode = "single"

        def objective(y: np.ndarray) -> float:
            if stop_event.is_set():
                raise StopOptimization()
            key = np.asarray(y).round(11).tobytes()
            if key in cache:
                return cache[key]
            c = unpack(y, normalize=True)
            score, _ = objective_score_for_state(c, xvec, pvec, objective_channels, objective_mode, cfg)
            value = -score
            cache[key] = value
            return value

        def push_live(c: np.ndarray, label: str, local_history: Optional[List[float]] = None) -> None:
            # Compute the scalar score and objective fidelity. For fidelity-only
            # runs this avoids extra Wigner-negativity diagnostics. If the user
            # selected negativity survival, the required Wigner diagnostics are
            # computed because they are part of the mathematical objective.
            score, score_components = objective_score_for_state(c, xvec, pvec, objective_channels, objective_mode, cfg)
            objective_F = float(score_components["objective_fidelity"])
            selection_score = float(score)

            d = state_diagnostics(c)
            tail_prob = tail_probability(c, cfg.tail_levels)
            feas = feasibility_diagnostics(c, cfg, xvec=xvec, pvec=pvec)
            is_feasible = bool(feas["is_feasible"])

            # Live Wigner preview is visualization-only. It can be disabled or
            # computed on a smaller grid so presentations remain smooth and
            # performance runs do not waste time drawing phase-space images.
            W_in = None
            W_out = None
            plot_xvec = None
            plot_pvec = None
            has_wigner = preview_xvec is not None and preview_pvec is not None

            if has_wigner:
                F_preview, W_in, W_out = evaluate_state(c, preview_xvec, preview_pvec, base_params)
                plot_xvec, plot_pvec = preview_xvec, preview_pvec
                display_value = float(F_preview) if objective_mode == "single" else objective_F
                neg = wigner_negativity(W_in, preview_xvec, preview_pvec)
                output_neg = wigner_negativity(W_out, preview_xvec, preview_pvec)
                survival_ratio_base = output_neg / neg if neg >= max(cfg.min_input_negativity_for_survival, 1e-300) else 0.0
            else:
                display_value = objective_F
                neg = float(score_components.get("input_negativity", float("nan")))
                output_neg = float(score_components.get("output_negativity", float("nan")))
                survival_ratio_base = float(score_components.get("negativity_survival_ratio", float("nan")))

            history.append(display_value)
            if is_feasible:
                if len(feasible_history) == 0 or not np.isfinite(feasible_history[-1]):
                    feasible_history.append(display_value)
                else:
                    feasible_history.append(max(feasible_history[-1], display_value))
            else:
                feasible_history.append(feasible_history[-1] if len(feasible_history) > 0 else np.nan)

            if local_history is not None:
                local_history.append(display_value)

            payload = {
                "type": "live",
                "label": label,
                "fidelity": display_value,
                "objective_fidelity": objective_F,
                "selection_score": selection_score,
                "energy": d["energy"],
                "a": d["a"],
                "parity": d["parity"],
                "n_variance": d["n_variance"],
                "mandel_q": d["mandel_q"],
                "x_var": d["x_var"],
                "p_var": d["p_var"],
                "wigner_negativity": neg,
                "output_wigner_negativity": output_neg,
                "negativity_survival_ratio": survival_ratio_base,
                "objective_negativity_survival_ratio": score_components.get("negativity_survival_ratio", float("nan")),
                "tail_probability": tail_prob,
                "feasible": is_feasible,
                "constraint_violation": float(feas["max_violation"]),
                "coeffs": normalize_coeffs(c),
                "history": list(history),
                "feasible_history": list(feasible_history),
                "has_wigner": bool(has_wigner),
                "preview_mode": preview_mode,
            }
            if has_wigner:
                payload.update({"W_in": W_in, "W_out": W_out, "xvec": plot_xvec, "pvec": plot_pvec})
            out_queue.put(payload)

        def make_candidate_item(
            c_state: np.ndarray,
            start_index: int,
            success: bool,
            message: str,
            local_history: Optional[List[float]],
        ) -> Dict[str, object]:
            """Build a fully evaluated candidate dictionary for final selection.

            This helper evaluates seed, SLSQP, and polish candidates with the
            same diagnostics and feasibility checks before deciding which state
            is the true final best feasible state.
            """
            c_state = normalize_coeffs(c_state)

            F_base, W_in_full, W_out_full = evaluate_state(c_state, xvec, pvec, base_params)
            score, comps = objective_score_for_state(c_state, xvec, pvec, objective_channels, objective_mode, cfg)
            d = state_diagnostics(c_state)
            tail_prob = tail_probability(c_state, cfg.tail_levels)
            feas = feasibility_diagnostics(c_state, cfg, xvec=xvec, pvec=pvec)

            input_neg_raw = wigner_negativity(W_in_full, xvec, pvec)
            output_neg_raw = wigner_negativity(W_out_full, xvec, pvec)
            input_neg = clean_wigner_negativity_value(input_neg_raw, c=c_state, cfg=cfg, state_name="Optimized state")
            coherent_like_candidate = is_coherent_like_state(c_state, cfg)
            output_neg = clean_wigner_negativity_value(output_neg_raw, c=c_state if coherent_like_candidate else None, cfg=cfg if coherent_like_candidate else None, state_name="Optimized state")
            if input_neg >= NEGATIVITY_DISPLAY_ZERO_TOL:
                survival_ratio = output_neg / input_neg
            else:
                survival_ratio = 0.0
            if not np.isfinite(survival_ratio):
                survival_ratio = 0.0

            return {
                "start_index": int(start_index),
                "success": bool(success),
                "message": str(message),
                "coeffs": c_state,
                "fidelity": float(F_base),
                "objective_fidelity": float(comps.get("objective_fidelity", F_base)),
                "selection_score": float(score),
                "energy": float(d["energy"]),
                "a": complex(d["a"]),
                "parity": float(d["parity"]),
                "n_variance": float(d["n_variance"]),
                "mandel_q": float(d["mandel_q"]),
                "x_var": float(d["x_var"]),
                "p_var": float(d["p_var"]),
                "xp_cov": float(d["xp_cov"]),
                "fluctuation_energy": float(d["fluctuation_energy"]),
                "displacement_squared": float(d["displacement_squared"]),
                "W_in": W_in_full,
                "W_out": W_out_full,
                "wigner_negativity": float(input_neg),
                "output_wigner_negativity": float(output_neg),
                "negativity_survival_ratio": float(survival_ratio),
                "objective_negativity_survival_ratio": float(comps.get("negativity_survival_ratio", survival_ratio)),
                "tail_probability": float(tail_prob),
                "feasible": bool(feas["is_feasible"]),
                "constraint_violation": float(feas["max_violation"]),
                "violated_constraints": list(feas.get("violated", [])),
                "history": list(local_history or []),
            }

        def consider_for_best(item: Dict[str, object], live_label: str) -> None:
            """Accept only feasible candidates when selecting the final best state."""
            nonlocal best, best_feasible_score
            if not bool(item.get("feasible", False)):
                return
            score = float(item.get("selection_score", item.get("objective_fidelity", item.get("fidelity", -np.inf))))
            if not np.isfinite(score):
                return
            if best is None or best_feasible_score is None or score > best_feasible_score + 1e-12:
                best = item
                best_feasible_score = score
                push_live(np.asarray(item["coeffs"]), live_label)

        for start_index, c0 in enumerate(seeds):
            if stop_event.is_set():
                raise StopOptimization()

            out_queue.put({"type": "status", "text": f"Running start {start_index + 1}/{len(seeds)}..."})
            push_live(c0, f"start {start_index + 1} initial")
            seed_item = make_candidate_item(c0, start_index, True, "initial seed", [])
            consider_for_best(seed_item, f"new best feasible seed from start {start_index + 1}")

            y0 = active_to_y(c0[active], real_only=cfg.use_real_coefficients)
            local_history: List[float] = []

            def callback(y_current: np.ndarray) -> None:
                nonlocal callback_count
                if stop_event.is_set():
                    raise StopOptimization()
                callback_count += 1
                if callback_count % cfg.live_update_every != 0:
                    return
                c_current = unpack(y_current, normalize=True)
                push_live(c_current, f"start {start_index + 1}, callback {callback_count}", local_history)

            result = minimize(
                objective,
                y0,
                method="SLSQP",
                constraints=constraints,
                bounds=[(-2.0, 2.0)] * n_vars,
                callback=callback,
                options={"maxiter": cfg.maxiter, "ftol": cfg.ftol, "disp": False},
            )

            c_opt = unpack(result.x, normalize=True)
            item = make_candidate_item(c_opt, start_index, bool(result.success), str(result.message), local_history)
            all_results.append(item)

            out_queue.put(
                {
                    "type": "status",
                    "text": (
                        f"Finished start {start_index + 1}/{len(seeds)}: "
                        f"F={float(item['fidelity']):.9f}, objective={float(item['objective_fidelity']):.9f}, "
                        f"score={float(item['selection_score']):.9f}, tail={float(item['tail_probability']):.3e}, "
                        f"feasible={bool(item['feasible'])}, violation={float(item['constraint_violation']):.2e}, success={result.success}"
                    ),
                }
            )

            consider_for_best(item, f"new best feasible from start {start_index + 1}")

        if best is None:
            raise RuntimeError(
                "No feasible result was produced. The displayed sampled scores may include infeasible points. "
                "Try increasing maxiter, relaxing active constraints, or changing Ncut/target_energy."
            )

        # Final polishing pass: rerun SLSQP from the best state found by all starts.
        # This often removes Ncut-dependent local-search artifacts.
        if cfg.use_best_polishing:
            out_queue.put({"type": "status", "text": "Polishing best state..."})
            y0_best = active_to_y(best["coeffs"][active], real_only=cfg.use_real_coefficients)
            polish_result = minimize(
                objective,
                y0_best,
                method="SLSQP",
                constraints=constraints,
                bounds=[(-2.0, 2.0)] * n_vars,
                options={"maxiter": cfg.polish_maxiter, "ftol": min(cfg.ftol, 1e-8), "disp": False},
            )
            c_polish = unpack(polish_result.x, normalize=True)
            polish_item = make_candidate_item(c_polish, len(all_results), bool(polish_result.success), "polish: " + str(polish_result.message), [])
            all_results.append(polish_item)
            out_queue.put({
                "type": "status",
                "text": (
                    f"Polish result: F={float(polish_item['fidelity']):.9f}, objective={float(polish_item['objective_fidelity']):.9f}, "
                    f"score={float(polish_item['selection_score']):.9f}, tail={float(polish_item['tail_probability']):.3e}, "
                    f"feasible={bool(polish_item['feasible'])}, violation={float(polish_item['constraint_violation']):.2e}, success={polish_result.success}"
                ),
            })
            consider_for_best(polish_item, "new best feasible after polishing")

        refs = evaluate_references(cfg, xvec, pvec, base_params)

        if cfg.run_analytic_convergence_diagnostics or cfg.run_local_optimality_probe or cfg.run_cutoff_scan:
            out_queue.put({"type": "status", "text": "Running convergence diagnostics..."})
            # Make the current optimization context available to diagnostic routines.
            best["xvec"] = xvec
            best["pvec"] = pvec
            best["params"] = base_params
            best["objective_channels"] = objective_channels
            best["objective_mode"] = objective_mode
            best["active"] = active
            best["config"] = cfg
            try:
                best["convergence_report"] = convergence_diagnostics(best, cfg, stop_event=stop_event)
                verdict = best["convergence_report"].get("verdict", "")
                out_queue.put({"type": "status", "text": "Convergence diagnostics finished: " + str(verdict)})
            except StopOptimization:
                raise
            except Exception as exc:
                best["convergence_report"] = {"error": str(exc)}
                out_queue.put({"type": "status", "text": "Convergence diagnostics failed: " + str(exc)})

        best["all_results"] = all_results
        best["references"] = refs
        best["xvec"] = xvec
        best["pvec"] = pvec
        best["params"] = base_params
        best["objective_channels"] = objective_channels
        best["objective_mode"] = objective_mode
        best["config"] = cfg
        best["history"] = history
        best["feasible_history"] = feasible_history
        best["active"] = active

        out_queue.put({"type": "done", "best": best})

    except StopOptimization:
        out_queue.put({"type": "stopped", "text": "Optimization stopped by user."})
    except Exception as exc:
        out_queue.put({"type": "error", "text": str(exc)})



# =============================================================================
# Comparison and validation runner
# =============================================================================

def safe_filename(name: str) -> str:
    """Return a filesystem-safe label for plot/data export."""
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "state"


def wigner_shape_overlap(W1: np.ndarray, W2: np.ndarray) -> float:
    """Cosine similarity between two sampled Wigner functions."""
    num = float(np.sum(W1 * W2))
    den = float(np.sqrt(np.sum(W1 ** 2) * np.sum(W2 ** 2)))
    if den < 1e-15:
        return float("nan")
    return num / den



def compass_coeffs(Ncut: int, alpha: float = 1.6) -> np.ndarray:
    """Compass-state coefficients used only for comparison/validation."""
    ket = (
        qt.coherent(Ncut, alpha)
        + qt.coherent(Ncut, -alpha)
        + qt.coherent(Ncut, 1j * alpha)
        + qt.coherent(Ncut, -1j * alpha)
    )
    return normalize_coeffs(np.asarray(ket.full()).flatten())


def approximate_gkp_coeffs(Ncut: int, delta: float = 0.45, K: int = 2) -> np.ndarray:
    """Small approximate GKP-like grid state for comparison plots.

    This is intentionally modest because exact GKP diagnostics are expensive.
    It is a visual/benchmark reference, not a fully optimized GKP construction.
    """
    psi = 0 * qt.basis(Ncut, 0)
    for k in range(-K, K + 1):
        x_shift = 2.0 * k * np.sqrt(np.pi)
        beta = x_shift / np.sqrt(2.0)
        peak = qt.displace(Ncut, beta) * qt.squeeze(Ncut, -np.log(delta)) * qt.basis(Ncut, 0)
        envelope = np.exp(-0.5 * (delta * x_shift) ** 2)
        psi = psi + envelope * peak
    return normalize_coeffs(np.asarray(psi.unit().full()).flatten())


def evaluate_from_cached_wigner(
    W_in: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    params: ChannelParams,
) -> Tuple[float, np.ndarray, float, float, float]:
    """Apply a channel to a cached input Wigner function and return diagnostics."""
    W_out = apply_channel(W_in, xvec, pvec, params)
    F = wigner_fidelity(W_in, W_out, xvec, pvec)
    neg_in = wigner_negativity(W_in, xvec, pvec)
    neg_out = wigner_negativity(W_out, xvec, pvec)
    overlap = wigner_shape_overlap(W_in, W_out)
    return F, W_out, neg_in, neg_out, overlap


def _comparison_channel_for_varied_parameter(cfg: UserConfig, parameter: str, value: float) -> ChannelParams:
    """Build the configured channel after varying one experimental parameter."""
    if parameter == "r":
        return build_channel(replace(cfg, r=float(value), use_finite_squeezing_noise=True))
    if parameter == "eta":
        return build_channel(replace(cfg, detector_efficiency_eta=float(value), use_detector_noise=True))
    if parameter == "gain":
        return build_channel(replace(cfg, use_gain_mismatch=True, gain_if_enabled=float(value)))
    if parameter == "phase_sigma":
        return build_channel(replace(cfg, use_phase_diffusion=True, phase_sigma=float(value)))
    raise ValueError(f"Unknown comparison sweep parameter: {parameter}")


def _comparison_channel_for_loss_heatmap(cfg: UserConfig, transmissivity: float, n_th: float) -> ChannelParams:
    return build_channel(
        replace(
            cfg,
            use_loss_channel_noise=True,
            loss_transmissivity=float(transmissivity),
            loss_thermal_nbar=float(n_th),
        )
    )


def _comparison_channel_for_asymmetry(cfg: UserConfig, asymmetry: float) -> ChannelParams:
    """Create a channel with x/p noise imbalance while preserving the base scale."""
    base = build_channel(replace(cfg, use_anisotropic_noise=False))
    scale = max(np.sqrt(max(base.noise_x2 * base.noise_p2, 0.0)), 1e-14)
    nx = scale * np.exp(float(asymmetry))
    np_ = scale * np.exp(-float(asymmetry))
    n_xp = getattr(base, "noise_xp", 0.0)
    nx, np_, n_xp = _enforce_full_gaussian_cp(base.gain, nx, np_, n_xp, cfg.auto_enforce_cp_noise)
    return replace(base, noise_x2=float(nx), noise_p2=float(np_), noise_xp=float(n_xp))

def comparison_state_dictionary(best: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Build the standard states compared against the optimized state.

    The optimized state and coherent/squeezed references are energy-matched to
    the configured target whenever possible. Other states are included as named
    experimental/nonclassical benchmarks and their actual energies are reported
    explicitly in the comparison table.
    """
    cfg: UserConfig = best["config"]
    Ncut = int(cfg.Ncut)
    target = float(cfg.target_energy)
    states: Dict[str, np.ndarray] = {
        "Optimized state": normalize_coeffs(np.asarray(best["coeffs"])),
        "Coherent": coherent_coeffs(Ncut, target),
    }

    try:
        states["Even cat"] = cat_coeffs(Ncut, alpha=max(0.8, np.sqrt(max(target, 0.2))), phase=0.0)
    except Exception:
        pass
    try:
        states["Odd cat"] = cat_coeffs(Ncut, alpha=max(0.8, np.sqrt(max(target, 0.2))), phase=np.pi)
    except Exception:
        pass
    try:
        nearest = int(np.clip(round(target), 0, Ncut - 1))
        states[f"Fock |{nearest}>"] = fock_coeffs(Ncut, nearest)
    except Exception:
        pass
    try:
        states["Squeezed vacuum"] = squeezed_vacuum_coeffs(Ncut, target)
    except Exception:
        pass
    try:
        states["Compass"] = compass_coeffs(Ncut, alpha=max(1.0, np.sqrt(max(target, 0.4))))
    except Exception:
        pass
    try:
        states["Approx. GKP"] = approximate_gkp_coeffs(Ncut, delta=0.45, K=2)
    except Exception:
        pass

    try:
        states["Feasible two-Fock"] = two_fock_seed(
            Ncut,
            target,
            active_fock_indices(cfg),
            np.random.default_rng(cfg.random_seed + 99),
            zero_displacement=cfg.use_zero_displacement_constraint,
            deterministic=True,
        )
    except Exception:
        pass

    # Preserve insertion order while removing accidental duplicates.
    unique: Dict[str, np.ndarray] = {}
    for name, c in states.items():
        c = normalize_coeffs(c)
        if not any(abs(np.vdot(c, u)) > 1.0 - 1e-10 for u in unique.values()):
            unique[name] = c
    return unique


def run_comparison_analysis(best: Dict[str, object], output_root: str = DEFAULT_COMPARISON_EXPORT_ROOT, save_to_disk: bool = False) -> Dict[str, object]:
    """Compare the optimized state with reference states and generate sweep data.

    The comparison uses the same configured channel as the optimization for the
    summary metrics.  It also produces the main parameter-sweep plots from the
    previous standalone comparison code, but keeps the results in memory until
    the user explicitly presses Save Comparison Results.
    """
    cfg: UserConfig = best["config"]
    params: ChannelParams = best["params"]
    xvec = np.asarray(best["xvec"])
    pvec = np.asarray(best["pvec"])
    states = comparison_state_dictionary(best)

    rows: List[Dict[str, object]] = []
    wigners: Dict[str, Dict[str, object]] = {}
    cached_W_in: Dict[str, np.ndarray] = {}

    for name, c in states.items():
        c = normalize_coeffs(c)
        W_in = wigner_from_coeffs(c, xvec, pvec)
        F, W_out, neg_in_raw, neg_out_raw, overlap = evaluate_from_cached_wigner(W_in, xvec, pvec, params)
        cached_W_in[name] = W_in

        coherent_like = ("coherent" in str(name).lower()) or is_coherent_like_state(c, cfg)
        neg_in = clean_wigner_negativity_value(neg_in_raw, c=c, cfg=cfg, state_name=name)
        neg_out = clean_wigner_negativity_value(neg_out_raw, c=c if coherent_like else None, cfg=cfg if coherent_like else None, state_name=name if coherent_like else "")

        if neg_in <= NEGATIVITY_DISPLAY_ZERO_TOL:
            survival = float("nan")
            survival_label = "N/A: input negativity is zero"
        else:
            survival = float(neg_out / neg_in)
            survival_label = f"{survival:.6g}"

        d = state_diagnostics(c)
        try:
            feas = feasibility_diagnostics(c, cfg, xvec=xvec, pvec=pvec)
            allowed = bool(feas["is_feasible"])
            max_violation = float(feas["max_violation"])
            violated = list(feas["violated"])
        except Exception:
            allowed = state_allowed(c, cfg)
            max_violation = float("nan")
            violated = []

        row = {
            "state": name,
            "fidelity": float(F),
            "energy": float(d["energy"]),
            "parity": float(d["parity"]),
            "displacement_squared": float(d["displacement_squared"]),
            "n_variance": float(d["n_variance"]),
            "mandel_q": float(d["mandel_q"]),
            "input_negativity": float(neg_in),
            "output_negativity": float(neg_out),
            "negativity_survival": survival,
            "negativity_survival_label": survival_label,
            "wigner_shape_overlap": float(overlap),
            "allowed_by_active_constraints": allowed,
            "max_constraint_violation": max_violation,
            "violated_constraints": "; ".join(str(v) for v in violated),
        }
        rows.append(row)
        wigners[name] = {"W_in": W_in, "W_out": W_out, "coeffs": c}

    rows.sort(key=lambda r: float(r["fidelity"]), reverse=True)

    # Parameter sweeps.  These reproduce the previous comparison-code plots but
    # use cached input Wigner functions and the app's configured channel builder.
    sweep_state_names = list(states.keys())
    sweeps: Dict[str, object] = {}

    def fidelity_series_for_channels(channels: Sequence[ChannelParams], names: Optional[Sequence[str]] = None):
        selected = list(names) if names is not None else sweep_state_names
        series = {}
        kernel_cache: Dict[Tuple[float, float, float, int, int, float, float], np.ndarray] = {}
        for name in selected:
            W_in = cached_W_in[name]
            vals = []
            for ch in channels:
                n_xp = float(getattr(ch, "noise_xp", 0.0))
                kernel = None
                if ch.noise_x2 > 1e-14 or ch.noise_p2 > 1e-14 or abs(n_xp) > 1e-14:
                    key = channel_kernel_key(ch, xvec, pvec)
                    kernel = kernel_cache.get(key)
                    if kernel is None:
                        kernel = gaussian_kernel(xvec, pvec, ch.noise_x2, ch.noise_p2, n_xp)
                        kernel_cache[key] = kernel
                W_out = apply_channel(W_in, xvec, pvec, ch, precomputed_kernel=kernel)
                vals.append(wigner_fidelity(W_in, W_out, xvec, pvec))
            series[name] = vals
        return series

    # 1. Fidelity versus squeezing parameter for all comparison states.
    sweep_points = int(np.clip(getattr(cfg, "comparison_sweep_points", 31), 5, 151))
    heatmap_points = int(np.clip(sweep_points, 8, 60))
    r_values = np.linspace(0.0, 2.5, sweep_points)
    channels_r = [_comparison_channel_for_varied_parameter(cfg, "r", float(r)) for r in r_values]
    fidelity_vs_r_series = fidelity_series_for_channels(channels_r)
    sweeps["fidelity_vs_squeezing"] = {
        "title": "Fidelity vs squeezing parameter",
        "x_label": "squeezing parameter r",
        "y_label": "teleportation fidelity",
        "x": r_values.tolist(),
        "series": fidelity_vs_r_series,
        # Use the actual coherent-state curve under the configured channel, not
        # the ideal finite-squeezing-only formula, when extra noises are active.
        "benchmark": fidelity_vs_r_series.get("Coherent", (1.0 / (1.0 + np.exp(-2.0 * r_values))).tolist()),
        "ideal_coherent_benchmark": (1.0 / (1.0 + np.exp(-2.0 * r_values))).tolist(),
    }

    # 2. Detector efficiency effect: optimized state, curves over r for eta values.
    eta_values = [1.0, 0.95, 0.90, 0.80, 0.70]
    detector_series = {}
    W_opt = cached_W_in["Optimized state"]
    for eta in eta_values:
        vals = []
        for r in r_values:
            ch = build_channel(replace(cfg, r=float(r), use_finite_squeezing_noise=True, use_detector_noise=True, detector_efficiency_eta=float(eta)))
            W_out = apply_channel(W_opt, xvec, pvec, ch)
            vals.append(wigner_fidelity(W_opt, W_out, xvec, pvec))
        detector_series[f"eta={eta:g}"] = vals
    sweeps["detector_efficiency_vs_squeezing"] = {
        "title": "Detector efficiency effect on optimized state",
        "x_label": "squeezing parameter r",
        "y_label": "teleportation fidelity",
        "x": r_values.tolist(),
        "series": detector_series,
    }

    # 3. Loss/thermal heat map for the optimized state.
    T_values = np.linspace(0.75, 1.0, heatmap_points)
    nth_values = np.linspace(0.0, 0.08, heatmap_points)
    F_map = np.zeros((len(nth_values), len(T_values)))
    for i, nth in enumerate(nth_values):
        for j, T in enumerate(T_values):
            ch = _comparison_channel_for_loss_heatmap(cfg, float(T), float(nth))
            W_out = apply_channel(W_opt, xvec, pvec, ch)
            F_map[i, j] = wigner_fidelity(W_opt, W_out, xvec, pvec)
    sweeps["loss_thermal_heatmap"] = {
        "title": "Loss and thermal-noise heat map: optimized state",
        "x_label": "transmissivity T",
        "y_label": "thermal photon number n_th",
        "x": T_values.tolist(),
        "y": nth_values.tolist(),
        "map": F_map.tolist(),
    }

    # 4. Fidelity versus asymmetric quadrature noise for all states.
    asymmetry_values = np.linspace(-2.5, 2.5, sweep_points)
    channels_asym = [_comparison_channel_for_asymmetry(cfg, float(a)) for a in asymmetry_values]
    sweeps["fidelity_vs_asymmetry"] = {
        "title": "Effect of asymmetric quadrature noise",
        "x_label": "asymmetry parameter",
        "y_label": "teleportation fidelity",
        "x": asymmetry_values.tolist(),
        "series": fidelity_series_for_channels(channels_asym),
    }

    # 5. Fidelity versus feed-forward gain for all states.
    gain_values = np.linspace(0.5, 2.0, sweep_points)
    channels_gain = [_comparison_channel_for_varied_parameter(cfg, "gain", float(g)) for g in gain_values]
    sweeps["fidelity_vs_gain"] = {
        "title": "Gain optimization under configured noise",
        "x_label": "feed-forward gain g",
        "y_label": "teleportation fidelity",
        "x": gain_values.tolist(),
        "series": fidelity_series_for_channels(channels_gain),
    }

    # 6. Wigner-negativity survival versus squeezing for nonclassical states.
    survival_series = {}
    for name in sweep_state_names:
        c_state = states[name]
        W_in = cached_W_in[name]
        neg_in = clean_wigner_negativity(W_in, xvec, pvec, c=c_state, cfg=cfg, state_name=name)
        if neg_in <= NEGATIVITY_DISPLAY_ZERO_TOL:
            continue
        vals = []
        for ch in channels_r:
            W_out = apply_channel(W_in, xvec, pvec, ch)
            neg_out = clean_wigner_negativity_value(wigner_negativity(W_out, xvec, pvec), c=c_state, cfg=cfg, state_name=name)
            vals.append(neg_out / neg_in if neg_in > 0 else float("nan"))
        survival_series[name] = vals
    sweeps["negativity_survival_vs_squeezing"] = {
        "title": "Wigner negativity survival vs squeezing",
        "x_label": "squeezing parameter r",
        "y_label": "negativity survival ratio",
        "x": r_values.tolist(),
        "series": survival_series,
    }

    # 7. Phase diffusion sensitivity for optimized state.
    phase_values = np.linspace(0.0, 0.5, sweep_points)
    F_phase = []
    neg_phase = []
    for ph in phase_values:
        ch = _comparison_channel_for_varied_parameter(cfg, "phase_sigma", float(ph))
        W_out = apply_channel(W_opt, xvec, pvec, ch)
        F_phase.append(wigner_fidelity(W_opt, W_out, xvec, pvec))
        neg_phase.append(clean_wigner_negativity_value(wigner_negativity(W_out, xvec, pvec), c=states.get("Optimized state"), cfg=cfg, state_name="Optimized state"))
    sweeps["phase_diffusion_sensitivity"] = {
        "title": "Phase diffusion sensitivity: optimized state",
        "x_label": "phase diffusion sigma",
        "y_label": "metric value",
        "x": phase_values.tolist(),
        "series": {
            "fidelity": F_phase,
            "output negativity": neg_phase,
        },
    }

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    channel_summary = {
        "format": COMPARISON_EXPORT_FORMAT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_stamp": run_stamp,
        "channel": _json_safe(params),
        "config_snapshot": _json_safe(asdict(cfg)),
        "negativity_zero_tolerance": NEGATIVITY_ZERO_TOL,
        "comparison_plots_generated": list(sweeps.keys()),
        "comparison_sweep_points": sweep_points,
        "comparison_heatmap_points": heatmap_points,
        "noise_combination_convention": "finite squeezing is baseline noise; anisotropic/thermal/detector/loss/extra noises are additive contributions",
        "output_folder": "",
    }

    result = {
        "rows": rows,
        "wigners": wigners,
        "xvec": xvec,
        "pvec": pvec,
        "sweeps": sweeps,
        "output_folder": "",
        "png_folder": "",
        "data_folder": "",
        "csv_path": "",
        "channel_summary": channel_summary,
    }
    if save_to_disk:
        export_comparison_result_payload(result, output_root)
    return result


def export_comparison_result_payload(result: Dict[str, object], output_root: str = DEFAULT_COMPARISON_EXPORT_ROOT) -> str:
    """Write an already-computed comparison result to a timestamped folder.

    This function is intentionally separate from run_comparison_analysis so the
    GUI can display comparison results first and save only after the user asks.
    """
    rows = list(result.get("rows", []))
    wigners = dict(result.get("wigners", {}))
    xvec = np.asarray(result.get("xvec"))
    pvec = np.asarray(result.get("pvec"))

    channel_summary = dict(result.get("channel_summary", {}))
    stamp = str(channel_summary.get("run_stamp") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
    run_dir = os.path.join(output_root, f"comparison_run_{stamp}")
    png_dir = os.path.join(run_dir, "pngs")
    wigner_dir = os.path.join(png_dir, "wigner_pairs")
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(wigner_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "comparison_results.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(_json_safe(row))

    channel_summary["output_folder"] = run_dir
    channel_summary["exported_at"] = datetime.now().isoformat(timespec="seconds")
    channel_summary["environment"] = {
        "app_version": APP_VERSION,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "qutip_version": getattr(qt, "__version__", "unknown"),
        "quadrature_convention": "x=(a+a_dag)/sqrt(2), Var(vacuum)=0.5",
    }
    with open(os.path.join(data_dir, "comparison_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(channel_summary), f, indent=2, ensure_ascii=False)

    sweeps = dict(result.get("sweeps", {}))
    if sweeps:
        with open(os.path.join(data_dir, "comparison_sweep_data.json"), "w", encoding="utf-8") as f:
            json.dump(_json_safe(sweeps), f, indent=2, ensure_ascii=False)

    for name, pack in wigners.items():
        fig = Figure(figsize=(11, 4.8), dpi=140, facecolor="#111c33")
        ax1 = fig.add_axes([0.07, 0.15, 0.36, 0.75])
        cax1 = fig.add_axes([0.44, 0.15, 0.018, 0.75])
        ax2 = fig.add_axes([0.56, 0.15, 0.36, 0.75])
        cax2 = fig.add_axes([0.93, 0.15, 0.018, 0.75])
        for ax, cax, W, title in [
            (ax1, cax1, np.asarray(pack["W_in"]), f"{name}: input"),
            (ax2, cax2, np.asarray(pack["W_out"]), f"{name}: output"),
        ]:
            zmax = float(np.max(np.abs(W)))
            if zmax < 1e-12:
                zmax = 1.0
            im = ax.imshow(W, origin="lower", extent=[xvec[0], xvec[-1], pvec[0], pvec[-1]], aspect="equal", vmin=-zmax, vmax=zmax, cmap="RdBu_r")
            ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold")
            ax.set_xlabel("x", color="#f8fafc")
            ax.set_ylabel("p", color="#f8fafc")
            ax.tick_params(colors="#cbd5e1")
            ax.set_facecolor("#0f172a")
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(colors="#cbd5e1")
            cb.outline.set_edgecolor("#243656")
        fig.savefig(os.path.join(wigner_dir, f"wigner_pair_{safe_filename(name)}.png"), bbox_inches="tight")

    result["output_folder"] = run_dir
    result["png_folder"] = png_dir
    result["data_folder"] = data_dir
    result["csv_path"] = csv_path
    result["channel_summary"] = channel_summary
    return run_dir


def comparison_worker(best: Dict[str, object], out_queue: queue.Queue) -> None:
    try:
        out_queue.put({"type": "comparison_status", "text": "Running comparison tests and parameter sweeps with the final optimized state..."})
        result = run_comparison_analysis(best, DEFAULT_COMPARISON_EXPORT_ROOT, save_to_disk=False)
        out_queue.put({"type": "comparison_done", "result": result})
    except Exception as exc:
        out_queue.put({"type": "comparison_error", "text": str(exc)})


# =============================================================================
# GUI
# =============================================================================

class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background="#07111f")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.canvas = canvas
        self.scrollbar = scrollbar
        self.inner = ttk.Frame(canvas)
        self._window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def _update_scrollregion(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _fit_inner_width(event):
            # Make the settings content use the full available width instead of
            # staying as a narrow strip pinned to the far left.
            canvas.itemconfigure(self._window_id, width=event.width)

        self.inner.bind("<Configure>", _update_scrollregion)
        canvas.bind("<Configure>", _fit_inner_width)
        canvas.configure(yscrollcommand=scrollbar.set)

        def _mousewheel(event):
            # Windows/macOS mouse-wheel event.  Bind only while the pointer is
            # over this scroll frame; otherwise hidden tabs can steal scrolling
            # from the Information or Comparison pages.
            delta = int(-1 * (event.delta / 120)) if getattr(event, "delta", 0) else 0
            if delta:
                canvas.yview_scroll(delta, "units")

        def _bind_mousewheel(_event=None):
            canvas.bind_all("<MouseWheel>", _mousewheel)

        def _unbind_mousewheel(_event=None):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)
        self.inner.bind("<Enter>", _bind_mousewheel)
        self.inner.bind("<Leave>", _unbind_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def refresh_layout(self) -> None:
        """Recompute the scroll region and inner width after hidden-tab or fullscreen layout changes."""
        try:
            self.update_idletasks()
            self.canvas.itemconfigure(self._window_id, width=max(1, self.canvas.winfo_width()))
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except Exception:
            pass


class ModernDropdown(tk.Frame):
    """Custom dark-theme dropdown using a Toplevel popup.

    Windows' native ttk.Combobox popup can ignore dark styling and sometimes
    fails to open reliably inside complex themed Tk layouts. This widget avoids
    that by drawing its own small popup list. It intentionally implements only
    the subset of Combobox behavior used by this app: get(), set(), configure
    with values/state, and the <<ComboboxSelected>> virtual event.
    """

    def __init__(
        self,
        master,
        textvariable: tk.StringVar,
        values: Sequence[str],
        width: int = 26,
        colors: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.variable = textvariable
        self.values: List[str] = []
        self.width_chars = int(width)
        self.colors = colors or {
            "bg": "#0b172c",
            "card": "#111c33",
            "plot": "#0f172a",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "subtle": "#94a3b8",
            "active": "#2563eb",
            "accent": "#38bdf8",
            "border": "#243656",
        }
        self._state = str(kwargs.pop("state", "normal"))
        self.popup: Optional[tk.Toplevel] = None
        self.display_var = tk.StringVar(value=self._display_text(self.variable.get()))

        bg = self.colors.get("plot", self.colors.get("bg", "#0b172c"))
        border = self.colors.get("border", "#243656")
        accent = self.colors.get("accent", "#38bdf8")
        fg = self.colors.get("text", "#f8fafc")

        super().__init__(
            master,
            background=bg,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=accent,
            borderwidth=0,
            cursor="hand2",
            **kwargs,
        )

        self.label = tk.Label(
            self,
            textvariable=self.display_var,
            width=self.width_chars,
            anchor="w",
            justify="left",
            padx=10,
            pady=7,
            background=bg,
            foreground=fg,
            font=("Segoe UI", 10),
            cursor="hand2",
        )
        self.label.pack(side="left", fill="both", expand=True)

        self.arrow = tk.Label(
            self,
            text="▾",
            width=2,
            anchor="center",
            background=bg,
            foreground=accent,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.arrow.pack(side="right", fill="y")

        for widget in (self, self.label, self.arrow):
            widget.bind("<Button-1>", self._open_popup)
            widget.bind("<Return>", self._open_popup)
            widget.bind("<space>", self._open_popup)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

        self.set_values(values)
        self.variable.trace_add("write", lambda *_: self.display_var.set(self._display_text(self.variable.get())))
        self._refresh_state_style()

    def _display_text(self, value: str) -> str:
        value = str(value) if value is not None else ""
        return value if value else "Select..."

    def _on_enter(self, _event=None) -> None:
        if self._state == "disabled":
            return
        hover = self.colors.get("card2", self.colors.get("card", "#152443"))
        self.configure(background=hover)
        self.label.configure(background=hover)
        self.arrow.configure(background=hover)

    def _on_leave(self, _event=None) -> None:
        bg = self.colors.get("plot", self.colors.get("bg", "#0b172c"))
        self.configure(background=bg)
        self.label.configure(background=bg)
        self.arrow.configure(background=bg)

    def _refresh_state_style(self) -> None:
        fg = self.colors.get("text", "#f8fafc") if self._state != "disabled" else self.colors.get("subtle", "#94a3b8")
        self.label.configure(foreground=fg)
        self.arrow.configure(foreground=self.colors.get("accent", "#38bdf8") if self._state != "disabled" else self.colors.get("subtle", "#94a3b8"))

    def set_values(self, values: Sequence[str]) -> None:
        self.values = [str(v) for v in values]
        if self.variable.get() and self.variable.get() not in self.values and self.values:
            # Keep user-entered custom values if they were intentionally set.
            pass

    def _open_popup(self, _event=None) -> str:
        if self._state == "disabled":
            return "break"
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
            self.popup = None
            return "break"
        if not self.values:
            return "break"

        self.update_idletasks()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 2
        width_px = max(self.winfo_width(), 220)
        visible = min(max(len(self.values), 1), 12)
        item_h = 26
        height_px = visible * item_h + 6

        top = tk.Toplevel(self)
        self.popup = top
        top.overrideredirect(True)
        top.transient(self.winfo_toplevel())
        top.configure(background=self.colors.get("border", "#243656"))
        top.geometry(f"{width_px}x{height_px}+{x}+{y}")
        try:
            top.attributes("-topmost", True)
        except Exception:
            pass

        frame = tk.Frame(top, background=self.colors.get("border", "#243656"), padx=1, pady=1)
        frame.pack(fill="both", expand=True)
        listbox = tk.Listbox(
            frame,
            activestyle="none",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            selectmode="browse",
            height=visible,
            background=self.colors.get("plot", "#0f172a"),
            foreground=self.colors.get("text", "#f8fafc"),
            selectbackground=self.colors.get("accent2", "#2563eb"),
            selectforeground="#ffffff",
            font=("Segoe UI", 10),
            exportselection=False,
        )
        for value in self.values:
            listbox.insert("end", value)
        try:
            current_index = self.values.index(self.variable.get())
            listbox.selection_set(current_index)
            listbox.see(current_index)
        except Exception:
            pass
        listbox.pack(side="left", fill="both", expand=True)

        if len(self.values) > visible:
            sb = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
            listbox.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")

        def choose(_event=None):
            selection = listbox.curselection()
            if selection:
                self._select(self.values[int(selection[0])])
            self._close_popup()
            return "break"

        listbox.bind("<ButtonRelease-1>", choose)
        listbox.bind("<Return>", choose)
        listbox.bind("<Escape>", lambda _e: self._close_popup())
        top.bind("<Escape>", lambda _e: self._close_popup())
        top.bind("<FocusOut>", lambda _e: top.after(160, self._close_popup))
        top.lift()
        top.focus_force()
        listbox.focus_set()
        return "break"

    def _close_popup(self, _event=None):
        if self.popup is not None:
            try:
                if self.popup.winfo_exists():
                    self.popup.destroy()
            except Exception:
                pass
        self.popup = None
        return "break"

    def _select(self, value: str) -> None:
        self.variable.set(value)
        self.display_var.set(self._display_text(value))
        self.event_generate("<<ComboboxSelected>>")

    def get(self) -> str:
        return self.variable.get()

    def set(self, value: str) -> None:
        self.variable.set(value)
        self.display_var.set(self._display_text(value))

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        values = kwargs.pop("values", None)
        if values is not None:
            self.set_values(values)
        state = kwargs.pop("state", None)
        if state is not None:
            self._state = str(state)
            self._refresh_state_style()
        width = kwargs.pop("width", None)
        if width is not None:
            self.width_chars = int(width)
            self.label.configure(width=self.width_chars)
        if kwargs:
            return super().configure(**kwargs)
        return None

    config = configure

    def cget(self, key: str):
        if key == "state":
            return self._state
        if key == "values":
            return tuple(self.values)
        if key == "width":
            return self.width_chars
        return super().cget(key)

class FidelityApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CV Teleportation Fidelity Optimizer")
        self.root.geometry("1600x950")
        self.root.minsize(1180, 760)
        self._setup_style()

        self.vars: Dict[str, tk.Variable] = {}
        self.out_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.comparison_thread: Optional[threading.Thread] = None
        self.last_best: Optional[Dict[str, object]] = None
        self.current_wigner_payload: Optional[Dict[str, object]] = None
        self.combo_widgets: Dict[str, object] = {}
        self.entry_widgets: Dict[str, object] = {}
        self.check_widgets: Dict[str, object] = {}
        self._last_combo_values: Dict[str, str] = {}
        self._profile_apply_after_id = None

        # Computation profiles are one-shot numerical templates, not locks.
        # This flag lets bulk profile application update many variables without
        # marking the profile as Custom after every internal set().
        self._applying_template = False

        self._build()
        self._layout_refresh_after_id = None
        self._install_layout_stabilizers()
        # Realize hidden tabs once before the user maximizes/fullscreens the app.
        # This prevents Matplotlib canvases from keeping the small hidden-tab geometry
        # that caused plots to appear shifted until every tab had been visited manually.
        self._warmup_tabs_for_layout()
        self._schedule_layout_refresh(delay=80)
        self._poll_queue()

    def _setup_style(self) -> None:
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        # Presentation palette: deep navy background, soft slate cards, and cyan/blue accents.
        self.COLORS = {
            "bg": "#07111f",
            "bg2": "#0b1220",
            "card": "#111c33",
            "card2": "#152443",
            "plot": "#0f172a",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "subtle": "#94a3b8",
            "accent": "#38bdf8",
            "accent2": "#2563eb",
            "border": "#243656",
            "ok": "#34d399",
            "warn": "#fbbf24",
            "bad": "#fb7185",
        }
        C = self.COLORS

        self.root.configure(background=C["bg"])
        self.style.configure("TFrame", background=C["bg"])
        self.style.configure("Card.TFrame", background=C["card"], relief="flat")
        self.style.configure("PlotCard.TFrame", background=C["card"], relief="flat")
        self.style.configure("Toolbar.TFrame", background=C["card"], relief="flat")
        self.style.configure("Header.TFrame", background=C["bg"], relief="flat")
        self.style.configure("Hero.TFrame", background=C["bg"], relief="flat")
        self.style.configure("HeroInner.TFrame", background=C["bg"], relief="flat")
        self.style.configure("MetricCard.TFrame", background=C["card"], relief="flat")

        self.style.configure("TLabel", background=C["bg"], foreground=C["text"], font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", background=C["bg"], foreground=C["text"], font=("Segoe UI", 24, "bold"))
        self.style.configure("Subtitle.TLabel", background=C["bg"], foreground=C["muted"], font=("Segoe UI", 10))
        self.style.configure("HeroTitle.TLabel", background=C["bg"], foreground=C["text"], font=("Segoe UI", 32, "bold"))
        self.style.configure("HeroSubtitle.TLabel", background=C["bg"], foreground=C["muted"], font=("Segoe UI", 12))
        self.style.configure("HeroSmall.TLabel", background=C["bg"], foreground=C["accent"], font=("Segoe UI", 10, "bold"))
        self.style.configure("Section.TLabel", background=C["card"], foreground=C["accent"], font=("Segoe UI", 11, "bold"))
        self.style.configure("Card.TLabel", background=C["card"], foreground=C["muted"], font=("Segoe UI", 10))
        self.style.configure("Small.TLabel", background=C["card"], foreground=C["subtle"], font=("Segoe UI", 9))
        self.style.configure("Metric.TLabel", background=C["card"], foreground=C["text"], font=("Segoe UI", 9, "bold"))
        self.style.configure("Toolbar.TLabel", background=C["card"], foreground=C["muted"], font=("Segoe UI", 10, "bold"))

        self.style.configure("TNotebook", background=C["bg"], borderwidth=0)
        self.style.configure("TNotebook.Tab", padding=(22, 10), font=("Segoe UI", 10, "bold"), background=C["card"], foreground=C["muted"])
        self.style.map("TNotebook.Tab", background=[("selected", C["accent2"]), ("active", C["card2"])], foreground=[("selected", "#ffffff"), ("active", "#ffffff")])

        self.style.configure("TButton", padding=(12, 7), font=("Segoe UI", 10, "bold"), background=C["card2"], foreground=C["text"], bordercolor=C["border"])
        self.style.map("TButton", background=[("active", C["accent2"]), ("disabled", "#1e293b")], foreground=[("disabled", "#64748b")])
        self.style.configure("Accent.TButton", padding=(14, 9), font=("Segoe UI", 11, "bold"), background=C["accent2"], foreground="#ffffff")
        self.style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#334155")], foreground=[("disabled", "#94a3b8")])

        self.style.configure("TCheckbutton", background=C["card"], foreground=C["muted"], font=("Segoe UI", 10))
        self.style.map("TCheckbutton", background=[("active", C["card"])], foreground=[("active", C["text"])])

        # Modern dark input widgets.  The older light-gray native comboboxes looked
        # disconnected from the presentation theme, especially when opened in
        # fullscreen mode.  The option_add calls also style the dropdown listbox
        # used internally by ttk.Combobox on Windows/Tk.
        input_bg = "#0b172c"
        input_active = "#10213d"
        self.root.option_add("*TCombobox*Listbox.background", input_bg)
        self.root.option_add("*TCombobox*Listbox.foreground", C["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", C["accent2"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.font", "Segoe UI 10")

        self.style.configure(
            "TCombobox",
            padding=(10, 7),
            arrowsize=18,
            font=("Segoe UI", 10),
            fieldbackground=input_bg,
            background=input_bg,
            foreground=C["text"],
            arrowcolor=C["accent"],
            bordercolor=C["border"],
            lightcolor=C["border"],
            darkcolor=C["border"],
            relief="flat",
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", input_bg), ("focus", input_active), ("active", input_active), ("disabled", "#111827")],
            background=[("readonly", input_bg), ("focus", input_active), ("active", input_active), ("disabled", "#111827")],
            foreground=[("readonly", C["text"]), ("focus", C["text"]), ("active", C["text"]), ("disabled", "#64748b")],
            arrowcolor=[("active", "#67e8f9"), ("disabled", "#64748b")],
            bordercolor=[("focus", C["accent"]), ("active", C["accent"]), ("readonly", C["border"])],
        )
        self.style.configure(
            "TEntry",
            padding=(9, 7),
            font=("Segoe UI", 10),
            fieldbackground=input_bg,
            foreground=C["text"],
            insertcolor=C["text"],
            bordercolor=C["border"],
            lightcolor=C["border"],
            darkcolor=C["border"],
            relief="flat",
        )
        self.style.map(
            "TEntry",
            fieldbackground=[("focus", input_active), ("disabled", "#111827")],
            foreground=[("disabled", "#64748b")],
            bordercolor=[("focus", C["accent"]), ("active", C["accent"]), ("!focus", C["border"])],
        )
        self.style.configure("Horizontal.TSeparator", background=C["border"])

    def _build(self) -> None:
        # Settings, plots, fields, and final report are true full-page tabs.
        # This avoids the old permanent left sidebar and gives plots the full width.
        shell = ttk.Frame(self.root)
        shell.pack(fill="both", expand=True, padx=10, pady=10)

        self.main_notebook = ttk.Notebook(shell)
        self.main_notebook.pack(fill="both", expand=True)

        self.welcome_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.welcome_tab, text="Welcome")
        self._build_welcome(self.welcome_tab)

        self.info_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.info_tab, text="Information")
        self._build_information(self.info_tab)

        self.settings_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.settings_tab, text="Settings / Run")

        self._build_controls(self.settings_tab)
        self._build_plots(self.main_notebook)
        self.main_notebook.select(self.welcome_tab)

    def _build_welcome(self, parent) -> None:
        """Presentation welcome screen shown when the application starts."""
        C = self.COLORS
        parent.configure(style="Hero.TFrame")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        shell = ttk.Frame(parent, style="Hero.TFrame", padding=(46, 40))
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        title_row = ttk.Frame(shell, style="HeroInner.TFrame")
        title_row.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(title_row, text="Quantum Teleportation State Explorer", style="HeroTitle.TLabel").pack(anchor="w")
        tk.Frame(title_row, background=C["accent"], height=3, width=420).pack(anchor="w", pady=(12, 0))

        ttk.Label(
            shell,
            text=(
                "Interactive design and diagnosis of continuous-variable quantum states using "
                "Wigner-space teleportation fidelity, nonclassicality survival, and configurable noise models."
            ),
            style="HeroSubtitle.TLabel",
            wraplength=1040,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        cards = ttk.Frame(shell, style="HeroInner.TFrame")
        cards.grid(row=2, column=0, sticky="nsew", pady=(36, 24))
        for k in range(3):
            cards.columnconfigure(k, weight=1, uniform="welcome_cards")

        def welcome_card(col: int, title: str, body: str) -> None:
            card = tk.Frame(cards, background=C["card"], highlightthickness=1, highlightbackground=C["border"])
            card.grid(row=0, column=col, sticky="nsew", padx=10, ipady=18)
            tk.Label(
                card,
                text=title,
                background=C["card"],
                foreground=C["accent"],
                font=("Segoe UI", 13, "bold"),
                anchor="w",
            ).pack(fill="x", padx=20, pady=(18, 6))
            tk.Label(
                card,
                text=body,
                background=C["card"],
                foreground=C["muted"],
                font=("Segoe UI", 10),
                justify="left",
                wraplength=360,
                anchor="nw",
            ).pack(fill="both", expand=True, padx=20, pady=(0, 18))

        welcome_card(
            0,
            "1  Configure",
            "Choose the Fock cutoff, target energy, teleportation noise model, constraints, and Wigner visualization style.",
        )
        welcome_card(
            1,
            "2  Optimize",
            "Run multi-start constrained optimization and watch fidelity, nonclassicality, photon statistics, and Wigner functions evolve live.",
        )
        welcome_card(
            2,
            "3  Present",
            "Compare the optimized state with coherent, squeezed, Fock, and cat references; then save the final state for reuse.",
        )

        button_row = ttk.Frame(shell, style="HeroInner.TFrame")
        button_row.grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Button(
            button_row,
            text="Open Settings / Run",
            style="Accent.TButton",
            command=lambda: self.main_notebook.select(self.settings_tab),
        ).pack(side="left", padx=(0, 12))
        ttk.Button(
            button_row,
            text="Read How It Works",
            command=lambda: self.main_notebook.select(self.info_tab),
        ).pack(side="left", padx=(0, 12))
        ttk.Button(
            button_row,
            text="View Wigner Dashboard",
            command=lambda: self.main_notebook.select(getattr(self, "tab_wigner", self.settings_tab)),
        ).pack(side="left")

        contributors = (
            "Project contributors: Yusuf Kayra Gül, Ege Erçiftçi, Arda Çelik, "
            "Muhammad Shahryar Khan, Mustafa Öztürk"
        )
        ttk.Label(shell, text=contributors, style="HeroSmall.TLabel", wraplength=1100, justify="left").grid(
            row=4, column=0, sticky="w", pady=(28, 0)
        )

    def _build_information(self, parent) -> None:
        """Full in-app user manual with readable equation blocks."""
        C = self.COLORS
        parent.configure(style="Hero.TFrame")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        shell = ttk.Frame(parent, style="Hero.TFrame", padding=(24, 18))
        shell.grid(row=0, column=0, sticky="nsew")
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        header = ttk.Frame(shell, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(header, text="How the App Works", style="Title.TLabel").pack(anchor="center")
        ttk.Label(
            header,
            text="User manual for setup, noise models, constraints, objectives, optimization, diagnostics, comparison tests, and exporting.",
            style="Subtitle.TLabel",
        ).pack(anchor="center", pady=(4, 0))

        frame = tk.Frame(shell, background=C["bg"])
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        canvas = tk.Canvas(frame, background=C["bg"], highlightthickness=0, borderwidth=0)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        content = tk.Frame(canvas, background=C["bg"])
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _on_content_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", _on_content_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _info_mousewheel(event):
            delta = int(-1 * (event.delta / 120)) if getattr(event, "delta", 0) else 0
            if delta:
                canvas.yview_scroll(delta, "units")
            return "break"

        def _bind_info_scroll(_event=None):
            canvas.bind_all("<MouseWheel>", _info_mousewheel)

        def _unbind_info_scroll(_event=None):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_info_scroll)
        canvas.bind("<Leave>", _unbind_info_scroll)
        content.bind("<Enter>", _bind_info_scroll)
        content.bind("<Leave>", _unbind_info_scroll)

        def add_card(title: str, subtitle: str = ""):
            card = tk.Frame(content, background=C["card"], highlightthickness=1, highlightbackground=C["border"])
            card.pack(fill="x", padx=8, pady=(0, 14))
            inner = tk.Frame(card, background=C["card"])
            inner.pack(fill="x", padx=24, pady=18)
            tk.Label(
                inner,
                text=title,
                background=C["card"],
                foreground=C["accent"],
                font=("Segoe UI", 15, "bold"),
                anchor="w",
                justify="left",
            ).pack(fill="x", pady=(0, 4))
            if subtitle:
                tk.Label(
                    inner,
                    text=subtitle,
                    background=C["card"],
                    foreground=C["subtle"],
                    font=("Segoe UI", 10, "italic"),
                    anchor="w",
                    justify="left",
                    wraplength=1380,
                ).pack(fill="x", pady=(0, 10))
            return inner

        def add_body(parent_frame, text: str):
            tk.Label(
                parent_frame,
                text=text,
                background=C["card"],
                foreground=C["muted"],
                font=("Segoe UI", 10),
                anchor="w",
                justify="left",
                wraplength=1380,
            ).pack(fill="x", pady=(0, 8))

        def add_bullets(parent_frame, items):
            for item in items:
                tk.Label(
                    parent_frame,
                    text="• " + item,
                    background=C["card"],
                    foreground=C["muted"],
                    font=("Segoe UI", 10),
                    anchor="w",
                    justify="left",
                    wraplength=1340,
                ).pack(fill="x", padx=(16, 0), pady=(0, 5))

        def add_equation(parent_frame, latex: str, height: float = 0.52, fontsize: int = 13):
            eq_box = tk.Frame(parent_frame, background=C["plot"], highlightthickness=1, highlightbackground=C["border"])
            eq_box.pack(fill="x", pady=(2, 10))
            widget = None
            try:
                fig = Figure(figsize=(10.0, height), dpi=100, facecolor=C["plot"])
                ax = fig.add_subplot(111)
                ax.set_facecolor(C["plot"])
                ax.axis("off")
                ax.text(0.5, 0.5, latex, ha="center", va="center", color=C["text"], fontsize=fontsize)
                fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
                eq_canvas = FigureCanvasTkAgg(fig, master=eq_box)
                widget = eq_canvas.get_tk_widget()
                widget.configure(background=C["plot"], highlightthickness=0, borderwidth=0)
                widget.pack(fill="x", expand=True, padx=8, pady=6)
                eq_canvas.draw()
            except Exception:
                if widget is not None:
                    try:
                        widget.destroy()
                    except Exception:
                        pass
                # Fallback: show the equation as clean mathematical text rather
                # than leaving an empty rectangle.
                pretty = latex.strip("$")
                pretty = pretty.replace(r"\rangle", "⟩").replace(r"\langle", "⟨")
                pretty = pretty.replace(r"\psi", "ψ").replace(r"\hat n", "n̂")
                pretty = pretty.replace(r"\mathcal{E}", "𝓔").replace(r"\mathcal N", "𝓝")
                tk.Label(
                    eq_box,
                    text=pretty,
                    background=C["plot"],
                    foreground=C["text"],
                    font=("Consolas", 13),
                    anchor="center",
                    justify="center",
                    wraplength=1050,
                    pady=14,
                ).pack(fill="x", padx=10, pady=6)

        card = add_card("1. Quick start", "The shortest reliable workflow")
        add_bullets(card, [
            "Open Settings / Run.",
            "Choose a computation profile only if you want a speed preset. Press Apply Profile. This changes only numerical/runtime settings.",
            "Edit the physical channel manually: squeezing r, gain, noise switches, phase diffusion, correlated/rotated EPR noise, loss, and drift.",
            "Edit the constraints manually: energy is always active; all other constraints are optional.",
            "Press Start optimization and wait for the final result.",
            "Inspect Wigner functions, photon distribution, optimization history, and the Final results report.",
            "Press Run Comparison to generate reference-state and sweep plots. Press Save Comparison Results only after checking the plots.",
        ])

        card = add_card("2. What is being optimized?", "The mathematical object behind the interface")
        add_body(card, "The app searches over a normalized pure single-mode state written in the finite Fock basis. The cutoff Ncut is numerical; it is not a physical assumption about the state.")
        add_equation(card, r"$|\psi\rangle=\sum_{n=0}^{N_{\rm cut}-1}c_n|n\rangle$", height=0.46, fontsize=13)
        add_body(card, "Two constraints are always active: normalization and target mean photon number. These are the base resource constraints for every run.")
        add_equation(card, r"$\sum_n |c_n|^2=1,\qquad \sum_n n|c_n|^2=N$", height=0.46, fontsize=13)
        add_body(card, "For the standard objective the optimizer maximizes the teleportation fidelity between the input state and the channel output.")
        add_equation(card, r"$F(\psi)=\langle\psi|\mathcal E(|\psi\rangle\langle\psi|)|\psi\rangle$", height=0.48, fontsize=13)

        card = add_card("3. Step 1: choose numerical settings", "Ncut, grid, starts, iterations, and live preview")
        add_bullets(card, [
            "Ncut sets the number of Fock coefficients. Larger Ncut gives a more general search space but makes optimization slower and can reveal cutoff-dependent behavior.",
            "x_max and grid_points set the Wigner phase-space grid. If Wigner norms or plotted structures look clipped, increase x_max or grid_points.",
            "Number of starts controls how many initial states are tried. Increase it when results depend strongly on the initial seed.",
            "Max iterations controls how long SLSQP is allowed to improve each start.",
            "Live Wigner preview is cosmetic for fidelity-only objectives. Use Off or Final only for speed; use Low-resolution for presentations.",
            "Comparison sweep points controls the smoothness of comparison plots. Larger values give smoother curves but increase comparison runtime.",
        ])

        card = add_card("4. Step 2: configure the teleportation channel", "The physical experiment modeled by the app")
        add_body(card, "The channel is represented as a gain transformation plus Gaussian noise, optional phase diffusion, and optional displacement drift. The Gaussian noise covariance is")
        add_equation(card, r"$Y=\begin{pmatrix}\nu_x&\nu_{xp}\\ \nu_{xp}&\nu_p\end{pmatrix}$", height=0.58, fontsize=13)
        add_body(card, "A schematic Wigner-space view is: first rescale by gain, then blur by the Gaussian kernel, then apply phase diffusion and drift if enabled.")
        add_equation(card, r"$W_{\rm out}\approx G_Y * W_{\rm in}^{(g,d)}$", height=0.46, fontsize=13)
        add_bullets(card, [
            "Finite-squeezing noise adds the standard term exp(-2r). Increasing r lowers this part of the noise.",
            "Gain mismatch changes the feed-forward gain g. It can help some noisy channels but it also changes the physical CP condition.",
            "Anisotropic noise adds different extra noise to x and p.",
            "Correlated EPR noise introduces xp covariance. It tilts the Gaussian blur in phase space.",
            "Rotated asymmetric EPR noise adds an elliptical covariance rotated by the selected angle.",
            "Thermal noise and detector noise add isotropic imperfections controlled by strength parameters.",
            "Loss/mode mismatch rescales the effective gain and adds bath noise.",
            "Phase diffusion averages over random phase rotations and can strongly affect phase-sensitive states.",
            "Displacement drift shifts the output Wigner function and models calibration offsets.",
        ])

        card = add_card("5. Step 3: check physicality of the noise", "Complete positivity and numerical covariance stability")
        add_body(card, "For a one-mode Gaussian channel with X=gI, the noise covariance must satisfy the CP condition")
        add_equation(card, r"$\det Y=\nu_x\nu_p-\nu_{xp}^{2}\geq (1-g^2)^2/4$", height=0.48, fontsize=13)
        add_body(card, "Auto-enforce CP/noise stability prevents unphysical or nearly singular covariance matrices. Nearly singular covariance can create artificial jumps in Wigner-grid sweeps, so the app also keeps the effective xp-correlation slightly below one.")

        card = add_card("6. Step 4: choose constraints", "How to restrict the state search")
        add_bullets(card, [
            "Parity constraints restrict the active Fock basis to even or odd photon numbers.",
            "Real-coefficient and real-displacement-axis constraints are useful when you want a simpler phase convention.",
            "Fock range and modular Fock sector constraints restrict which photon numbers may appear.",
            "Zero, fixed, or bounded displacement constraints control the coherent amplitude <a>.",
            "Fluctuation-energy constraints control N-|<a>|², separating coherent displacement energy from nonclassical fluctuation energy.",
            "Photon-variance and Mandel-Q constraints control number statistics.",
            "Quadrature-variance and covariance constraints control squeezing-like second moments.",
            "Overlap constraints can force the optimizer away from coherent/squeezed states or toward cat-like states.",
            "Wigner-negativity constraints force or limit nonclassicality, but they slow the run because Wigner negativity must be evaluated inside the optimization.",
            "High-Fock tail constraints are important when the state starts exploiting the artificial cutoff.",
        ])

        card = add_card("7. Step 5: choose the objective", "What the optimizer tries to maximize")
        add_bullets(card, [
            "Single objective maximizes the configured-channel fidelity.",
            "Robust average maximizes average fidelity over selected r, gain, and phase samples.",
            "Robust worst-case maximizes the minimum fidelity over those samples.",
            "Fidelity + negativity survival maximizes a weighted combination of fidelity and N_out/N_in.",
            "When negativity survival is active, the app rejects states with nearly zero input negativity because the ratio would be undefined.",
        ])
        add_equation(card, r"$R_{\rm surv}=\mathcal N(W_{\rm out})/\mathcal N(W_{\rm in})$", height=0.46, fontsize=13)

        card = add_card("8. Step 6: run optimization", "What happens after pressing Start optimization")
        add_bullets(card, [
            "The app builds feasible seed states: coherent, squeezed, Fock/two-Fock, and broad random seeds when enabled.",
            "SLSQP varies the real and imaginary parts of the coefficients while enforcing normalization, energy, and all active constraints.",
            "The blue history curve shows sampled values; the best-feasible curve shows the best accepted feasible result so far.",
            "A failed start is not automatically a bad run. Multi-start optimization expects some starts to fail or converge to lower local optima.",
            "The final selected state is the best feasible state, not necessarily the final point from the last start.",
        ])

        card = add_card("9. Step 7: read the main result tabs", "What to inspect first")
        add_bullets(card, [
            "Wigner functions: compare the input Wigner function with the teleported output. Use the colormap controls after the run.",
            "History and photon distribution: check whether the optimizer improved smoothly and whether the photon distribution hits the cutoff.",
            "Time-dependent fields: use only as a single-mode quadrature visualization, not as a separate dynamical simulation.",
            "Final results: check feasibility, fidelity, energy, displacement, photon statistics, Wigner negativity, tail probability, and start summary.",
            "Convergence diagnostics: warnings about cutoff dependence should be taken seriously before making physics claims.",
        ])

        card = add_card("10. Step 8: run comparison and validation", "How to use the comparison section")
        add_bullets(card, [
            "Press Run Comparison after optimization finishes.",
            "The Summary Table compares the optimized state with coherent, Fock, squeezed, cat, compass, GKP-like, and feasible two-Fock references.",
            "Allowed=True means the reference satisfies the active constraints. A high-fidelity reference with Allowed=False is not a fair constrained competitor.",
            "Wigner Pairs shows one selected state at a time with matched input/output color limits.",
            "Fidelity Sweeps shows fidelity versus squeezing, detector-efficiency curves, gain sweeps, and asymmetry sweeps.",
            "Noise / Heat Maps shows loss/thermal maps and phase-diffusion sensitivity.",
            "Nonclassicality Sweeps shows Wigner-negativity survival for states with nonzero input negativity.",
            "Increase comparison sweep points for smoother curves. Use fewer points when comparison runtime becomes too long.",
        ])

        card = add_card("11. How to interpret comparison sweeps", "Avoiding common misreadings")
        add_bullets(card, [
            "Fidelity vs squeezing varies r while keeping the other configured noise switches active.",
            "The ideal coherent benchmark is shown only as a reference for the ideal finite-squeezing channel. It is not the configured noisy benchmark when extra noise is active.",
            "Detector-efficiency curves vary both r and eta, so curves may be close together if detector noise is weak compared with other active noise.",
            "Gain sweeps are useful for finding whether unit gain is still optimal under your selected imperfections.",
            "Negativity survival can be N/A for coherent or Gaussian-like states because their input Wigner negativity is zero.",
            "If a survival curve jumps sharply, check whether correlated and rotated covariance settings are pushing the covariance close to singularity.",
        ])

        card = add_card("12. Exporting results", "Saving only when you decide to")
        add_bullets(card, [
            "Save final state exports coefficients, metadata, channel parameters, constraints, and diagnostics.",
            "Run Comparison does not save automatically.",
            "Save Comparison Results exports PNG figures, CSV tables, JSON metadata, sweep data, and Wigner-pair plots.",
            "Export folders are timestamped and placed under the QTSE output directory.",
        ])

        card = add_card("13. Troubleshooting checklist", "What to do when something looks wrong")
        add_bullets(card, [
            "Coherent state has nonzero Wigner negativity: increase grid_points/Ncut or treat values below the displayed tolerance as numerical artifacts.",
            "Plots look clipped or fidelity exceeds expected bounds: increase x_max and grid_points.",
            "Optimization result changes strongly with Ncut: enable tail constraints or run the cutoff scan.",
            "Comparison curves have sudden jumps: inspect xp covariance, effective rho, and whether correlated plus rotated EPR noise is nearly singular.",
            "App feels slow: reduce comparison sweep points, use Fast Search, turn live Wigner preview Off, or disable Wigner-negativity objectives.",
            "A reference beats the optimized state: check the Allowed column. It may violate your constraints or energy target.",
        ])

        card = add_card("14. Recommended presentation workflow", "A clean sequence for a live demo")
        add_bullets(card, [
            "Start with a simple finite-squeezing-only channel and show that coherent is competitive or optimal.",
            "Add one imperfection at a time, such as phase diffusion or rotated asymmetric noise.",
            "Run the optimizer with Balanced or Presentation profile.",
            "Show Wigner functions and photon distribution of the final state.",
            "Run Comparison and show the fidelity sweep and Wigner-pair tab.",
            "Finish with the final report and explain which constraints were active.",
        ])

        def _bind_tree(widget):
            try:
                widget.bind("<MouseWheel>", _info_mousewheel)
            except Exception:
                pass
            for child in widget.winfo_children():
                _bind_tree(child)

        _bind_tree(content)

        btn_row = ttk.Frame(shell, style="HeroInner.TFrame")
        btn_row.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(btn_row, text="Go to Settings / Run", style="Accent.TButton", command=lambda: self.main_notebook.select(self.settings_tab)).pack(side="left", padx=(0, 10))
        ttk.Button(btn_row, text="Go to Comparison Tests", command=lambda: self.main_notebook.select(getattr(self, "tab_comparison", self.settings_tab))).pack(side="left")

    def _build_controls(self, parent) -> None:
        scroll = ScrollFrame(parent)
        scroll.pack(fill="both", expand=True, padx=12, pady=12)
        outer = scroll.inner
        outer.columnconfigure(0, weight=1)

        header = ttk.Frame(outer, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=34, pady=(12, 8))
        ttk.Label(header, text="Fidelity Optimization", style="Title.TLabel").pack(anchor="center")
        ttk.Label(
            header,
            text="Configure the teleportation channel, constraints, convergence diagnostics, and field plots.",
            style="Subtitle.TLabel",
        ).pack(anchor="center", pady=(3, 0))

        f = ttk.Frame(outer, style="Card.TFrame", padding=(24, 18))
        f.grid(row=1, column=0, sticky="new", padx=42, pady=(0, 14))
        self._section_started = False

        self._section(f, "Computation profile and live preview")
        self._combo(
            f,
            "Computation profile",
            "computation_profile",
            "Balanced",
            ["Fast Search", "Balanced", "Presentation", "High Accuracy", "Custom"],
        )
        self._combo(
            f,
            "Live Wigner preview",
            "live_wigner_preview_mode",
            "Low-resolution",
            ["Off", "Final only", "Low-resolution", "Full-resolution"],
        )
        self._num(f, "live preview grid points", "live_wigner_preview_grid_points", 61)
        profile_buttons = ttk.Frame(f, style="Card.TFrame")
        profile_buttons.pack(fill="x", padx=10, pady=(8, 4))
        ttk.Button(profile_buttons, text="Apply Profile", command=self.apply_computation_profile).pack(side="left", padx=(0, 8))
        ttk.Button(profile_buttons, text="Open Information", command=lambda: self.main_notebook.select(self.info_tab)).pack(side="left")
        ttk.Label(
            f,
            text=(
                "Computation profiles are optional numerical templates for speed, presentation, or high-accuracy verification. "
                "All experimental noise and constraint parameters below are manual fields. Profiles only fill numerical/runtime settings once; after applying a profile every box remains editable."
            ),
            style="Small.TLabel",
            wraplength=1100,
        ).pack(anchor="w", padx=10, pady=(2, 8))

        self._section(f, "Hilbert space and grid")
        self._num(f, "Ncut", "Ncut", 10)
        self._num(f, "Target energy N", "target_energy", 1.0)
        self._num(f, "x_max", "x_max", 6.0)
        self._num(f, "Grid points", "grid_points", 81)

        self._section(f, "Optimization")
        self._num(f, "Number of starts", "n_starts", 6)
        self._num(f, "Max iterations", "maxiter", 80)
        self._num(f, "ftol", "ftol", 1e-7)
        self._num(f, "Random seed", "random_seed", 1234)
        self._num(f, "Live update every k callbacks", "live_update_every", 3)

        self._section(f, "Convergence and cutoff stability")
        self._check(f, "Use broad random full-subspace seeds", "use_broad_random_seeds", True)
        self._check(f, "Constrain high-Fock tail probability", "use_tail_probability_constraint", False)
        self._num(f, "tail levels at top of cutoff", "tail_levels", 3)
        self._num(f, "max high-Fock tail probability", "max_tail_probability", 1e-3)
        self._check(f, "Use soft high-Fock tail penalty", "use_tail_penalty", False)
        self._num(f, "tail penalty strength", "tail_penalty_strength", 0.02)
        self._check(f, "Polish best state at end", "use_best_polishing", True)
        self._num(f, "polish max iterations", "polish_maxiter", 80)

        self._section(f, "Convergence diagnostics")
        self._check(f, "Run analytical convergence diagnostics", "run_analytic_convergence_diagnostics", True)
        self._check(f, "Run local optimality probe", "run_local_optimality_probe", False)
        self._num(f, "local probe trials", "local_probe_trials", 5)
        self._num(f, "local probe max iterations", "local_probe_maxiter", 25)
        self._num(f, "local probe perturbation", "local_probe_perturbation", 1e-2)
        self._check(f, "Run automatic Ncut cutoff scan", "run_cutoff_scan", False)
        self._num(f, "cutoff scan Ncut list", "cutoff_scan_values", "8,10,12,14,16,20,24,30")
        self._num(f, "cutoff scan starts", "cutoff_scan_n_starts", 3)
        self._num(f, "cutoff scan max iterations", "cutoff_scan_maxiter", 50)

        self._section(f, "Base teleportation channel")
        self._check(f, "Finite-squeezing noise", "use_finite_squeezing_noise", True)
        self._num(f, "Squeezing r", "r", 0.9)
        self._check(f, "Use gain mismatch", "use_gain_mismatch", False)
        self._num(f, "Gain g if enabled", "gain_if_enabled", 0.75)

        self._section(f, "Noise effects")
        self._check(f, "Use anisotropic noise", "use_anisotropic_noise", False)
        self._num(f, "anisotropic noise x^2", "anisotropic_noise_x2", 0.04)
        self._num(f, "anisotropic noise p^2", "anisotropic_noise_p2", 0.50)

        self._check(f, "Add thermal noise", "use_thermal_noise", False)
        self._num(f, "thermal nbar", "thermal_nbar", 0.5)
        self._num(f, "thermal noise strength", "thermal_noise_strength", 0.02)

        self._check(f, "Add detector noise", "use_detector_noise", False)
        self._num(f, "detector efficiency eta", "detector_efficiency_eta", 0.90)
        self._num(f, "detector noise strength", "detector_noise_strength", 0.05)

        self._check(f, "Add custom extra noise", "use_extra_additive_noise", False)
        self._num(f, "extra x noise", "extra_noise_x2", 0.0)
        self._num(f, "extra p noise", "extra_noise_p2", 0.0)

        self._section(f, "EPR-resource and calibration noise")
        self._check(f, "Correlated non-ideal EPR noise", "use_correlated_epr_noise", False)
        self._num(f, "correlation rho(x,p)", "correlated_epr_rho", 0.30)
        self._check(f, "Rotated asymmetric EPR noise", "use_rotated_asymmetric_epr_noise", False)
        self._num(f, "rotated major variance", "rotated_epr_noise_major2", 0.25)
        self._num(f, "rotated minor variance", "rotated_epr_noise_minor2", 0.02)
        self._num(f, "rotation angle degrees", "rotated_epr_angle_degrees", 30.0)
        self._check(f, "Loss / mode-mismatch to thermal bath", "use_loss_channel_noise", False)
        self._num(f, "loss transmissivity T", "loss_transmissivity", 0.95)
        self._num(f, "loss bath nbar", "loss_thermal_nbar", 0.0)
        self._check(f, "Deterministic displacement drift", "use_displacement_drift", False)
        self._num(f, "drift in x", "drift_x", 0.0)
        self._num(f, "drift in p", "drift_p", 0.0)

        self._section(f, "Phase diffusion")
        self._check(f, "Use phase diffusion", "use_phase_diffusion", False)
        self._num(f, "phase sigma", "phase_sigma", 0.40)
        self._num(f, "phase quadrature points", "n_phase_quad", 11)

        self._section(f, "Basic state constraints")
        self._check(f, "Zero displacement <a>=0", "use_zero_displacement_constraint", False)
        self._check(f, "Even parity only", "use_even_parity_constraint", False)
        self._check(f, "Odd parity only", "use_odd_parity_constraint", False)
        self._check(f, "Real Fock coefficients only", "use_real_coefficients", False)
        self._check(f, "Force displacement along x axis: Im<a>=0", "use_real_displacement_axis", False)
        self._check(f, "Auto-enforce CP bound", "auto_enforce_cp_noise", True)

        self._section(f, "Fock-subspace constraints")
        self._check(f, "Restrict Fock range", "use_fock_range_constraint", False)
        self._num(f, "Fock n min", "fock_min", 0)
        self._num(f, "Fock n max", "fock_max", 9)
        self._check(f, "Use modular Fock sector n = k mod m", "use_modular_fock_constraint", False)
        self._num(f, "modulus m", "modulus_m", 3)
        self._num(f, "residue k", "residue_k", 0)

        self._section(f, "Displacement constraints")
        self._check(f, "Bound |<a>|^2", "use_bounded_displacement", False)
        self._num(f, "max |<a>|^2", "max_displacement_squared", 0.25)
        self._check(f, "Fix <a>", "use_fixed_displacement", False)
        self._num(f, "fixed Re<a>", "fixed_a_real", 0.0)
        self._num(f, "fixed Im<a>", "fixed_a_imag", 0.0)

        self._section(f, "Fluctuation energy constraints")
        self._check(f, "Minimum fluctuation energy", "use_fluctuation_energy_min", False)
        self._num(f, "min N - |<a>|^2", "min_fluctuation_energy", 0.10)
        self._check(f, "Maximum fluctuation energy", "use_fluctuation_energy_max", False)
        self._num(f, "max N - |<a>|^2", "max_fluctuation_energy", 0.50)

        self._section(f, "Photon statistics constraints")
        self._check(f, "Minimum photon-number variance", "use_photon_variance_min", False)
        self._num(f, "min Var(n)", "min_photon_variance", 0.0)
        self._check(f, "Maximum photon-number variance", "use_photon_variance_max", False)
        self._num(f, "max Var(n)", "max_photon_variance", 1.0)
        self._check(f, "Minimum Mandel Q", "use_mandel_q_min", False)
        self._num(f, "min Q", "min_mandel_q", -1.0)
        self._check(f, "Maximum Mandel Q", "use_mandel_q_max", False)
        self._num(f, "max Q", "max_mandel_q", 0.0)

        self._section(f, "Quadrature constraints")
        self._check(f, "Minimum x variance", "use_x_variance_min", False)
        self._num(f, "min Var(x)", "min_x_variance", 0.0)
        self._check(f, "Maximum x variance", "use_x_variance_max", False)
        self._num(f, "max Var(x)", "max_x_variance", 0.5)
        self._check(f, "Minimum p variance", "use_p_variance_min", False)
        self._num(f, "min Var(p)", "min_p_variance", 0.0)
        self._check(f, "Maximum p variance", "use_p_variance_max", False)
        self._num(f, "max Var(p)", "max_p_variance", 0.5)
        self._check(f, "Minimum xp covariance", "use_covariance_min", False)
        self._num(f, "min Cov(x,p)", "min_xp_covariance", -0.5)
        self._check(f, "Maximum xp covariance", "use_covariance_max", False)
        self._num(f, "max Cov(x,p)", "max_xp_covariance", 0.5)

        self._section(f, "Overlap and non-Gaussianity constraints")
        self._check(f, "Maximum overlap with coherent state", "use_coherent_overlap_max", False)
        self._num(f, "max |<coh|psi>|^2", "max_coherent_overlap", 0.80)
        self._check(f, "Maximum overlap with squeezed vacuum", "use_squeezed_overlap_max", False)
        self._num(f, "max |<sq|psi>|^2", "max_squeezed_overlap", 0.80)
        self._check(f, "Minimum overlap with cat reference", "use_cat_overlap_min", False)
        self._num(f, "cat alpha", "cat_alpha", 1.0)
        self._num(f, "cat phase", "cat_phase", 0.0)
        self._num(f, "min |<cat|psi>|^2", "min_cat_overlap", 0.50)

        self._section(f, "Wigner negativity constraints")
        self._check(f, "Minimum Wigner negativity", "use_wigner_negativity_min", False)
        self._num(f, "min negativity", "min_wigner_negativity", 0.01)
        self._check(f, "Maximum Wigner negativity", "use_wigner_negativity_max", False)
        self._num(f, "max negativity", "max_wigner_negativity", 0.50)

        self._section(f, "Fidelity + negativity survival objective")
        self._check(f, "Optimize fidelity and negativity survival", "use_negativity_survival_objective", False)
        self._num(f, "fidelity weight", "negativity_survival_fidelity_weight", 0.50)
        self._num(f, "survival-ratio weight", "negativity_survival_ratio_weight", 0.50)
        self._num(f, "minimum input negativity", "min_input_negativity_for_survival", 1e-4)
        self._num(f, "survival-ratio clip", "survival_ratio_clip", 1.0)

        self._section(f, "Robust objective")
        self._check(f, "Maximize average fidelity over samples", "use_robust_average_objective", False)
        self._check(f, "Maximize worst-case fidelity over samples", "use_robust_worstcase_objective", False)
        self._num(f, "robust r min", "robust_r_min", 0.75)
        self._num(f, "robust r max", "robust_r_max", 1.05)
        self._num(f, "robust r samples", "robust_r_samples", 3)
        self._num(f, "robust gain min", "robust_gain_min", 0.90)
        self._num(f, "robust gain max", "robust_gain_max", 1.10)
        self._num(f, "robust gain samples", "robust_gain_samples", 3)
        self._num(f, "robust phase max", "robust_phase_max", 0.20)
        self._num(f, "robust phase samples", "robust_phase_samples", 1)

        self._section(f, "Time-dependent field plots")
        self._num(f, "field angular frequency omega", "field_omega", 1.0)
        self._num(f, "number of periods", "field_periods", 3.0)
        self._num(f, "time points", "field_time_points", 400)
        self._num(f, "electric-field scale E0", "field_E_scale", 1.0)
        self._num(f, "magnetic-field scale B0", "field_B_scale", 1.0)
        self._check(f, "show quantum uncertainty bands", "field_show_uncertainty", True)

        self._section(f, "Presentation and Wigner color style")
        self._combo(
            f,
            "Wigner colormap",
            "wigner_colormap",
            "RdBu_r",
            ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "RdBu_r", "seismic", "coolwarm", "twilight"],
        )
        self._combo(
            f,
            "Wigner color scale",
            "wigner_color_scale",
            "symmetric",
            ["symmetric", "auto", "positive"],
        )

        self._section(f, "Comparison plot settings")
        self._num(f, "comparison sweep points", "comparison_sweep_points", 31)
        ttk.Label(
            f,
            text=(
                "This controls the finesse of comparison plots such as fidelity vs squeezing, detector-efficiency curves, gain sweeps, "
                "phase-diffusion sensitivity, and heat-map resolution. Larger values make smoother plots but the comparison run takes longer."
            ),
            style="Small.TLabel",
            wraplength=1100,
        ).pack(anchor="w", padx=10, pady=(2, 8))

        button_frame = ttk.Frame(f)
        button_frame.pack(fill="x", padx=10, pady=12)

        self.start_button = ttk.Button(button_frame, text="Start optimization", command=self.start, style="Accent.TButton")
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(5, 5))

        self.save_button = ttk.Button(button_frame, text="Save final state", command=self.save_final_state, state="disabled")
        self.save_button.pack(side="left", fill="x", expand=True, padx=(5, 0))

        self.status = tk.Text(
            f,
            height=10,
            width=80,
            wrap="word",
            borderwidth=0,
            relief="flat",
            background=self.COLORS["plot"],
            foreground=self.COLORS["muted"],
            insertbackground=self.COLORS["text"],
            font=("Consolas", 9),
            padx=12,
            pady=10,
        )
        self.status.pack(fill="x", padx=10, pady=(0, 10))
        self._log("Ready. Press Start optimization.")

    def _build_plots(self, parent_notebook) -> None:
        # Plot/result tabs are now siblings of the Settings tab, so the plots
        # are not squeezed by a permanent parameter sidebar.
        self.metric_labels: List[ttk.Label] = []

        def add_metric_bar(tab: ttk.Frame) -> ttk.Frame:
            # Compact multi-line information card.  The previous version used one
            # extremely long line, which did not fit even on fullscreen monitors.
            top = ttk.Frame(tab, style="Card.TFrame", padding=(12, 8))
            top.pack(fill="x", padx=14, pady=(12, 8))
            label = ttk.Label(
                top,
                text="Fidelity: -- | Objective: -- | Energy: -- | <a>: -- | Parity: --",
                style="Metric.TLabel",
                justify="left",
                anchor="w",
            )
            label.pack(fill="x", anchor="w")
            self.metric_labels.append(label)

            def _resize_metric_label(event, lbl=label):
                # Keep wrapping tied to the current window width so resizing and
                # fullscreen mode do not produce horizontal clipping.
                lbl.configure(wraplength=max(500, int(event.width) - 24))

            top.bind("<Configure>", _resize_metric_label)

            body = ttk.Frame(tab)
            body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
            return body

        tab_wigner = ttk.Frame(parent_notebook)
        tab_history = ttk.Frame(parent_notebook)
        tab_fields = ttk.Frame(parent_notebook)
        tab_results = ttk.Frame(parent_notebook)
        tab_comparison = ttk.Frame(parent_notebook)
        self.tab_wigner = tab_wigner
        self.tab_history = tab_history
        self.tab_fields = tab_fields
        self.tab_results = tab_results
        self.tab_comparison = tab_comparison

        parent_notebook.add(tab_wigner, text="Wigner functions")
        parent_notebook.add(tab_history, text="History and photon distribution")
        parent_notebook.add(tab_fields, text="Time-dependent fields")
        parent_notebook.add(tab_results, text="Final results")
        parent_notebook.add(tab_comparison, text="Comparison & Validation")

        # ------------------------------------------------------------------
        # Wigner tab: equal-size left/right plot panels.
        # ------------------------------------------------------------------
        wigner_body = add_metric_bar(tab_wigner)
        wigner_body.rowconfigure(0, weight=0)
        wigner_body.rowconfigure(1, weight=1, uniform="wigner_row")
        wigner_body.columnconfigure(0, weight=1, uniform="wigner_col")
        wigner_body.columnconfigure(1, weight=1, uniform="wigner_col")

        wigner_toolbar = ttk.Frame(wigner_body, style="Toolbar.TFrame", padding=(12, 8))
        wigner_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(wigner_toolbar, text="Wigner style", style="Toolbar.TLabel").pack(side="left", padx=(0, 12))
        ttk.Label(wigner_toolbar, text="Colormap", style="Toolbar.TLabel").pack(side="left", padx=(0, 6))
        self.wigner_runtime_cmap = ModernDropdown(
            wigner_toolbar,
            textvariable=self.vars["wigner_colormap"],
            values=["RdBu_r", "seismic", "coolwarm", "twilight", "viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
            width=14,
            colors=self.COLORS,
        )
        self.wigner_runtime_cmap.pack(side="left", padx=(0, 14))
        ttk.Label(wigner_toolbar, text="Scale", style="Toolbar.TLabel").pack(side="left", padx=(0, 6))
        self.wigner_runtime_scale = ModernDropdown(
            wigner_toolbar,
            textvariable=self.vars["wigner_color_scale"],
            values=["symmetric", "auto", "positive"],
            width=13,
            colors=self.COLORS,
        )
        self.wigner_runtime_scale.pack(side="left", padx=(0, 14))
        ttk.Button(wigner_toolbar, text="Apply Wigner Style", command=self.refresh_wigner_style).pack(side="left")
        self.wigner_runtime_cmap.bind("<<ComboboxSelected>>", lambda event: self.refresh_wigner_style())
        self.wigner_runtime_scale.bind("<<ComboboxSelected>>", lambda event: self.refresh_wigner_style())

        self.fig_in = Figure(figsize=(6, 5.4), dpi=100, facecolor=self.COLORS["card"])
        self.ax_in = self.fig_in.add_axes([0.11, 0.13, 0.71, 0.76])
        self.cax_in = self.fig_in.add_axes([0.86, 0.13, 0.035, 0.76])
        panel_in = ttk.Frame(wigner_body, style="PlotCard.TFrame", padding=4)
        panel_in.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        panel_in.rowconfigure(0, weight=1)
        panel_in.columnconfigure(0, weight=1)
        self.canvas_in = FigureCanvasTkAgg(self.fig_in, master=panel_in)
        self.canvas_in.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.fig_out = Figure(figsize=(6, 5.4), dpi=100, facecolor=self.COLORS["card"])
        self.ax_out = self.fig_out.add_axes([0.11, 0.13, 0.71, 0.76])
        self.cax_out = self.fig_out.add_axes([0.86, 0.13, 0.035, 0.76])
        panel_out = ttk.Frame(wigner_body, style="PlotCard.TFrame", padding=4)
        panel_out.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        panel_out.rowconfigure(0, weight=1)
        panel_out.columnconfigure(0, weight=1)
        self.canvas_out = FigureCanvasTkAgg(self.fig_out, master=panel_out)
        self.canvas_out.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ------------------------------------------------------------------
        # Diagnostics tab: equal-size history and photon panels.
        # ------------------------------------------------------------------
        history_body = add_metric_bar(tab_history)
        history_body.rowconfigure(0, weight=1, uniform="diag_row")
        history_body.columnconfigure(0, weight=1, uniform="diag_col")
        history_body.columnconfigure(1, weight=1, uniform="diag_col")

        self.fig_hist = Figure(figsize=(6, 4.8), dpi=100, facecolor=self.COLORS["card"])
        self.ax_hist = self.fig_hist.add_subplot(111)
        panel_hist = ttk.Frame(history_body, style="PlotCard.TFrame", padding=4)
        panel_hist.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        panel_hist.rowconfigure(0, weight=1)
        panel_hist.columnconfigure(0, weight=1)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=panel_hist)
        self.canvas_hist.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.fig_photon = Figure(figsize=(6, 4.8), dpi=100, facecolor=self.COLORS["card"])
        self.ax_photon = self.fig_photon.add_subplot(111)
        panel_photon = ttk.Frame(history_body, style="PlotCard.TFrame", padding=4)
        panel_photon.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        panel_photon.rowconfigure(0, weight=1)
        panel_photon.columnconfigure(0, weight=1)
        self.canvas_photon = FigureCanvasTkAgg(self.fig_photon, master=panel_photon)
        self.canvas_photon.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ------------------------------------------------------------------
        # Field tab: equal-size electric and magnetic field panels.
        # ------------------------------------------------------------------
        field_body = add_metric_bar(tab_fields)
        field_body.rowconfigure(0, weight=1, uniform="field_row")
        field_body.columnconfigure(0, weight=1, uniform="field_col")
        field_body.columnconfigure(1, weight=1, uniform="field_col")

        self.fig_E = Figure(figsize=(6, 4.8), dpi=100, facecolor=self.COLORS["card"])
        self.ax_E = self.fig_E.add_subplot(111)
        panel_E = ttk.Frame(field_body, style="PlotCard.TFrame", padding=4)
        panel_E.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        panel_E.rowconfigure(0, weight=1)
        panel_E.columnconfigure(0, weight=1)
        self.canvas_E = FigureCanvasTkAgg(self.fig_E, master=panel_E)
        self.canvas_E.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.fig_B = Figure(figsize=(6, 4.8), dpi=100, facecolor=self.COLORS["card"])
        self.ax_B = self.fig_B.add_subplot(111)
        panel_B = ttk.Frame(field_body, style="PlotCard.TFrame", padding=4)
        panel_B.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        panel_B.rowconfigure(0, weight=1)
        panel_B.columnconfigure(0, weight=1)
        self.canvas_B = FigureCanvasTkAgg(self.fig_B, master=panel_B)
        self.canvas_B.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Results tab.
        result_frame = ttk.Frame(tab_results, style="Card.TFrame", padding=8)
        result_frame.pack(fill="both", expand=True, padx=14, pady=14)
        self.result_text = tk.Text(
            result_frame,
            wrap="word",
            borderwidth=0,
            relief="flat",
            font=("Consolas", 10),
            background=self.COLORS["plot"],
            foreground=self.COLORS["muted"],
            padx=18,
            pady=16,
        )
        self.result_text.pack(fill="both", expand=True)
        self.result_text.tag_configure("report_title", font=("Segoe UI", 17, "bold"), foreground=self.COLORS["text"], spacing3=8)
        self.result_text.tag_configure("report_section", font=("Segoe UI", 12, "bold"), foreground=self.COLORS["accent"], spacing1=8, spacing3=4)
        self.result_text.tag_configure("report_note", font=("Segoe UI", 10), foreground=self.COLORS["muted"], spacing3=8)
        self.result_text.tag_configure("report_ok", foreground=self.COLORS["ok"], font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("report_warn", foreground=self.COLORS["warn"], font=("Consolas", 10, "bold"))
        self.result_text.tag_configure("report_bad", foreground=self.COLORS["bad"], font=("Consolas", 10, "bold"))

        self._build_comparison_tab(tab_comparison)

        self._draw_empty()

    def _build_comparison_tab(self, tab: ttk.Frame) -> None:
        """Build the integrated comparison and validation section."""
        C = self.COLORS
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(tab, style="Card.TFrame", padding=(14, 10))
        toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        toolbar.columnconfigure(8, weight=1)

        ttk.Label(toolbar, text="Comparison & Validation Tests", style="Toolbar.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 14))
        self.compare_button = ttk.Button(
            toolbar,
            text="Run Comparison",
            style="Accent.TButton",
            command=self.start_comparison,
            state="disabled",
        )
        self.compare_button.grid(row=0, column=1, padx=(0, 10))

        self.save_comparison_button = ttk.Button(
            toolbar,
            text="Save Comparison Results",
            command=self.save_comparison_results,
            state="disabled",
        )
        self.save_comparison_button.grid(row=0, column=2, padx=(0, 10))

        self.open_comparison_button = ttk.Button(
            toolbar,
            text="Open Results Folder",
            command=self.open_comparison_folder,
            state="disabled",
        )
        self.open_comparison_button.grid(row=0, column=3, padx=(0, 10))

        ttk.Label(toolbar, text="Wigner state", style="Toolbar.TLabel").grid(row=0, column=4, sticky="e", padx=(8, 6))
        self.comparison_state_var = tk.StringVar(value="")
        self.comparison_state_combo = ModernDropdown(
            toolbar,
            textvariable=self.comparison_state_var,
            values=[],
            width=28,
            colors=self.COLORS,
        )
        self.comparison_state_combo.grid(row=0, column=5, sticky="ew", padx=(0, 10))
        self.comparison_state_combo.bind("<<ComboboxSelected>>", lambda event: self.update_comparison_wigner_pair())

        ttk.Label(toolbar, text="Sweep points", style="Toolbar.TLabel").grid(row=0, column=6, sticky="e", padx=(8, 6))
        self.comparison_sweep_entry = tk.Entry(
            toolbar,
            textvariable=self.vars.get("comparison_sweep_points"),
            width=6,
            relief="flat",
            borderwidth=1,
            background=C["plot"],
            foreground=C["text"],
            insertbackground=C["text"],
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["accent"],
            font=("Segoe UI", 10),
        )
        self.comparison_sweep_entry.grid(row=0, column=7, sticky="w", padx=(0, 12))

        self.comparison_folder_label = ttk.Label(
            toolbar,
            text="Run an optimization first, then press Run Comparison.",
            style="Toolbar.TLabel",
        )
        self.comparison_folder_label.grid(row=0, column=8, sticky="e")

        comp_nb = ttk.Notebook(tab)
        self.comparison_notebook = comp_nb
        comp_nb.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        tab_summary = ttk.Frame(comp_nb)
        tab_curves = ttk.Frame(comp_nb)
        tab_wigner = ttk.Frame(comp_nb)
        tab_fidelity_sweeps = ttk.Frame(comp_nb)
        tab_noise_sweeps = ttk.Frame(comp_nb)
        tab_nonclassical = ttk.Frame(comp_nb)
        comp_nb.add(tab_summary, text="Summary Table")
        comp_nb.add(tab_curves, text="Metrics")
        comp_nb.add(tab_wigner, text="Wigner Pairs")
        comp_nb.add(tab_fidelity_sweeps, text="Fidelity Sweeps")
        comp_nb.add(tab_noise_sweeps, text="Noise / Heat Maps")
        comp_nb.add(tab_nonclassical, text="Nonclassicality Sweeps")

        # Summary tab.
        summary_frame = ttk.Frame(tab_summary, style="Card.TFrame", padding=8)
        summary_frame.pack(fill="both", expand=True, padx=10, pady=10)
        summary_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        self.comparison_text = tk.Text(
            summary_frame,
            wrap="word",
            borderwidth=0,
            relief="flat",
            font=("Consolas", 10),
            background=C["plot"],
            foreground=C["muted"],
            insertbackground=C["text"],
            padx=16,
            pady=14,
        )
        text_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=self.comparison_text.yview)
        self.comparison_text.configure(yscrollcommand=text_scroll.set)
        self.comparison_text.grid(row=0, column=0, sticky="nsew")
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.comparison_text.tag_configure("title", foreground=C["text"], font=("Segoe UI", 15, "bold"), spacing3=8)
        self.comparison_text.tag_configure("section", foreground=C["accent"], font=("Segoe UI", 11, "bold"), spacing1=6, spacing3=3)
        self.comparison_text.tag_configure("warn", foreground=C["warn"], font=("Consolas", 10, "bold"))
        self.comparison_text.insert("1.0", "Comparison results will appear here after you run an optimization and press Run Comparison.")
        self.comparison_text.configure(state="disabled")

        def make_scroll(parent_tab):
            scroll = ScrollFrame(parent_tab)
            scroll.pack(fill="both", expand=True)
            return scroll.inner

        def add_large_plot(parent_frame, fig_attr: str, ax_attr: str, canvas_attr: str, title_hint: str = ""):
            panel = ttk.Frame(parent_frame, style="PlotCard.TFrame", padding=8)
            panel.pack(fill="x", expand=False, padx=12, pady=12)
            panel.columnconfigure(0, weight=1)
            fig = Figure(figsize=(12.5, 5.8), dpi=100, facecolor=C["card"])
            ax = fig.add_subplot(111)
            setattr(self, fig_attr, fig)
            setattr(self, ax_attr, ax)
            canvas = FigureCanvasTkAgg(fig, master=panel)
            widget = canvas.get_tk_widget()
            widget.configure(background=C["card"], highlightthickness=0, borderwidth=0)
            widget.grid(row=0, column=0, sticky="ew")
            setattr(self, canvas_attr, canvas)
            if title_hint:
                ax.set_title(title_hint)
                self._style_axis(ax)
                canvas.draw_idle()

        # Metrics tab: one large plot per row.
        metrics_inner = make_scroll(tab_curves)
        add_large_plot(metrics_inner, "fig_comp_fidelity", "ax_comp_fidelity", "canvas_comp_fidelity", "Teleportation fidelity")
        add_large_plot(metrics_inner, "fig_comp_neg", "ax_comp_neg", "canvas_comp_neg", "Input/output Wigner negativity")
        add_large_plot(metrics_inner, "fig_comp_survival", "ax_comp_survival", "canvas_comp_survival", "Negativity survival ratio")
        add_large_plot(metrics_inner, "fig_comp_overlap", "ax_comp_overlap", "canvas_comp_overlap", "Wigner input-output shape overlap")

        # Wigner pair tab: show one selected state at a time.
        # This keeps the pair large and readable for presentation.
        tab_wigner.rowconfigure(0, weight=1)
        tab_wigner.columnconfigure(0, weight=1)
        self.wigner_pairs_scroll = ScrollFrame(tab_wigner)
        self.wigner_pairs_scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.wigner_pairs_inner = self.wigner_pairs_scroll.inner
        self.comparison_wigner_pair_figures: List[Tuple[Figure, Figure, FigureCanvasTkAgg, FigureCanvasTkAgg]] = []

        # Fidelity sweeps tab: stacked large plots from the previous comparison code.
        fidelity_inner = make_scroll(tab_fidelity_sweeps)
        add_large_plot(fidelity_inner, "fig_sweep_r", "ax_sweep_r", "canvas_sweep_r", "Fidelity vs squeezing parameter")
        add_large_plot(fidelity_inner, "fig_sweep_detector", "ax_sweep_detector", "canvas_sweep_detector", "Detector efficiency effect")
        add_large_plot(fidelity_inner, "fig_sweep_gain", "ax_sweep_gain", "canvas_sweep_gain", "Fidelity vs feed-forward gain")
        add_large_plot(fidelity_inner, "fig_sweep_asymmetry", "ax_sweep_asymmetry", "canvas_sweep_asymmetry", "Asymmetric quadrature noise")

        # Noise and heat-map tab.
        noise_inner = make_scroll(tab_noise_sweeps)
        add_large_plot(noise_inner, "fig_heat_loss", "ax_heat_loss", "canvas_heat_loss", "Loss and thermal-noise heat map")
        add_large_plot(noise_inner, "fig_phase_sensitivity", "ax_phase_sensitivity", "canvas_phase_sensitivity", "Phase diffusion sensitivity")

        # Nonclassicality tab.
        nonclassical_inner = make_scroll(tab_nonclassical)
        add_large_plot(nonclassical_inner, "fig_survival_r", "ax_survival_r", "canvas_survival_r", "Wigner negativity survival vs squeezing")

        self.comparison_result: Optional[Dict[str, object]] = None
        self.comparison_output_folder: Optional[str] = None
        self._draw_comparison_empty()

    def _section(self, parent, text: str) -> None:
        if getattr(self, "_section_started", False):
            ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=(14, 8))
        self._section_started = True
        ttk.Label(parent, text=text, style="Section.TLabel").pack(anchor="w", padx=10, pady=(2, 5))

    def _register_manual_template_trace(self, key: str, var: tk.Variable) -> None:
        """Compatibility no-op for old profile tracing code.

        Earlier development versions attached Tk variable traces to runtime
        fields so that manual edits changed the profile label to Custom.  On
        some Windows/Tk builds those traces made entries feel locked after
        Apply Profile.  Computation profiles are now deliberately one-shot
        templates: pressing Apply Profile sets numerical values once, and no
        trace remains attached to any parameter afterwards.
        """
        return

    def _handle_manual_template_edit(self, key: str) -> None:
        """No-op retained for compatibility with older callback references."""
        return

    def _num(self, parent, label: str, key: str, default) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", padx=10, pady=3)
        ttk.Label(row, text=label, width=42, style="Card.TLabel").pack(side="left")
        var = tk.StringVar(value=str(default))
        # Use a plain tk.Entry instead of ttk.Entry for numerical fields.
        # On some Windows/Tk themes, ttk.Entry can visually retain a readonly-like
        # state after bulk StringVar updates.  tk.Entry keeps editing behavior
        # fully under our control and remains editable after Apply Profile.
        C = self.COLORS
        entry = tk.Entry(
            row,
            textvariable=var,
            width=18,
            background=C["plot"],
            foreground=C["text"],
            insertbackground=C["text"],
            selectbackground=C["accent2"],
            selectforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["accent"],
            font=("Segoe UI", 10),
        )
        entry.pack(side="right")
        self.vars[key] = var
        self.entry_widgets[key] = entry
        self._register_manual_template_trace(key, var)

    def _combo(self, parent, label: str, key: str, default: str, values: Sequence[str]) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", padx=10, pady=3)
        ttk.Label(row, text=label, width=42, style="Card.TLabel").pack(side="left")
        var = tk.StringVar(value=str(default))
        width = max(34, min(54, max([len(str(v)) for v in values] + [len(str(default))]) + 6))
        combo = ModernDropdown(row, textvariable=var, values=list(values), width=width, colors=self.COLORS)
        combo.pack(side="right")
        self.vars[key] = var
        self.combo_widgets[key] = combo
        self._last_combo_values[key] = str(default)

        def _selected(_event=None, combo_key=key, combo_var=var):
            # Dropdowns only change their own variable. They never automatically
            # overwrite numerical, noise, or constraint fields. Use Apply Profile
            # when a computation template should be applied intentionally.
            value = str(combo_var.get())
            self._last_combo_values[combo_key] = value

        combo.bind("<<ComboboxSelected>>", _selected)
        self._register_manual_template_trace(key, var)

    def _check(self, parent, label: str, key: str, default: bool) -> None:
        var = tk.BooleanVar(value=default)
        widget = ttk.Checkbutton(parent, text=label, variable=var)
        widget.pack(anchor="w", padx=10, pady=2)
        self.vars[key] = var
        self.check_widgets[key] = widget
        self._register_manual_template_trace(key, var)

    def _cfg(self) -> UserConfig:
        def i(key: str) -> int:
            return int(float(self.vars[key].get()))

        def x(key: str) -> float:
            return float(self.vars[key].get())

        def b(key: str) -> bool:
            return bool(self.vars[key].get())

        def s(key: str) -> str:
            return str(self.vars[key].get())

        cfg = UserConfig(
            Ncut=i("Ncut"),
            target_energy=x("target_energy"),
            x_max=x("x_max"),
            grid_points=i("grid_points"),

            n_starts=i("n_starts"),
            maxiter=i("maxiter"),
            ftol=x("ftol"),
            random_seed=i("random_seed"),
            live_update_every=max(1, i("live_update_every")),

            use_broad_random_seeds=b("use_broad_random_seeds"),
            use_tail_probability_constraint=b("use_tail_probability_constraint"),
            tail_levels=i("tail_levels"),
            max_tail_probability=x("max_tail_probability"),
            use_tail_penalty=b("use_tail_penalty"),
            tail_penalty_strength=x("tail_penalty_strength"),
            use_best_polishing=b("use_best_polishing"),
            polish_maxiter=i("polish_maxiter"),

            run_analytic_convergence_diagnostics=b("run_analytic_convergence_diagnostics"),
            run_local_optimality_probe=b("run_local_optimality_probe"),
            local_probe_trials=i("local_probe_trials"),
            local_probe_maxiter=i("local_probe_maxiter"),
            local_probe_perturbation=x("local_probe_perturbation"),
            run_cutoff_scan=b("run_cutoff_scan"),
            cutoff_scan_values=s("cutoff_scan_values"),
            cutoff_scan_n_starts=i("cutoff_scan_n_starts"),
            cutoff_scan_maxiter=i("cutoff_scan_maxiter"),

            use_finite_squeezing_noise=b("use_finite_squeezing_noise"),
            r=x("r"),
            use_gain_mismatch=b("use_gain_mismatch"),
            gain_if_enabled=x("gain_if_enabled"),

            use_anisotropic_noise=b("use_anisotropic_noise"),
            anisotropic_noise_x2=x("anisotropic_noise_x2"),
            anisotropic_noise_p2=x("anisotropic_noise_p2"),

            use_thermal_noise=b("use_thermal_noise"),
            thermal_nbar=x("thermal_nbar"),
            thermal_noise_strength=x("thermal_noise_strength"),

            use_detector_noise=b("use_detector_noise"),
            detector_efficiency_eta=x("detector_efficiency_eta"),
            detector_noise_strength=x("detector_noise_strength"),

            use_extra_additive_noise=b("use_extra_additive_noise"),
            extra_noise_x2=x("extra_noise_x2"),
            extra_noise_p2=x("extra_noise_p2"),

            use_correlated_epr_noise=b("use_correlated_epr_noise"),
            correlated_epr_rho=x("correlated_epr_rho"),
            use_rotated_asymmetric_epr_noise=b("use_rotated_asymmetric_epr_noise"),
            rotated_epr_noise_major2=x("rotated_epr_noise_major2"),
            rotated_epr_noise_minor2=x("rotated_epr_noise_minor2"),
            rotated_epr_angle_degrees=x("rotated_epr_angle_degrees"),
            use_loss_channel_noise=b("use_loss_channel_noise"),
            loss_transmissivity=x("loss_transmissivity"),
            loss_thermal_nbar=x("loss_thermal_nbar"),
            use_displacement_drift=b("use_displacement_drift"),
            drift_x=x("drift_x"),
            drift_p=x("drift_p"),

            use_phase_diffusion=b("use_phase_diffusion"),
            phase_sigma=x("phase_sigma"),
            n_phase_quad=i("n_phase_quad"),

            use_zero_displacement_constraint=b("use_zero_displacement_constraint"),
            use_even_parity_constraint=b("use_even_parity_constraint"),
            use_odd_parity_constraint=b("use_odd_parity_constraint"),
            use_real_coefficients=b("use_real_coefficients"),
            use_real_displacement_axis=b("use_real_displacement_axis"),
            auto_enforce_cp_noise=b("auto_enforce_cp_noise"),

            use_fock_range_constraint=b("use_fock_range_constraint"),
            fock_min=i("fock_min"),
            fock_max=i("fock_max"),
            use_modular_fock_constraint=b("use_modular_fock_constraint"),
            modulus_m=i("modulus_m"),
            residue_k=i("residue_k"),

            use_bounded_displacement=b("use_bounded_displacement"),
            max_displacement_squared=x("max_displacement_squared"),
            use_fixed_displacement=b("use_fixed_displacement"),
            fixed_a_real=x("fixed_a_real"),
            fixed_a_imag=x("fixed_a_imag"),

            use_fluctuation_energy_min=b("use_fluctuation_energy_min"),
            min_fluctuation_energy=x("min_fluctuation_energy"),
            use_fluctuation_energy_max=b("use_fluctuation_energy_max"),
            max_fluctuation_energy=x("max_fluctuation_energy"),

            use_photon_variance_min=b("use_photon_variance_min"),
            min_photon_variance=x("min_photon_variance"),
            use_photon_variance_max=b("use_photon_variance_max"),
            max_photon_variance=x("max_photon_variance"),

            use_mandel_q_min=b("use_mandel_q_min"),
            min_mandel_q=x("min_mandel_q"),
            use_mandel_q_max=b("use_mandel_q_max"),
            max_mandel_q=x("max_mandel_q"),

            use_x_variance_min=b("use_x_variance_min"),
            min_x_variance=x("min_x_variance"),
            use_x_variance_max=b("use_x_variance_max"),
            max_x_variance=x("max_x_variance"),

            use_p_variance_min=b("use_p_variance_min"),
            min_p_variance=x("min_p_variance"),
            use_p_variance_max=b("use_p_variance_max"),
            max_p_variance=x("max_p_variance"),

            use_covariance_min=b("use_covariance_min"),
            min_xp_covariance=x("min_xp_covariance"),
            use_covariance_max=b("use_covariance_max"),
            max_xp_covariance=x("max_xp_covariance"),

            use_coherent_overlap_max=b("use_coherent_overlap_max"),
            max_coherent_overlap=x("max_coherent_overlap"),
            use_squeezed_overlap_max=b("use_squeezed_overlap_max"),
            max_squeezed_overlap=x("max_squeezed_overlap"),
            use_cat_overlap_min=b("use_cat_overlap_min"),
            cat_alpha=x("cat_alpha"),
            cat_phase=x("cat_phase"),
            min_cat_overlap=x("min_cat_overlap"),

            use_wigner_negativity_min=b("use_wigner_negativity_min"),
            min_wigner_negativity=x("min_wigner_negativity"),
            use_wigner_negativity_max=b("use_wigner_negativity_max"),
            max_wigner_negativity=x("max_wigner_negativity"),

            use_negativity_survival_objective=b("use_negativity_survival_objective"),
            negativity_survival_fidelity_weight=x("negativity_survival_fidelity_weight"),
            negativity_survival_ratio_weight=x("negativity_survival_ratio_weight"),
            min_input_negativity_for_survival=x("min_input_negativity_for_survival"),
            survival_ratio_clip=x("survival_ratio_clip"),

            use_robust_average_objective=b("use_robust_average_objective"),
            use_robust_worstcase_objective=b("use_robust_worstcase_objective"),
            robust_r_min=x("robust_r_min"),
            robust_r_max=x("robust_r_max"),
            robust_r_samples=i("robust_r_samples"),
            robust_gain_min=x("robust_gain_min"),
            robust_gain_max=x("robust_gain_max"),
            robust_gain_samples=i("robust_gain_samples"),
            robust_phase_max=x("robust_phase_max"),
            robust_phase_samples=i("robust_phase_samples"),

            field_omega=x("field_omega"),
            field_periods=x("field_periods"),
            field_time_points=i("field_time_points"),
            field_E_scale=x("field_E_scale"),
            field_B_scale=x("field_B_scale"),
            field_show_uncertainty=b("field_show_uncertainty"),
            computation_profile=str(self.vars["computation_profile"].get()),
            live_wigner_preview_mode=str(self.vars["live_wigner_preview_mode"].get()),
            live_wigner_preview_grid_points=i("live_wigner_preview_grid_points"),
            wigner_colormap=str(self.vars["wigner_colormap"].get()),
            wigner_color_scale=str(self.vars["wigner_color_scale"].get()),
            comparison_sweep_points=i("comparison_sweep_points"),
        )

        if cfg.Ncut < 2:
            raise ValueError("Ncut must be at least 2.")
        if cfg.grid_points < 21:
            raise ValueError("Grid points must be at least 21.")
        if cfg.n_starts < 1:
            raise ValueError("Number of starts must be at least 1.")
        if cfg.maxiter < 1:
            raise ValueError("Max iterations must be at least 1.")
        if cfg.tail_levels < 1:
            raise ValueError("Tail levels must be at least 1.")
        if cfg.max_tail_probability < 0.0:
            raise ValueError("Maximum tail probability must be non-negative.")
        if cfg.tail_penalty_strength < 0.0:
            raise ValueError("Tail penalty strength must be non-negative.")
        if cfg.polish_maxiter < 1:
            raise ValueError("Polish max iterations must be at least 1.")
        if cfg.local_probe_trials < 0:
            raise ValueError("Local probe trials cannot be negative.")
        if cfg.local_probe_maxiter < 1:
            raise ValueError("Local probe max iterations must be at least 1.")
        if cfg.local_probe_perturbation < 0.0:
            raise ValueError("Local probe perturbation must be non-negative.")
        if cfg.cutoff_scan_n_starts < 1:
            raise ValueError("Cutoff scan starts must be at least 1.")
        if cfg.cutoff_scan_maxiter < 1:
            raise ValueError("Cutoff scan max iterations must be at least 1.")
        if not parse_cutoff_values(cfg.cutoff_scan_values, cfg.Ncut):
            raise ValueError("Cutoff scan Ncut list contains no valid values.")
        if cfg.use_correlated_epr_noise and abs(cfg.correlated_epr_rho) >= 1.0:
            raise ValueError("Correlated EPR rho must satisfy -1 < rho < 1.")
        if cfg.rotated_epr_noise_major2 < 0.0 or cfg.rotated_epr_noise_minor2 < 0.0:
            raise ValueError("Rotated EPR noise variances must be non-negative.")
        if cfg.loss_transmissivity <= 0.0 or cfg.loss_transmissivity > 1.0:
            raise ValueError("Loss transmissivity must satisfy 0 < T <= 1.")
        if cfg.loss_thermal_nbar < 0.0:
            raise ValueError("Loss bath nbar must be non-negative.")
        if cfg.n_phase_quad < 3:
            raise ValueError("Phase quadrature points must be at least 3.")
        if cfg.n_phase_quad % 2 == 0:
            cfg.n_phase_quad += 1
        if cfg.fock_max < cfg.fock_min:
            raise ValueError("Fock n max must be >= Fock n min.")
        if cfg.use_even_parity_constraint and cfg.use_odd_parity_constraint:
            raise ValueError("Even parity and odd parity cannot both be enabled.")
        if cfg.use_fixed_displacement and cfg.use_zero_displacement_constraint:
            raise ValueError("Fixed displacement and zero displacement cannot both be enabled.")
        if cfg.use_robust_average_objective and cfg.use_robust_worstcase_objective:
            raise ValueError("Choose robust average or robust worst-case, not both.")
        if cfg.robust_r_samples < 1 or cfg.robust_gain_samples < 1 or cfg.robust_phase_samples < 1:
            raise ValueError("Robust sample counts must be at least 1.")
        if cfg.negativity_survival_fidelity_weight < 0.0 or cfg.negativity_survival_ratio_weight < 0.0:
            raise ValueError("Negativity-survival objective weights must be non-negative.")
        if cfg.negativity_survival_fidelity_weight + cfg.negativity_survival_ratio_weight <= 0.0:
            raise ValueError("At least one negativity-survival objective weight must be positive.")
        if cfg.min_input_negativity_for_survival <= 0.0:
            raise ValueError("Minimum input negativity for survival objective must be positive.")
        if cfg.survival_ratio_clip <= 0.0:
            raise ValueError("Survival-ratio clip must be positive.")
        if cfg.field_omega <= 0:
            raise ValueError("Field angular frequency omega must be positive.")
        if cfg.field_periods <= 0:
            raise ValueError("Number of field periods must be positive.")
        if cfg.field_time_points < 20:
            raise ValueError("Field time points must be at least 20.")
        if cfg.live_wigner_preview_grid_points < 21:
            raise ValueError("Live Wigner preview grid points must be at least 21.")
        if cfg.live_wigner_preview_grid_points % 2 == 0:
            cfg.live_wigner_preview_grid_points += 1
        if cfg.comparison_sweep_points < 5:
            raise ValueError("Comparison sweep points must be at least 5.")
        if cfg.comparison_sweep_points > 151:
            raise ValueError("Comparison sweep points must be at most 151 to keep comparison runtime manageable.")

        return cfg

    def _restore_manual_parameter_editing(self) -> None:
        """Force all manual inputs back to editable state after profile application.

        This is intentionally defensive.  Computation profiles must never lock
        the experimental configuration.  Some themed Tk widgets can retain a
        transient disabled/readonly state after bulk variable updates, so this
        method explicitly restores normal editing for every Entry, Checkbutton,
        and custom dropdown.
        """
        for widget in getattr(self, "entry_widgets", {}).values():
            try:
                widget.configure(state="normal")
            except Exception:
                pass
        for widget in getattr(self, "check_widgets", {}).values():
            try:
                widget.configure(state="normal")
            except Exception:
                pass
        for widget in getattr(self, "combo_widgets", {}).values():
            try:
                widget.configure(state="normal")
            except Exception:
                pass
        # Do not force focus to the root window here.  On some Windows/Tk
        # builds that made the next Entry click appear unresponsive until the
        # window was minimized/restored.  The user should keep normal focus flow.

    def _set_var(self, key: str, value) -> bool:
        """Set a Tk variable safely from profile buttons.

        Returns True when the key exists.  Some ttk widgets can be visually
        stubborn in dark themes unless idle events are flushed after a bulk
        update, so the caller updates the GUI after all variables are changed.
        """
        var = self.vars.get(key)
        if var is None:
            return False

        # Tk BooleanVar accepts Python booleans reliably.  For numeric entries
        # and readonly comboboxes, setting the associated StringVar updates the
        # visible widget.
        try:
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value))
            if hasattr(self, "_last_combo_values") and key in self._last_combo_values:
                self._last_combo_values[key] = str(value)
            return True
        except Exception:
            # Fallback for unusual Tk variable implementations.
            try:
                var.set(value)
                return True
            except Exception:
                return False

    def _close_all_dropdown_popups(self) -> None:
        """Close any custom dropdown popups and release stale grabs/focus captures."""
        for widget in getattr(self, "combo_widgets", {}).values():
            try:
                if hasattr(widget, "_close_popup"):
                    widget._close_popup()
            except Exception:
                pass
        try:
            grab = self.root.grab_current()
            if grab is not None:
                grab.grab_release()
        except Exception:
            pass

    def _after_profile_repaint(self) -> None:
        """Asynchronous repaint after applying a computation profile.

        This emulates the minimize/restore redraw that was manually fixing the
        settings page on Windows, but without moving or hiding the window.  The
        important part is that the button callback returns first; then Tk handles
        focus, redraw, and geometry events normally.
        """
        self._restore_manual_parameter_editing()
        self._schedule_layout_refresh(delay=1)
        self.root.after(60, self._restore_manual_parameter_editing)
        self.root.after(80, lambda: self._schedule_layout_refresh(delay=1))
        self.root.after(220, lambda: self._schedule_layout_refresh(delay=1))

    def _install_layout_stabilizers(self) -> None:
        """Install resize/tab-change hooks so hidden Matplotlib canvases stay aligned."""
        def schedule(_event=None):
            self._schedule_layout_refresh(delay=80)

        for widget in (getattr(self, "main_notebook", None), getattr(self, "comparison_notebook", None), self.root):
            if widget is None:
                continue
            try:
                widget.bind("<Configure>", schedule, add="+")
            except TypeError:
                widget.bind("<Configure>", schedule)
            except Exception:
                pass
            try:
                widget.bind("<<NotebookTabChanged>>", schedule, add="+")
            except TypeError:
                try:
                    widget.bind("<<NotebookTabChanged>>", schedule)
                except Exception:
                    pass
            except Exception:
                pass

    def _warmup_tabs_for_layout(self) -> None:
        """Realize all notebook pages once so fullscreen layout is correct immediately."""
        notebooks = [getattr(self, "main_notebook", None), getattr(self, "comparison_notebook", None)]
        for nb in notebooks:
            if nb is None:
                continue
            try:
                current = nb.select()
                for tab_id in nb.tabs():
                    nb.select(tab_id)
                    nb.update_idletasks()
                if current:
                    nb.select(current)
            except Exception:
                pass
        self._refresh_scrollframes()
        self._refresh_matplotlib_canvases()

    def _schedule_layout_refresh(self, delay: int = 80) -> None:
        """Debounced layout refresh for fullscreen, restore, and tab-change events."""
        try:
            old = getattr(self, "_layout_refresh_after_id", None)
            if old is not None:
                self.root.after_cancel(old)
        except Exception:
            pass
        try:
            self._layout_refresh_after_id = self.root.after(int(delay), self._refresh_all_layouts)
        except Exception:
            self._layout_refresh_after_id = None

    def _refresh_all_layouts(self) -> None:
        """Refresh scroll regions and Matplotlib canvases after geometry changes."""
        self._layout_refresh_after_id = None
        self._restore_manual_parameter_editing()
        self._refresh_scrollframes()
        self._refresh_matplotlib_canvases()

    def _refresh_scrollframes(self) -> None:
        def walk(widget):
            try:
                if isinstance(widget, ScrollFrame):
                    widget.refresh_layout()
                for child in widget.winfo_children():
                    walk(child)
            except Exception:
                pass
        walk(self.root)

    def _refresh_matplotlib_canvases(self) -> None:
        """Force visible Matplotlib canvases to match their Tk widget size.

        Hidden notebook pages can report tiny sizes until first visited.  We avoid
        drawing those, but once a page is visible or warmed up this keeps the
        figure geometry synchronized with fullscreen/resized windows.
        """
        seen = set()
        canvases = []
        for value in self.__dict__.values():
            if isinstance(value, FigureCanvasTkAgg):
                canvases.append(value)
        for pack in getattr(self, "comparison_wigner_pair_figures", []) or []:
            for value in pack:
                if isinstance(value, FigureCanvasTkAgg):
                    canvases.append(value)
        for canvas in canvases:
            if id(canvas) in seen:
                continue
            seen.add(id(canvas))
            try:
                widget = canvas.get_tk_widget()
                widget.update_idletasks()
                w = int(widget.winfo_width())
                h = int(widget.winfo_height())
                if w > 60 and h > 60:
                    fig = canvas.figure
                    dpi = float(fig.dpi) if fig.dpi else 100.0
                    fig.set_size_inches(max(w, 100) / dpi, max(h, 100) / dpi, forward=False)
                    canvas.draw_idle()
            except Exception:
                pass

    def apply_computation_profile(self) -> None:
        """Apply numerical settings for fast search, balanced work, presentation, or verification."""
        profile_var = self.vars.get("computation_profile")
        profile = str(profile_var.get()) if profile_var is not None else "Balanced"
        self._set_var("computation_profile", profile)
        profiles = {
            "Fast Search": {
                "Ncut": 8,
                "grid_points": 51,
                "n_starts": 3,
                "maxiter": 45,
                "live_update_every": 8,
                "live_wigner_preview_mode": "Off",
                "live_wigner_preview_grid_points": 41,
                "run_local_optimality_probe": False,
                "run_cutoff_scan": False,
                "comparison_sweep_points": 21,
            },
            "Balanced": {
                "Ncut": 10,
                "grid_points": 81,
                "n_starts": 6,
                "maxiter": 80,
                "live_update_every": 5,
                "live_wigner_preview_mode": "Low-resolution",
                "live_wigner_preview_grid_points": 61,
                "run_local_optimality_probe": False,
                "run_cutoff_scan": False,
                "comparison_sweep_points": 31,
            },
            "Presentation": {
                "Ncut": 10,
                "grid_points": 81,
                "n_starts": 4,
                "maxiter": 60,
                "live_update_every": 2,
                "live_wigner_preview_mode": "Low-resolution",
                "live_wigner_preview_grid_points": 61,
                "run_local_optimality_probe": False,
                "run_cutoff_scan": False,
                "comparison_sweep_points": 41,
            },
            "High Accuracy": {
                "Ncut": 16,
                "grid_points": 121,
                "n_starts": 10,
                "maxiter": 150,
                "live_update_every": 10,
                "live_wigner_preview_mode": "Final only",
                "live_wigner_preview_grid_points": 81,
                "run_local_optimality_probe": True,
                "run_cutoff_scan": False,
                "comparison_sweep_points": 61,
            },
        }
        if profile == "Custom":
            self._log("Custom profile selected. Existing numerical settings were left unchanged.")
            return

        # One-shot template application.  Do not run a blocking root.update()
        # or update_idletasks() inside this button callback.  On some Windows/Tk
        # builds that made the settings page feel frozen until the window was
        # minimized/restored.  We set variables, return control to Tk, and let a
        # scheduled repaint/focus cleanup finish the visual refresh.
        self._applying_template = True
        try:
            for key, value in profiles.get(profile, {}).items():
                self._set_var(key, value)
        finally:
            self._applying_template = False

        self._restore_manual_parameter_editing()
        self._close_all_dropdown_popups()
        self._log(f"Applied computation profile: {profile}. Manual editing remains enabled for all parameters.")
        self._after_profile_repaint()

    def apply_experiment_preset(self) -> None:
        """Backward-compatibility no-op for older saved UI callbacks."""
        self._log("Edit noise and constraint fields manually. Only computation profiles are available as templates.")

    def start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Already running", "Optimization is already running.")
            return

        try:
            cfg = self._cfg()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.stop_event.clear()
        self.out_queue = queue.Queue()
        self.last_best = None

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.save_button.configure(state="disabled")
        if hasattr(self, "compare_button"):
            self.compare_button.configure(state="disabled")
        if hasattr(self, "open_comparison_button"):
            self.open_comparison_button.configure(state="disabled")
        if hasattr(self, "save_comparison_button"):
            self.save_comparison_button.configure(state="disabled")
        self.comparison_result = None if hasattr(self, "comparison_result") else None
        self.comparison_output_folder = None if hasattr(self, "comparison_output_folder") else None
        self.result_text.delete("1.0", "end")
        self._draw_empty()
        if hasattr(self, "_draw_comparison_empty"):
            self._draw_comparison_empty()
            self._write_comparison_text("Comparison results will appear here after the new optimization finishes.")
        self._log("Starting optimization...")

        self.worker = threading.Thread(
            target=optimize_worker,
            args=(cfg, self.out_queue, self.stop_event),
            daemon=True,
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._log("Stop requested. Waiting for current numerical step to finish...")

    def _poll_queue(self) -> None:
        handled = 0
        max_per_poll = 6
        try:
            while handled < max_per_poll:
                msg = self.out_queue.get_nowait()
                self._handle(msg)
                handled += 1
        except queue.Empty:
            pass
        self.root.after(100 if handled else 250, self._poll_queue)

    def _handle(self, msg: Dict[str, object]) -> None:
        typ = msg.get("type")

        if typ == "status":
            self._log(str(msg.get("text", "")))

        elif typ == "live":
            self._update_live(msg)

        elif typ == "done":
            self.last_best = msg["best"]

            final_history = list(self.last_best.get("history", []))
            # History is a displayed fidelity history, not the internal selection
            # score. The final point should therefore be the final base-channel
            # fidelity, while the metric bar still reports the selection score.
            final_display_fidelity = float(self.last_best.get("fidelity", self.last_best["objective_fidelity"]))
            if len(final_history) == 0 or abs(float(final_history[-1]) - final_display_fidelity) > 1e-12:
                final_history.append(final_display_fidelity)

            final_feasible_history = list(self.last_best.get("feasible_history", []))
            if len(final_feasible_history) == 0:
                final_feasible_history = final_history.copy()
            if len(final_feasible_history) < len(final_history):
                last_feasible = final_feasible_history[-1] if len(final_feasible_history) > 0 else np.nan
                final_feasible_history.extend([last_feasible] * (len(final_history) - len(final_feasible_history)))
            if len(final_feasible_history) == 0 or not np.isfinite(final_feasible_history[-1]) or abs(float(final_feasible_history[-1]) - final_display_fidelity) > 1e-12:
                final_feasible_history.append(max(float(final_feasible_history[-1]) if len(final_feasible_history) > 0 and np.isfinite(final_feasible_history[-1]) else -np.inf, final_display_fidelity))

            self._update_live(
                {
                    "label": "FINAL BEST FEASIBLE STATE",
                    "fidelity": self.last_best["fidelity"],
                    "objective_fidelity": self.last_best["objective_fidelity"],
                    "selection_score": self.last_best.get("selection_score", self.last_best["objective_fidelity"]),
                    "tail_probability": self.last_best.get("tail_probability", 0.0),
                    "feasible": self.last_best.get("feasible", True),
                    "constraint_violation": self.last_best.get("constraint_violation", 0.0),
                    "energy": self.last_best["energy"],
                    "a": self.last_best["a"],
                    "parity": self.last_best["parity"],
                    "n_variance": self.last_best["n_variance"],
                    "mandel_q": self.last_best["mandel_q"],
                    "x_var": self.last_best["x_var"],
                    "p_var": self.last_best["p_var"],
                    "wigner_negativity": self.last_best["wigner_negativity"],
                    "output_wigner_negativity": self.last_best.get("output_wigner_negativity", 0.0),
                    "negativity_survival_ratio": self.last_best.get("negativity_survival_ratio", 0.0),
                    "objective_negativity_survival_ratio": self.last_best.get("objective_negativity_survival_ratio", self.last_best.get("negativity_survival_ratio", 0.0)),
                    "coeffs": self.last_best["coeffs"],
                    "W_in": self.last_best["W_in"],
                    "W_out": self.last_best["W_out"],
                    "xvec": self.last_best["xvec"],
                    "pvec": self.last_best["pvec"],
                    "has_wigner": True,
                    "preview_mode": "Final high-resolution",
                    "history": final_history,
                    "feasible_history": final_feasible_history,
                }
            )

            self._log("Optimization complete. Final plots show the best state.")
            self._show_final(self.last_best)
            self._schedule_layout_refresh(delay=100)
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.save_button.configure(state="normal")
            if hasattr(self, "compare_button"):
                self.compare_button.configure(state="normal")

        elif typ == "comparison_status":
            self._log(str(msg.get("text", "")))
            if hasattr(self, "comparison_folder_label"):
                self.comparison_folder_label.configure(text=str(msg.get("text", "Running comparison...")))

        elif typ == "comparison_done":
            self._log("Comparison tests complete.")
            # Drawing all comparison figures can be expensive; defer it to the
            # Tk idle loop so queue polling returns promptly.
            result = msg["result"]
            self.root.after_idle(lambda r=result: self._display_comparison_results(r))

        elif typ == "comparison_error":
            text = str(msg.get("text", "Unknown comparison error."))
            self._log("COMPARISON ERROR: " + text)
            messagebox.showerror("Comparison error", text)
            if hasattr(self, "compare_button"):
                self.compare_button.configure(state="normal" if self.last_best is not None else "disabled")
            if hasattr(self, "save_comparison_button"):
                self.save_comparison_button.configure(state="disabled")

        elif typ == "stopped":
            self._log(str(msg.get("text", "Stopped.")))
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            if hasattr(self, "compare_button"):
                self.compare_button.configure(state="normal" if self.last_best is not None else "disabled")

        elif typ == "error":
            text = str(msg.get("text", "Unknown error."))
            self._log("ERROR: " + text)
            messagebox.showerror("Optimization error", text)
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            if hasattr(self, "compare_button"):
                self.compare_button.configure(state="disabled")

    def start_comparison(self) -> None:
        """Run the integrated comparison tests for the final optimized state."""
        if self.last_best is None:
            messagebox.showinfo("No optimized state", "Run an optimization before starting comparison tests.")
            return
        if getattr(self, "comparison_thread", None) is not None and self.comparison_thread.is_alive():
            messagebox.showinfo("Already running", "Comparison tests are already running.")
            return

        best_for_comparison = dict(self.last_best)
        try:
            sweep_points = int(float(self.vars.get("comparison_sweep_points").get()))
            sweep_points = int(np.clip(sweep_points, 5, 151))
            best_for_comparison["config"] = replace(
                self.last_best["config"],
                comparison_sweep_points=sweep_points,
            )
        except Exception:
            best_for_comparison["config"] = self.last_best["config"]

        self.compare_button.configure(state="disabled")
        if hasattr(self, "save_comparison_button"):
            self.save_comparison_button.configure(state="disabled")
        self.open_comparison_button.configure(state="disabled")
        self.comparison_folder_label.configure(text="Running comparison tests...")
        self._write_comparison_text("Running comparison tests with the configured channel. Please wait...")
        self.comparison_thread = threading.Thread(
            target=comparison_worker,
            args=(best_for_comparison, self.out_queue),
            daemon=True,
        )
        self.comparison_thread.start()

    def save_comparison_results(self) -> None:
        result = getattr(self, "comparison_result", None)
        if not result:
            messagebox.showinfo("No comparison results", "Run comparison tests before saving results.")
            return
        try:
            folder = export_comparison_result_payload(result, DEFAULT_COMPARISON_EXPORT_ROOT)
            self.comparison_output_folder = folder
            self._save_visible_comparison_figures(result)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.comparison_folder_label.configure(text="Saved to: " + folder)
        self.open_comparison_button.configure(state="normal")
        if hasattr(self, "save_comparison_button"):
            self.save_comparison_button.configure(state="normal")
        self._write_comparison_text(self._format_comparison_report(result))
        self._log("Saved comparison results to: " + folder)
        messagebox.showinfo("Comparison saved", "Comparison results saved successfully.\n\n" + folder)

    def open_comparison_folder(self) -> None:
        folder = getattr(self, "comparison_output_folder", None)
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo("No folder", "No comparison output folder is available yet.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception:
            messagebox.showinfo("Results folder", folder)

    def _draw_comparison_empty(self) -> None:
        for ax, canvas, title in [
            (self.ax_comp_fidelity, self.canvas_comp_fidelity, "Comparison fidelity"),
            (self.ax_comp_neg, self.canvas_comp_neg, "Input/output Wigner negativity"),
            (self.ax_comp_survival, self.canvas_comp_survival, "Negativity survival ratio"),
            (self.ax_comp_overlap, self.canvas_comp_overlap, "Wigner shape overlap"),
        ]:
            ax.clear()
            ax.set_title(title)
            ax.text(0.5, 0.5, "Run comparison after optimization", ha="center", va="center", transform=ax.transAxes, color=self.COLORS["muted"])
            self._style_axis(ax)
            canvas.draw_idle()
        self._draw_wigner_pair_placeholder("Run comparison to display input/output Wigner pairs for all compared states.")

        for ax, canvas, title in [
            (getattr(self, "ax_sweep_r", None), getattr(self, "canvas_sweep_r", None), "Fidelity vs squeezing"),
            (getattr(self, "ax_sweep_detector", None), getattr(self, "canvas_sweep_detector", None), "Detector efficiency sweep"),
            (getattr(self, "ax_sweep_gain", None), getattr(self, "canvas_sweep_gain", None), "Fidelity vs gain"),
            (getattr(self, "ax_sweep_asymmetry", None), getattr(self, "canvas_sweep_asymmetry", None), "Asymmetric noise sweep"),
            (getattr(self, "ax_heat_loss", None), getattr(self, "canvas_heat_loss", None), "Loss/thermal heat map"),
            (getattr(self, "ax_phase_sensitivity", None), getattr(self, "canvas_phase_sensitivity", None), "Phase diffusion sensitivity"),
            (getattr(self, "ax_survival_r", None), getattr(self, "canvas_survival_r", None), "Negativity survival vs squeezing"),
        ]:
            if ax is None or canvas is None:
                continue
            ax.clear()
            ax.set_title(title)
            ax.text(0.5, 0.5, "Run comparison after optimization", ha="center", va="center", transform=ax.transAxes, color=self.COLORS["muted"])
            self._style_axis(ax)
            canvas.draw_idle()

    def _write_comparison_text(self, content: str) -> None:
        self.comparison_text.configure(state="normal")
        self.comparison_text.delete("1.0", "end")
        self.comparison_text.insert("1.0", content)
        self.comparison_text.configure(state="disabled")

    def _display_comparison_results(self, result: Dict[str, object]) -> None:
        self.comparison_result = result
        self.comparison_output_folder = str(result.get("output_folder", ""))
        rows = list(result.get("rows", []))
        names = [str(row["state"]) for row in rows]
        self.comparison_state_combo.configure(values=names)
        if names:
            self.comparison_state_var.set(names[0])

        self.comparison_folder_label.configure(text="Comparison ready. Press Save Comparison Results to export files.")
        if hasattr(self, "save_comparison_button"):
            self.save_comparison_button.configure(state="normal")
        self.open_comparison_button.configure(state="disabled")
        self.compare_button.configure(state="normal")

        self._write_comparison_text(self._format_comparison_report(result))
        self._plot_comparison_metric_figures(result)
        self._plot_comparison_sweep_figures(result)
        self._populate_selected_comparison_wigner_pair(result)
        self.main_notebook.select(self.tab_comparison)
        self._schedule_layout_refresh(delay=100)

    def _format_comparison_report(self, result: Dict[str, object]) -> str:
        rows = list(result.get("rows", []))
        cfg: UserConfig = self.last_best["config"] if self.last_best is not None else self._cfg()
        params: ChannelParams = self.last_best["params"] if self.last_best is not None else build_channel(cfg)

        lines: List[str] = []
        lines.append("Comparison & Validation Results\n")
        lines.append("================================\n\n")
        lines.append("All states below were evaluated under the same configured teleportation channel used for the optimized state.\n")
        lines.append("The coherent and squeezed references are generated at the target mean photon number when possible.\n")
        lines.append("For states with zero input Wigner negativity, negativity survival is reported as N/A.\n")
        lines.append(f"Negativity values below {NEGATIVITY_DISPLAY_ZERO_TOL:.1e}, or coherent-state numerical artifacts, are displayed as zero.\n\n")

        lines.append("Active channel summary\n")
        lines.append("----------------------\n")
        lines.append(f"gain = {params.gain}\n")
        lines.append(f"noise_x2 = {params.noise_x2}\n")
        lines.append(f"noise_p2 = {params.noise_p2}\n")
        lines.append(f"noise_xp = {getattr(params, 'noise_xp', 0.0)}\n")
        lines.append(f"phase_sigma = {params.phase_sigma}\n")
        lines.append(f"drift = ({getattr(params, 'displacement_x', 0.0)}, {getattr(params, 'displacement_p', 0.0)})\n")
        lines.append(f"target energy = {cfg.target_energy}\n\n")

        lines.append("State metrics\n")
        lines.append("-------------\n")
        header = (
            f"{'state':24s} {'F':>9s} {'E':>9s} {'NegIn':>10s} {'NegOut':>10s} "
            f"{'Survival':>12s} {'W-overlap':>10s} {'allowed':>8s}\n"
        )
        lines.append(header)
        lines.append("-" * (len(header) - 1) + "\n")
        for row in rows:
            survival_label = row.get("negativity_survival_label", "N/A")
            if isinstance(survival_label, float):
                survival_label = f"{survival_label:.5f}"
            lines.append(
                f"{str(row['state'])[:24]:24s} "
                f"{float(row['fidelity']):9.5f} "
                f"{float(row['energy']):9.5f} "
                f"{float(row['input_negativity']):10.5g} "
                f"{float(row['output_negativity']):10.5g} "
                f"{str(survival_label)[:12]:>12s} "
                f"{float(row['wigner_shape_overlap']):10.5f} "
                f"{str(bool(row['allowed_by_active_constraints'])):>8s}\n"
            )

        if rows:
            best_row = max(rows, key=lambda r: float(r["fidelity"]))
            lines.append("\nInterpretation\n")
            lines.append("--------------\n")
            lines.append(f"Highest fidelity in this comparison: {best_row['state']} with F = {float(best_row['fidelity']):.6f}.\n")
            opt_rows = [r for r in rows if r["state"] == "Optimized state"]
            coh_rows = [r for r in rows if str(r["state"]).lower().startswith("coherent")]
            if opt_rows and coh_rows:
                diff = float(opt_rows[0]["fidelity"]) - float(coh_rows[0]["fidelity"])
                lines.append(f"Optimized minus coherent fidelity difference: {diff:+.6f}.\n")
            lines.append("Check the 'allowed' column before interpreting a reference as a fair constrained competitor.\n")

        lines.append("\nExport\n")
        lines.append("------\n")
        folder = str(result.get("output_folder", ""))
        csv_path = str(result.get("csv_path", ""))
        if folder:
            lines.append(f"Results folder: {folder}\n")
            lines.append(f"CSV table: {csv_path}\n")
        else:
            lines.append("Results have not been saved yet. Press Save Comparison Results to export PNG, CSV, and JSON files.\n")
        return "".join(lines)

    def _plot_comparison_metric_figures(self, result: Dict[str, object]) -> None:
        rows = list(result.get("rows", []))
        names = [str(r["state"]) for r in rows]
        x = np.arange(len(names))

        def finish_bar(ax, title: str, ylabel: str):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right")
            self._style_axis(ax)

        self.ax_comp_fidelity.clear()
        self.ax_comp_fidelity.bar(x, [float(r["fidelity"]) for r in rows])
        finish_bar(self.ax_comp_fidelity, "Teleportation fidelity", "F")
        self.fig_comp_fidelity.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.28)
        self.canvas_comp_fidelity.draw_idle()

        self.ax_comp_neg.clear()
        width = 0.38
        neg_in_values = np.asarray([float(r["input_negativity"]) for r in rows], dtype=float)
        neg_out_values = np.asarray([float(r["output_negativity"]) for r in rows], dtype=float)
        self.ax_comp_neg.bar(x - width / 2, neg_in_values, width=width, label="input")
        self.ax_comp_neg.bar(x + width / 2, neg_out_values, width=width, label="output")
        if neg_in_values.size == 0 or np.nanmax(np.r_[neg_in_values, neg_out_values, 0.0]) <= 0.0:
            self.ax_comp_neg.text(0.5, 0.52, "All plotted Wigner negativities are zero within numerical tolerance.", ha="center", va="center", transform=self.ax_comp_neg.transAxes, color=self.COLORS["muted"])
            self.ax_comp_neg.set_ylim(0.0, 1.0)
        finish_bar(self.ax_comp_neg, "Input/output Wigner negativity", r"$\mathcal{N}$")
        leg = self.ax_comp_neg.legend(loc="best")
        if leg is not None:
            leg.get_frame().set_facecolor(self.COLORS["card"])
            leg.get_frame().set_edgecolor(self.COLORS["border"])
            for txt in leg.get_texts():
                txt.set_color(self.COLORS["text"])
        self.fig_comp_neg.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.28)
        self.canvas_comp_neg.draw_idle()

        self.ax_comp_survival.clear()
        surv = np.array([float(r["negativity_survival"]) if np.isfinite(float(r["negativity_survival"])) else np.nan for r in rows])
        mask = np.isfinite(surv)
        if np.any(mask):
            self.ax_comp_survival.bar(x[mask], surv[mask])
        else:
            self.ax_comp_survival.text(0.5, 0.52, "Survival ratio is N/A for all states with zero input Wigner negativity.", ha="center", va="center", transform=self.ax_comp_survival.transAxes, color=self.COLORS["muted"])
            self.ax_comp_survival.set_ylim(0.0, 1.0)
        for xi, finite in zip(x, mask):
            if not finite:
                self.ax_comp_survival.text(xi, 0.02, "N/A", ha="center", va="bottom", color=self.COLORS["warn"], rotation=90)
        finish_bar(self.ax_comp_survival, "Negativity survival ratio", r"$\mathcal{N}_{out}/\mathcal{N}_{in}$")
        self.fig_comp_survival.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.28)
        self.canvas_comp_survival.draw_idle()

        self.ax_comp_overlap.clear()
        self.ax_comp_overlap.bar(x, [float(r["wigner_shape_overlap"]) for r in rows])
        finish_bar(self.ax_comp_overlap, "Wigner input-output shape overlap", "cosine overlap")
        self.fig_comp_overlap.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.28)
        self.canvas_comp_overlap.draw_idle()


    def _style_legend(self, leg) -> None:
        if leg is None:
            return
        leg.get_frame().set_facecolor(self.COLORS["card"])
        leg.get_frame().set_edgecolor(self.COLORS["border"])
        leg.get_frame().set_alpha(0.95)
        for txt in leg.get_texts():
            txt.set_color(self.COLORS["text"])

    def _plot_line_sweep(self, ax, canvas, sweep: Dict[str, object]) -> None:
        ax.clear()
        x = np.asarray(sweep.get("x", []), dtype=float)
        plotted = 0
        for label, values in dict(sweep.get("series", {})).items():
            y = np.asarray(values, dtype=float)
            y[~np.isfinite(y)] = np.nan
            if len(x) == len(y) and np.any(np.isfinite(y)):
                ax.plot(x, y, marker="o", markersize=3, linewidth=1.7, label=str(label))
                plotted += 1
        if "benchmark" in sweep:
            bench = np.asarray(sweep.get("benchmark", []), dtype=float)
            if len(bench) == len(x):
                # Avoid duplicating the coherent series when the configured
                # coherent benchmark is already present as a normal curve.
                duplicate = False
                for values in dict(sweep.get("series", {})).values():
                    y_cmp = np.asarray(values, dtype=float)
                    if len(y_cmp) == len(bench):
                        mask = np.isfinite(y_cmp) & np.isfinite(bench)
                        if np.any(mask) and np.allclose(y_cmp[mask], bench[mask], rtol=1e-8, atol=1e-10):
                            duplicate = True
                            break
                if not duplicate:
                    ax.plot(x, bench, linestyle="--", linewidth=1.6, label="configured coherent benchmark")
                    plotted += 1
        if "ideal_coherent_benchmark" in sweep:
            ideal = np.asarray(sweep.get("ideal_coherent_benchmark", []), dtype=float)
            if len(ideal) == len(x):
                ax.plot(x, ideal, linestyle=":", linewidth=1.8, label="ideal finite-squeezing coherent benchmark")
                plotted += 1
        if plotted == 0:
            ax.text(0.5, 0.52, "No finite data for this sweep under the current state/channel configuration.", ha="center", va="center", transform=ax.transAxes, color=self.COLORS["muted"])
        ax.set_title(str(sweep.get("title", "Sweep")))
        ax.set_xlabel(str(sweep.get("x_label", "x")))
        ax.set_ylabel(str(sweep.get("y_label", "metric")))
        leg = ax.legend(loc="best", fontsize=8) if plotted else None
        self._style_legend(leg)
        self._style_axis(ax)
        ax.figure.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16)
        canvas.draw_idle()

    def _plot_heatmap_sweep(self, ax, canvas, sweep: Dict[str, object]) -> None:
        # Remove old colorbar axes before drawing a new heat map. Otherwise
        # repeated comparison runs progressively shrink the main axes.
        for extra_ax in list(ax.figure.axes):
            if extra_ax is not ax:
                try:
                    ax.figure.delaxes(extra_ax)
                except Exception:
                    pass
        ax.clear()
        x = np.asarray(sweep.get("x", []), dtype=float)
        y = np.asarray(sweep.get("y", []), dtype=float)
        Z = np.asarray(sweep.get("map", []), dtype=float)
        if Z.size and len(x) and len(y):
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
            )
            try:
                cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
                cb.ax.yaxis.label.set_color(self.COLORS["text"])
                cb.ax.tick_params(colors=self.COLORS["muted"])
                cb.set_label("fidelity")
                cb.outline.set_edgecolor(self.COLORS["border"])
            except Exception:
                pass
        else:
            ax.text(0.5, 0.5, "No heat-map data", ha="center", va="center", transform=ax.transAxes, color=self.COLORS["muted"])
        ax.set_title(str(sweep.get("title", "Heat map")))
        ax.set_xlabel(str(sweep.get("x_label", "x")))
        ax.set_ylabel(str(sweep.get("y_label", "y")))
        self._style_axis(ax)
        ax.figure.subplots_adjust(left=0.10, right=0.92, top=0.86, bottom=0.16)
        canvas.draw_idle()

    def _plot_comparison_sweep_figures(self, result: Dict[str, object]) -> None:
        sweeps = dict(result.get("sweeps", {}))
        mapping = [
            ("fidelity_vs_squeezing", self.ax_sweep_r, self.canvas_sweep_r, "line"),
            ("detector_efficiency_vs_squeezing", self.ax_sweep_detector, self.canvas_sweep_detector, "line"),
            ("fidelity_vs_gain", self.ax_sweep_gain, self.canvas_sweep_gain, "line"),
            ("fidelity_vs_asymmetry", self.ax_sweep_asymmetry, self.canvas_sweep_asymmetry, "line"),
            ("loss_thermal_heatmap", self.ax_heat_loss, self.canvas_heat_loss, "heat"),
            ("phase_diffusion_sensitivity", self.ax_phase_sensitivity, self.canvas_phase_sensitivity, "line"),
            ("negativity_survival_vs_squeezing", self.ax_survival_r, self.canvas_survival_r, "line"),
        ]
        for key, ax, canvas, kind in mapping:
            if key not in sweeps:
                ax.clear()
                ax.text(0.5, 0.5, "Comparison data not available", ha="center", va="center", transform=ax.transAxes, color=self.COLORS["muted"])
                self._style_axis(ax)
                canvas.draw_idle()
                continue
            if kind == "heat":
                self._plot_heatmap_sweep(ax, canvas, sweeps[key])
            else:
                self._plot_line_sweep(ax, canvas, sweeps[key])

    def _draw_wigner_pair_placeholder(self, message: str) -> None:
        """Show a dark-theme placeholder inside the Wigner Pairs tab."""
        inner = getattr(self, "wigner_pairs_inner", None)
        if inner is None:
            return
        for child in inner.winfo_children():
            child.destroy()
        self.comparison_wigner_pair_figures = []
        card = ttk.Frame(inner, style="PlotCard.TFrame", padding=24)
        card.pack(fill="both", expand=True, padx=18, pady=18)
        ttk.Label(
            card,
            text=message,
            style="Card.TLabel",
            font=("Segoe UI", 13),
            anchor="center",
            justify="center",
        ).pack(fill="both", expand=True, pady=80)

    def _comparison_wigner_state_names(self, result: Dict[str, object]) -> List[str]:
        """Return Wigner-pair names in table order."""
        wigners = dict(result.get("wigners", {}))
        ordered: List[str] = []
        for row in list(result.get("rows", [])):
            name = str(row.get("state", ""))
            if name in wigners and name not in ordered:
                ordered.append(name)
        for name in wigners.keys():
            if name not in ordered:
                ordered.append(str(name))
        return ordered

    def _wigner_color_limits_for_pair(self, W_in: np.ndarray, W_out: np.ndarray) -> Tuple[float, float]:
        """Use one color scale for the input/output Wigner pair.

        The selected colormap and scale mode are shared with the main Wigner
        dashboard.  For a selected comparison state, this makes the two colorbars
        directly comparable instead of letting each panel rescale independently.
        """
        color_scale = "symmetric"
        try:
            color_scale = str(self.vars.get("wigner_color_scale").get())
        except Exception:
            pass

        if color_scale == "auto":
            vmin = float(min(np.nanmin(W_in), np.nanmin(W_out)))
            vmax = float(max(np.nanmax(W_in), np.nanmax(W_out)))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
                return -1.0, 1.0
            return vmin, vmax

        if color_scale == "positive":
            vmax = float(max(np.nanmax(W_in), np.nanmax(W_out)))
            if not np.isfinite(vmax) or vmax < 1e-12:
                vmax = 1.0
            return 0.0, vmax

        zmax = float(max(np.nanmax(np.abs(W_in)), np.nanmax(np.abs(W_out))))
        if not np.isfinite(zmax) or zmax < 1e-12:
            zmax = 1.0
        return -zmax, zmax

    def _populate_selected_comparison_wigner_pair(self, result: Dict[str, object], selected_name: Optional[str] = None) -> None:
        """Populate the Wigner Pairs tab with one selected input/output pair."""
        inner = getattr(self, "wigner_pairs_inner", None)
        if inner is None:
            return
        for child in inner.winfo_children():
            child.destroy()
        self.comparison_wigner_pair_figures = []

        wigners = dict(result.get("wigners", {}))
        if not wigners:
            self._draw_wigner_pair_placeholder("No Wigner-pair data was produced for this comparison run.")
            return

        names = self._comparison_wigner_state_names(result)
        if not names:
            self._draw_wigner_pair_placeholder("No Wigner-pair states are available for this comparison run.")
            return

        if selected_name not in names:
            selected_name = names[0]

        # Keep the dropdown synchronized without triggering reentrant redraws.
        try:
            if str(self.comparison_state_var.get()) != str(selected_name):
                self.comparison_state_var.set(str(selected_name))
        except Exception:
            pass

        pack = wigners.get(str(selected_name))
        if not pack:
            self._draw_wigner_pair_placeholder(f"No Wigner-pair data is available for {selected_name}.")
            return

        xvec = np.asarray(result.get("xvec"), dtype=float)
        pvec = np.asarray(result.get("pvec"), dtype=float)
        W_in = np.asarray(pack["W_in"], dtype=float)
        W_out = np.asarray(pack["W_out"], dtype=float)
        clim = self._wigner_color_limits_for_pair(W_in, W_out)

        card = ttk.Frame(inner, style="PlotCard.TFrame", padding=12)
        card.pack(fill="both", expand=True, padx=14, pady=14)

        row = ttk.Frame(card, style="PlotCard.TFrame")
        row.pack(fill="both", expand=True)
        row.columnconfigure(0, weight=1, uniform="wigner_pair_col")
        row.columnconfigure(1, weight=1, uniform="wigner_pair_col")

        fig_in = Figure(figsize=(7.4, 6.2), dpi=100, facecolor=self.COLORS["card"])
        ax_in = fig_in.add_axes([0.11, 0.13, 0.71, 0.76])
        cax_in = fig_in.add_axes([0.86, 0.13, 0.035, 0.76])
        panel_in = ttk.Frame(row, style="PlotCard.TFrame", padding=4)
        panel_in.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        canvas_in = FigureCanvasTkAgg(fig_in, master=panel_in)
        canvas_in.get_tk_widget().configure(background=self.COLORS["card"], highlightthickness=0, borderwidth=0)
        canvas_in.get_tk_widget().pack(fill="both", expand=True)

        fig_out = Figure(figsize=(7.4, 6.2), dpi=100, facecolor=self.COLORS["card"])
        ax_out = fig_out.add_axes([0.11, 0.13, 0.71, 0.76])
        cax_out = fig_out.add_axes([0.86, 0.13, 0.035, 0.76])
        panel_out = ttk.Frame(row, style="PlotCard.TFrame", padding=4)
        panel_out.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        canvas_out = FigureCanvasTkAgg(fig_out, master=panel_out)
        canvas_out.get_tk_widget().configure(background=self.COLORS["card"], highlightthickness=0, borderwidth=0)
        canvas_out.get_tk_widget().pack(fill="both", expand=True)

        self._plot_wigner(ax_in, cax_in, canvas_in, W_in, xvec, pvec, f"{selected_name}: input Wigner", clim=clim)
        self._plot_wigner(ax_out, cax_out, canvas_out, W_out, xvec, pvec, f"{selected_name}: output Wigner", clim=clim)
        self.comparison_wigner_pair_figures.append((fig_in, fig_out, canvas_in, canvas_out))

    def update_comparison_wigner_pair(self) -> None:
        result = getattr(self, "comparison_result", None)
        if not result:
            return
        name = str(self.comparison_state_var.get())
        self._populate_selected_comparison_wigner_pair(result, selected_name=name if name else None)

    def _save_visible_comparison_figures(self, result: Dict[str, object]) -> None:
        png_folder = str(result.get("png_folder", ""))
        if not png_folder:
            return
        try:
            os.makedirs(png_folder, exist_ok=True)
            figure_exports = [
                (self.fig_comp_fidelity, "comparison_fidelity.png"),
                (self.fig_comp_neg, "comparison_wigner_negativity.png"),
                (self.fig_comp_survival, "comparison_negativity_survival.png"),
                (self.fig_comp_overlap, "comparison_wigner_shape_overlap.png"),
                (getattr(self, "fig_sweep_r", None), "sweep_fidelity_vs_squeezing.png"),
                (getattr(self, "fig_sweep_detector", None), "sweep_detector_efficiency_vs_squeezing.png"),
                (getattr(self, "fig_sweep_gain", None), "sweep_fidelity_vs_gain.png"),
                (getattr(self, "fig_sweep_asymmetry", None), "sweep_fidelity_vs_asymmetry.png"),
                (getattr(self, "fig_heat_loss", None), "sweep_loss_thermal_heatmap.png"),
                (getattr(self, "fig_phase_sensitivity", None), "sweep_phase_diffusion_sensitivity.png"),
                (getattr(self, "fig_survival_r", None), "sweep_negativity_survival_vs_squeezing.png"),
            ]
            for fig, filename in figure_exports:
                if fig is not None:
                    fig.savefig(os.path.join(png_folder, filename), dpi=160, bbox_inches="tight")
        except Exception as exc:
            self._log("Could not save comparison figures: " + str(exc))

    def save_final_state(self) -> None:
        if self.last_best is None:
            messagebox.showinfo("No state to save", "Run an optimization before saving a state.")
            return
        try:
            folder = save_optimized_state_package(self.last_best, DEFAULT_STATE_EXPORT_ROOT)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._log("Saved final state to: " + folder)
        messagebox.showinfo("State saved", "Optimized state saved successfully.\n\n" + folder)

    def _style_axis(self, ax) -> None:
        C = self.COLORS
        ax.figure.patch.set_facecolor(C["card"])
        ax.set_facecolor(C["plot"])

        # Improve readability for presentation mode on the dark theme.
        ax.title.set_color(C["text"])
        ax.title.set_fontsize(17)
        ax.title.set_fontweight("bold")

        ax.xaxis.label.set_color(C["text"])
        ax.yaxis.label.set_color(C["text"])
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)

        ax.tick_params(colors=C["muted"], which="both", labelsize=11)
        for spine in ax.spines.values():
            spine.set_color(C["border"])
            spine.set_linewidth(1.2)
        ax.grid(False)

    def _style_colorbar_axis(self, cax) -> None:
        C = self.COLORS
        cax.set_facecolor(C["card"])
        cax.yaxis.label.set_color(C["text"])
        cax.yaxis.label.set_size(12)
        cax.tick_params(colors=C["muted"], labelsize=10)
        for spine in cax.spines.values():
            spine.set_color(C["border"])
            spine.set_linewidth(1.0)

    def refresh_wigner_style(self) -> None:
        payload = getattr(self, "current_wigner_payload", None)
        if not payload:
            return
        self._plot_wigner(
            self.ax_in,
            self.cax_in,
            self.canvas_in,
            np.asarray(payload["W_in"]),
            np.asarray(payload["xvec"]),
            np.asarray(payload["pvec"]),
            "Input Wigner function",
        )
        self._plot_wigner(
            self.ax_out,
            self.cax_out,
            self.canvas_out,
            np.asarray(payload["W_out"]),
            np.asarray(payload["xvec"]),
            np.asarray(payload["pvec"]),
            "Teleported output Wigner function",
        )

    def _draw_empty(self) -> None:
        for ax, cax, canvas, title in [
            (self.ax_in, self.cax_in, self.canvas_in, "Input Wigner function"),
            (self.ax_out, self.cax_out, self.canvas_out, "Teleported output Wigner function"),
        ]:
            ax.clear()
            cax.clear()
            cax.set_visible(False)
            ax.set_position([0.11, 0.13, 0.71, 0.76])
            cax.set_position([0.86, 0.13, 0.035, 0.76])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("p")
            self._style_axis(ax)
            canvas.draw_idle()

        self.ax_hist.clear()
        self.ax_hist.set_title("Fidelity history")
        self.ax_hist.set_xlabel("live step")
        self.ax_hist.set_ylabel("fidelity")
        self._style_axis(self.ax_hist)
        self.fig_hist.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.14)
        self.canvas_hist.draw_idle()

        self.ax_photon.clear()
        self.ax_photon.set_title("Photon-number distribution")
        self.ax_photon.set_xlabel("n")
        self.ax_photon.set_ylabel("probability")
        self._style_axis(self.ax_photon)
        self.fig_photon.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.14)
        self.canvas_photon.draw_idle()

        self.ax_E.clear()
        self.ax_E.set_title("Electric field expectation")
        self.ax_E.set_xlabel("time")
        self.ax_E.set_ylabel("E(t)")
        self._style_axis(self.ax_E)
        self.fig_E.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.14)
        self.canvas_E.draw_idle()

        self.ax_B.clear()
        self.ax_B.set_title("Magnetic field expectation")
        self.ax_B.set_xlabel("time")
        self.ax_B.set_ylabel("B(t)")
        self._style_axis(self.ax_B)
        self.fig_B.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.14)
        self.canvas_B.draw_idle()

    def _plot_wigner(
        self,
        ax,
        cax,
        canvas,
        W: np.ndarray,
        xvec: np.ndarray,
        pvec: np.ndarray,
        title: str,
        clim: Optional[Tuple[float, float]] = None,
    ) -> None:
        ax.clear()
        cax.clear()
        cax.set_visible(True)

        # Fixed axes geometry prevents progressive shrinking when the plot is
        # updated many times. The colorbar has its own fixed axis.
        ax.set_position([0.11, 0.13, 0.71, 0.76])
        cax.set_position([0.86, 0.13, 0.035, 0.76])

        cmap = "viridis"
        color_scale = "symmetric"
        try:
            cmap = str(self.vars.get("wigner_colormap").get())
            color_scale = str(self.vars.get("wigner_color_scale").get())
        except Exception:
            pass

        if clim is not None:
            vmin, vmax = float(clim[0]), float(clim[1])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
                vmin, vmax = -1.0, 1.0
        elif color_scale == "auto":
            vmin = float(np.min(W))
            vmax = float(np.max(W))
            if abs(vmax - vmin) < 1e-12:
                vmin, vmax = -1.0, 1.0
        elif color_scale == "positive":
            vmin = 0.0
            vmax = float(np.max(W))
            if vmax < 1e-12:
                vmax = 1.0
        else:
            zmax = float(np.max(np.abs(W)))
            if zmax < 1e-12:
                zmax = 1.0
            vmin, vmax = -zmax, zmax

        im = ax.imshow(
            W,
            origin="lower",
            extent=[xvec[0], xvec[-1], pvec[0], pvec[-1]],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("p")
        cb = canvas.figure.colorbar(im, cax=cax, label="W(x,p)")
        cb.ax.yaxis.label.set_color(self.COLORS["text"])
        cb.ax.yaxis.label.set_size(12)
        self._style_axis(ax)
        self._style_colorbar_axis(cax)
        cb.outline.set_edgecolor(self.COLORS["border"])
        canvas.draw_idle()

    def _plot_fields(self, c: np.ndarray, cfg: UserConfig) -> None:
        series = field_time_series(c, cfg)
        t = series["t"]
        E_mean = series["E_mean"]
        E_std = series["E_std"]
        B_mean = series["B_mean"]
        B_std = series["B_std"]

        self.ax_E.clear()
        self.ax_E.plot(t, E_mean, label="<E(t)>", linewidth=2.0)
        if cfg.field_show_uncertainty:
            self.ax_E.fill_between(t, E_mean - E_std, E_mean + E_std, alpha=0.25, label="±ΔE(t)")
        self.ax_E.set_title("Electric field of optimized state")
        self.ax_E.set_xlabel("time")
        self.ax_E.set_ylabel("E(t)")
        self._style_axis(self.ax_E)
        legE = self.ax_E.legend(loc="best", fontsize=11)
        if legE is not None:
            legE.get_frame().set_facecolor(self.COLORS["card"])
            legE.get_frame().set_edgecolor(self.COLORS["border"])
            legE.get_frame().set_alpha(0.95)
            for txt in legE.get_texts():
                txt.set_color(self.COLORS["text"])
        self.fig_E.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.14)
        self.canvas_E.draw_idle()

        self.ax_B.clear()
        self.ax_B.plot(t, B_mean, label="<B(t)>", linewidth=2.0)
        if cfg.field_show_uncertainty:
            self.ax_B.fill_between(t, B_mean - B_std, B_mean + B_std, alpha=0.25, label="±ΔB(t)")
        self.ax_B.set_title("Magnetic field of optimized state")
        self.ax_B.set_xlabel("time")
        self.ax_B.set_ylabel("B(t)")
        self._style_axis(self.ax_B)
        legB = self.ax_B.legend(loc="best", fontsize=11)
        if legB is not None:
            legB.get_frame().set_facecolor(self.COLORS["card"])
            legB.get_frame().set_edgecolor(self.COLORS["border"])
            legB.get_frame().set_alpha(0.95)
            for txt in legB.get_texts():
                txt.set_color(self.COLORS["text"])
        self.fig_B.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.14)
        self.canvas_B.draw_idle()

    def _update_live(self, msg: Dict[str, object]) -> None:
        F = float(msg["fidelity"])
        objF = float(msg.get("objective_fidelity", F))
        E = float(msg["energy"])
        a = complex(msg["a"])
        parity = float(msg["parity"])
        var_n = float(msg.get("n_variance", 0.0))
        Q = float(msg.get("mandel_q", 0.0))
        xvar = float(msg.get("x_var", 0.0))
        pvar = float(msg.get("p_var", 0.0))
        neg = float(msg.get("wigner_negativity", 0.0))
        out_neg = float(msg.get("output_wigner_negativity", 0.0))
        survival_ratio = float(msg.get("negativity_survival_ratio", 0.0))
        obj_survival_ratio = float(msg.get("objective_negativity_survival_ratio", survival_ratio))
        tail = float(msg.get("tail_probability", 0.0))
        score = float(msg.get("selection_score", objF))
        feasible = bool(msg.get("feasible", True))
        violation = float(msg.get("constraint_violation", 0.0))
        label = str(msg["label"])

        c = np.asarray(msg["coeffs"])
        has_wigner = bool(msg.get("has_wigner", True)) and ("W_in" in msg) and ("W_out" in msg)
        if has_wigner:
            W_in = np.asarray(msg["W_in"])
            W_out = np.asarray(msg["W_out"])
            xvec = np.asarray(msg["xvec"])
            pvec = np.asarray(msg["pvec"])
            self.current_wigner_payload = {"W_in": W_in, "W_out": W_out, "xvec": xvec, "pvec": pvec}
        else:
            W_in = W_out = xvec = pvec = None
            self.current_wigner_payload = None
        hist = np.asarray(msg["history"], dtype=float)
        feasible_hist = np.asarray(msg.get("feasible_history", []), dtype=float)

        # Multi-line metric text for presentation.  Keep each line short enough
        # to remain readable and avoid horizontal clipping in fullscreen mode.
        metric_text = (
            f"{label}\n"
            f"F={F:.7f}    Obj={objF:.7f}    Score={score:.7f}    "
            f"E={E:.5f}    Feasible={feasible}    Violation={violation:.1e}\n"
            f"<a>={a.real:+.3e}{a.imag:+.3e}j    Var(n)={var_n:.4g}    Q={Q:.4g}    "
            f"Vx={xvar:.4g}    Vp={pvar:.4g}\n"
            f"Negativity: in={neg:.4g}, out={out_neg:.4g}, survival={survival_ratio:.4g}, "
            f"objective survival={obj_survival_ratio:.4g}    Tail={tail:.2e}"
        )
        for metric_label in getattr(self, "metric_labels", []):
            metric_label.configure(text=metric_text)

        if has_wigner:
            self._plot_wigner(self.ax_in, self.cax_in, self.canvas_in, W_in, xvec, pvec, "Input Wigner function")
            self._plot_wigner(self.ax_out, self.cax_out, self.canvas_out, W_out, xvec, pvec, "Teleported output Wigner function")
        else:
            preview_mode = str(msg.get("preview_mode", "Off"))
            for ax, cax, canvas, title in [
                (self.ax_in, self.cax_in, self.canvas_in, "Input Wigner function"),
                (self.ax_out, self.cax_out, self.canvas_out, "Teleported output Wigner function"),
            ]:
                ax.clear()
                cax.clear()
                cax.set_visible(False)
                ax.set_facecolor(self.COLORS["plot"])
                ax.text(
                    0.5,
                    0.5,
                    f"Live Wigner preview: {preview_mode}\nWigner plots will appear at the final result.",
                    ha="center",
                    va="center",
                    color=self.COLORS["muted"],
                    fontsize=13,
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                self._style_axis(ax)
                canvas.draw_idle()

        self.ax_hist.clear()
        if len(hist) > 0:
            steps = np.arange(len(hist))
            # Hide any legacy/internal penalty values from older messages. Physical
            # fidelities should be finite and normally lie between 0 and 1; the
            # survival objective may still report a separate scalar Score in the
            # metric bar.
            plot_hist = hist.astype(float).copy()
            plot_hist[~np.isfinite(plot_hist)] = np.nan
            plot_hist[np.abs(plot_hist) > 10.0] = np.nan
            finite_sampled = np.isfinite(plot_hist)
            if np.any(finite_sampled):
                self.ax_hist.plot(steps[finite_sampled], plot_hist[finite_sampled], marker="o", markersize=3, linestyle="-", label="sampled fidelity")

            if len(feasible_hist) == len(hist):
                plot_feasible = feasible_hist.astype(float).copy()
                plot_feasible[~np.isfinite(plot_feasible)] = np.nan
                plot_feasible[np.abs(plot_feasible) > 10.0] = np.nan
                finite_mask = np.isfinite(plot_feasible)
                if np.any(finite_mask):
                    self.ax_hist.plot(steps[finite_mask], plot_feasible[finite_mask], linewidth=2, label="best feasible fidelity")
            # Do not fall back to a “best sampled” curve: sampled states may violate
            # active constraints, so calling them best is misleading.
        self.ax_hist.set_title("Fidelity history")
        self.ax_hist.set_xlabel("live step")
        self.ax_hist.set_ylabel("fidelity")
        leg = self.ax_hist.legend(loc="best")
        if leg is not None:
            leg.get_frame().set_facecolor(self.COLORS["card"])
            leg.get_frame().set_edgecolor(self.COLORS["border"])
            for txt in leg.get_texts():
                txt.set_color(self.COLORS["text"])
        self._style_axis(self.ax_hist)
        self.fig_hist.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.14)
        self.canvas_hist.draw_idle()

        probs = np.abs(normalize_coeffs(c)) ** 2
        self.ax_photon.clear()
        self.ax_photon.bar(np.arange(len(probs)), probs)
        self.ax_photon.set_title("Photon-number distribution")
        self.ax_photon.set_xlabel("n")
        self.ax_photon.set_ylabel("probability")
        self._style_axis(self.ax_photon)
        self.fig_photon.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.14)
        self.canvas_photon.draw_idle()

        cfg = self.last_best["config"] if self.last_best is not None else self._cfg()
        self._plot_fields(c, cfg)

    def _show_final(self, best: Dict[str, object]) -> None:
        c = normalize_coeffs(np.asarray(best["coeffs"]))
        refs = best.get("references", [])
        cfg: UserConfig = best["config"]

        lines: List[str] = []

        # A short narrative summary makes the report readable during a presentation.
        coherent_row = next((row for row in refs if row.get("state") == "coherent"), None)
        coherent_fidelity = float(coherent_row["fidelity"]) if coherent_row is not None else None
        coherent_difference = None if coherent_fidelity is None else float(best["fidelity"]) - coherent_fidelity
        objective_mode = str(best.get("objective_mode", "single"))
        survival_enabled = bool(getattr(cfg, "use_negativity_survival_objective", False))

        lines.append("Executive summary\n")
        lines.append("=================\n")
        lines.append(
            "This section gives a presentation-friendly interpretation of the numerical run. "
            "The optimizer searches over normalized pure states in a truncated Fock basis, applies the selected teleportation channel in Wigner phase space, and keeps only states satisfying the active constraints.\n"
        )
        lines.append(
            f"The final state is {'feasible' if bool(best.get('feasible', True)) else 'not feasible'} under the active constraints. "
            f"The base-channel fidelity is {float(best['fidelity']):.6f}, and the selection score used by the optimizer is {float(best.get('selection_score', best['objective_fidelity'])):.6f}.\n"
        )
        if coherent_difference is not None:
            if coherent_difference > 1e-5:
                lines.append(
                    f"Compared with the coherent reference, this state improves the fidelity by {coherent_difference:+.6f}. "
                    "This usually indicates that the selected channel or constraints favor a nonclassical structure.\n"
                )
            elif abs(coherent_difference) <= 1e-5:
                lines.append(
                    "The optimized state is essentially tied with the coherent reference. "
                    "For a unit-gain isotropic Gaussian channel this is the expected benchmark behavior.\n"
                )
            else:
                lines.append(
                    f"The coherent reference has higher fidelity by {-coherent_difference:.6f}. "
                    "This can happen when extra constraints, regularization, or a non-fidelity objective changes the search problem.\n"
                )
        if survival_enabled:
            lines.append(
                "The survival objective is active, so the optimizer balances ordinary teleportation fidelity with the survival ratio of Wigner negativity. "
                "A state with zero input negativity is rejected because the survival ratio would be ill-defined.\n"
            )
        else:
            lines.append(
                f"The objective mode is '{objective_mode}'. Wigner negativity and convergence diagnostics are reported for interpretation, but they do not change the objective unless their corresponding switches are enabled.\n"
            )
        lines.append("\n")

        lines.append("Best optimized state\n")
        lines.append("====================\n")
        lines.append(f"Base-channel fidelity: {float(best['fidelity']):.12f}\n")
        lines.append(f"Objective fidelity:    {float(best['objective_fidelity']):.12f}\n")
        lines.append(f"Selection score:       {float(best.get('selection_score', best['objective_fidelity'])):.12f}\n")
        lines.append(f"Constraint feasible:   {bool(best.get('feasible', True))}\n")
        lines.append(f"Max violation:         {float(best.get('constraint_violation', 0.0)):.12e}\n")
        if best.get('violated_constraints'):
            lines.append(f"Violated constraints:  {best.get('violated_constraints')}\n")
        lines.append(f"Objective mode:        {best.get('objective_mode', 'single')}\n")
        lines.append("Experimental setup:    Manual configuration\n")
        lines.append(f"Computation profile:   {cfg.computation_profile}\n")
        lines.append(f"Live Wigner preview:   {cfg.live_wigner_preview_mode}\n")
        lines.append(f"Energy:                {float(best['energy']):.12f}\n")
        lines.append(f"<a>:                   {complex(best['a'])}\n")
        lines.append(f"|<a>|^2:               {float(best['displacement_squared']):.12f}\n")
        lines.append(f"N_fluct = N-|<a>|^2:   {float(best['fluctuation_energy']):.12f}\n")
        lines.append(f"Parity:                {float(best['parity']):.12f}\n")
        lines.append(f"Var(n):                {float(best['n_variance']):.12f}\n")
        lines.append(f"Mandel Q:              {float(best['mandel_q']):.12f}\n")
        lines.append(f"Var(x):                {float(best['x_var']):.12f}\n")
        lines.append(f"Var(p):                {float(best['p_var']):.12f}\n")
        lines.append(f"Cov(x,p):              {float(best['xp_cov']):.12f}\n")
        lines.append(f"Input Wigner negativity:   {float(best['wigner_negativity']):.12f}\n")
        lines.append(f"Output Wigner negativity:  {float(best.get('output_wigner_negativity', 0.0)):.12f}\n")
        lines.append(f"Negativity survival ratio: {float(best.get('negativity_survival_ratio', 0.0)):.12f}\n")
        lines.append(f"Objective survival ratio:  {float(best.get('objective_negativity_survival_ratio', best.get('negativity_survival_ratio', 0.0))):.12f}\n")
        lines.append(f"High-Fock tail prob.:      {float(best.get('tail_probability', 0.0)):.12e}\n")
        lines.append(f"Active Fock indices:   {list(best.get('active', []))}\n")
        lines.append("\nTime-dependent field model\n")
        lines.append("--------------------------\n")
        lines.append("E(t) = E0 X_theta, B(t) = B0 X_{theta + pi/2}.\n")
        lines.append("This is a dimensionless single-mode oscillator convention.\n")
        lines.append(f"omega = {float(cfg.field_omega):.8g}, periods = {float(cfg.field_periods):.8g}, ")
        lines.append(f"E0 = {float(cfg.field_E_scale):.8g}, B0 = {float(cfg.field_B_scale):.8g}\n\n")

        # ------------------------------------------------------------------
        # Complete constraint and configuration report
        # ------------------------------------------------------------------
        def _fmt_bool(value: bool) -> str:
            return "ON" if bool(value) else "off"

        def _fmt_float(value) -> str:
            try:
                return f"{float(value):.12g}"
            except Exception:
                return str(value)

        def _status(satisfied) -> str:
            if satisfied is None:
                return ""
            return "OK" if bool(satisfied) else "VIOLATION"

        def _constraint_line(name: str, enabled: bool, requirement: str, achieved: str = "", satisfied=None) -> None:
            enabled_text = _fmt_bool(enabled)
            status_text = _status(satisfied)
            line = f"{name:38s} [{enabled_text:3s}]  requirement: {requirement}"
            if achieved:
                line += f" | achieved: {achieved}"
            if status_text:
                line += f" | {status_text}"
            lines.append(line + "\n")

        d_final = state_diagnostics(c)
        norm_final = float(np.vdot(c, c).real)
        imag_weight = float(np.sum(np.abs(c.imag) ** 2))
        max_imag_coeff = float(np.max(np.abs(c.imag))) if len(c) > 0 else 0.0
        active_indices = list(best.get("active", []))
        tail_value = float(best.get("tail_probability", 0.0))
        params: ChannelParams = best.get("params")
        objective_channels = best.get("objective_channels", [])

        try:
            coh_ref = coherent_coeffs(cfg.Ncut, cfg.target_energy)
            coh_overlap_value = coherent_overlap(c, coh_ref)
        except Exception:
            coh_overlap_value = float("nan")

        try:
            sq_ref = squeezed_vacuum_coeffs(cfg.Ncut, cfg.target_energy)
            sq_overlap_value = squeezed_overlap(c, sq_ref)
        except Exception:
            sq_overlap_value = float("nan")

        try:
            cat_ref = cat_coeffs(cfg.Ncut, cfg.cat_alpha, cfg.cat_phase)
            cat_overlap_value = cat_overlap(c, cat_ref)
        except Exception:
            cat_overlap_value = float("nan")

        lines.append("Full constraint and configuration report\n")
        lines.append("========================================\n")

        lines.append("Always-active constraints\n")
        lines.append("-------------------------\n")
        _constraint_line(
            "Normalization",
            True,
            "<psi|psi> = 1",
            _fmt_float(norm_final),
            abs(norm_final - 1.0) <= 1e-6,
        )
        _constraint_line(
            "Fixed energy",
            True,
            f"<n> = {cfg.target_energy}",
            _fmt_float(d_final["energy"]),
            abs(float(d_final["energy"]) - cfg.target_energy) <= 5e-4,
        )

        lines.append("\nCutoff and numerical-stability controls\n")
        lines.append("---------------------------------------\n")
        _constraint_line("Broad random full-subspace seeds", cfg.use_broad_random_seeds, "seed-generation control")
        _constraint_line(
            "High-Fock tail probability",
            cfg.use_tail_probability_constraint,
            f"P_tail <= {cfg.max_tail_probability} over last {cfg.tail_levels} levels",
            _fmt_float(tail_value),
            (not cfg.use_tail_probability_constraint) or tail_value <= cfg.max_tail_probability + 1e-8,
        )
        _constraint_line(
            "Soft high-Fock tail penalty",
            cfg.use_tail_penalty,
            f"score = objective - {cfg.tail_penalty_strength} * P_tail",
            f"P_tail = {_fmt_float(tail_value)}",
        )
        _constraint_line("Polish best state", cfg.use_best_polishing, f"polish maxiter = {cfg.polish_maxiter}")

        lines.append("\nFock-subspace constraints\n")
        lines.append("-------------------------\n")
        _constraint_line(
            "Fock range",
            cfg.use_fock_range_constraint,
            f"{cfg.fock_min} <= n <= {cfg.fock_max}",
            f"active = {active_indices}",
        )
        _constraint_line(
            "Even parity only",
            cfg.use_even_parity_constraint,
            "only n even",
            f"parity = {_fmt_float(d_final['parity'])}",
            (not cfg.use_even_parity_constraint) or float(d_final["parity"]) > 1.0 - 5e-5,
        )
        _constraint_line(
            "Odd parity only",
            cfg.use_odd_parity_constraint,
            "only n odd",
            f"parity = {_fmt_float(d_final['parity'])}",
            (not cfg.use_odd_parity_constraint) or float(d_final["parity"]) < -1.0 + 5e-5,
        )
        _constraint_line(
            "Modular Fock sector",
            cfg.use_modular_fock_constraint,
            f"n = {cfg.residue_k} mod {cfg.modulus_m}",
            f"active = {active_indices}",
        )
        _constraint_line(
            "Real Fock coefficients",
            cfg.use_real_coefficients,
            "Im(c_n) = 0 for all n",
            f"max |Im(c_n)| = {_fmt_float(max_imag_coeff)}, sum Im^2 = {_fmt_float(imag_weight)}",
            (not cfg.use_real_coefficients) or imag_weight <= 5e-8,
        )

        lines.append("\nDisplacement constraints\n")
        lines.append("------------------------\n")
        _constraint_line(
            "Zero displacement",
            cfg.use_zero_displacement_constraint,
            "<a> = 0",
            f"<a> = {complex(d_final['a'])}, |<a>|^2 = {_fmt_float(d_final['displacement_squared'])}",
            (not cfg.use_zero_displacement_constraint) or abs(complex(d_final["a"])) <= 5e-5,
        )
        _constraint_line(
            "Bounded displacement",
            cfg.use_bounded_displacement,
            f"|<a>|^2 <= {cfg.max_displacement_squared}",
            _fmt_float(d_final["displacement_squared"]),
            (not cfg.use_bounded_displacement) or float(d_final["displacement_squared"]) <= cfg.max_displacement_squared + 1e-8,
        )
        _constraint_line(
            "Fixed displacement",
            cfg.use_fixed_displacement,
            f"<a> = {cfg.fixed_a_real} + {cfg.fixed_a_imag}j",
            f"<a> = {complex(d_final['a'])}",
            (not cfg.use_fixed_displacement)
            or (abs(complex(d_final["a"]).real - cfg.fixed_a_real) <= 5e-5
                and abs(complex(d_final["a"]).imag - cfg.fixed_a_imag) <= 5e-5),
        )
        _constraint_line(
            "Real displacement axis",
            cfg.use_real_displacement_axis,
            "Im<a> = 0",
            f"Im<a> = {_fmt_float(complex(d_final['a']).imag)}",
            (not cfg.use_real_displacement_axis) or abs(complex(d_final["a"]).imag) <= 5e-5,
        )

        lines.append("\nFluctuation-energy constraints\n")
        lines.append("------------------------------\n")
        _constraint_line(
            "Minimum fluctuation energy",
            cfg.use_fluctuation_energy_min,
            f"N - |<a>|^2 >= {cfg.min_fluctuation_energy}",
            _fmt_float(d_final["fluctuation_energy"]),
            (not cfg.use_fluctuation_energy_min) or float(d_final["fluctuation_energy"]) >= cfg.min_fluctuation_energy - 1e-8,
        )
        _constraint_line(
            "Maximum fluctuation energy",
            cfg.use_fluctuation_energy_max,
            f"N - |<a>|^2 <= {cfg.max_fluctuation_energy}",
            _fmt_float(d_final["fluctuation_energy"]),
            (not cfg.use_fluctuation_energy_max) or float(d_final["fluctuation_energy"]) <= cfg.max_fluctuation_energy + 1e-8,
        )

        lines.append("\nPhoton-statistics constraints\n")
        lines.append("-----------------------------\n")
        _constraint_line(
            "Minimum photon-number variance",
            cfg.use_photon_variance_min,
            f"Var(n) >= {cfg.min_photon_variance}",
            _fmt_float(d_final["n_variance"]),
            (not cfg.use_photon_variance_min) or float(d_final["n_variance"]) >= cfg.min_photon_variance - 1e-8,
        )
        _constraint_line(
            "Maximum photon-number variance",
            cfg.use_photon_variance_max,
            f"Var(n) <= {cfg.max_photon_variance}",
            _fmt_float(d_final["n_variance"]),
            (not cfg.use_photon_variance_max) or float(d_final["n_variance"]) <= cfg.max_photon_variance + 1e-8,
        )
        _constraint_line(
            "Minimum Mandel Q",
            cfg.use_mandel_q_min,
            f"Q >= {cfg.min_mandel_q}",
            _fmt_float(d_final["mandel_q"]),
            (not cfg.use_mandel_q_min) or float(d_final["mandel_q"]) >= cfg.min_mandel_q - 1e-8,
        )
        _constraint_line(
            "Maximum Mandel Q",
            cfg.use_mandel_q_max,
            f"Q <= {cfg.max_mandel_q}",
            _fmt_float(d_final["mandel_q"]),
            (not cfg.use_mandel_q_max) or float(d_final["mandel_q"]) <= cfg.max_mandel_q + 1e-8,
        )

        lines.append("\nQuadrature-moment constraints\n")
        lines.append("-----------------------------\n")
        _constraint_line(
            "Minimum x variance",
            cfg.use_x_variance_min,
            f"Var(x) >= {cfg.min_x_variance}",
            _fmt_float(d_final["x_var"]),
            (not cfg.use_x_variance_min) or float(d_final["x_var"]) >= cfg.min_x_variance - 1e-8,
        )
        _constraint_line(
            "Maximum x variance",
            cfg.use_x_variance_max,
            f"Var(x) <= {cfg.max_x_variance}",
            _fmt_float(d_final["x_var"]),
            (not cfg.use_x_variance_max) or float(d_final["x_var"]) <= cfg.max_x_variance + 1e-8,
        )
        _constraint_line(
            "Minimum p variance",
            cfg.use_p_variance_min,
            f"Var(p) >= {cfg.min_p_variance}",
            _fmt_float(d_final["p_var"]),
            (not cfg.use_p_variance_min) or float(d_final["p_var"]) >= cfg.min_p_variance - 1e-8,
        )
        _constraint_line(
            "Maximum p variance",
            cfg.use_p_variance_max,
            f"Var(p) <= {cfg.max_p_variance}",
            _fmt_float(d_final["p_var"]),
            (not cfg.use_p_variance_max) or float(d_final["p_var"]) <= cfg.max_p_variance + 1e-8,
        )
        _constraint_line(
            "Minimum xp covariance",
            cfg.use_covariance_min,
            f"Cov(x,p) >= {cfg.min_xp_covariance}",
            _fmt_float(d_final["xp_cov"]),
            (not cfg.use_covariance_min) or float(d_final["xp_cov"]) >= cfg.min_xp_covariance - 1e-8,
        )
        _constraint_line(
            "Maximum xp covariance",
            cfg.use_covariance_max,
            f"Cov(x,p) <= {cfg.max_xp_covariance}",
            _fmt_float(d_final["xp_cov"]),
            (not cfg.use_covariance_max) or float(d_final["xp_cov"]) <= cfg.max_xp_covariance + 1e-8,
        )

        lines.append("\nOverlap and non-Gaussianity constraints\n")
        lines.append("---------------------------------------\n")
        _constraint_line(
            "Maximum coherent overlap",
            cfg.use_coherent_overlap_max,
            f"|<coh|psi>|^2 <= {cfg.max_coherent_overlap}",
            _fmt_float(coh_overlap_value),
            (not cfg.use_coherent_overlap_max) or coh_overlap_value <= cfg.max_coherent_overlap + 1e-8,
        )
        _constraint_line(
            "Maximum squeezed-vacuum overlap",
            cfg.use_squeezed_overlap_max,
            f"|<sq|psi>|^2 <= {cfg.max_squeezed_overlap}",
            _fmt_float(sq_overlap_value),
            (not cfg.use_squeezed_overlap_max) or sq_overlap_value <= cfg.max_squeezed_overlap + 1e-8,
        )
        _constraint_line(
            "Minimum cat-state overlap",
            cfg.use_cat_overlap_min,
            f"|<cat|psi>|^2 >= {cfg.min_cat_overlap}, alpha={cfg.cat_alpha}, phase={cfg.cat_phase}",
            _fmt_float(cat_overlap_value),
            (not cfg.use_cat_overlap_min) or cat_overlap_value >= cfg.min_cat_overlap - 1e-8,
        )
        _constraint_line(
            "Minimum Wigner negativity",
            cfg.use_wigner_negativity_min,
            f"negativity >= {cfg.min_wigner_negativity}",
            _fmt_float(best["wigner_negativity"]),
            (not cfg.use_wigner_negativity_min) or float(best["wigner_negativity"]) >= cfg.min_wigner_negativity - 1e-8,
        )
        _constraint_line(
            "Maximum Wigner negativity",
            cfg.use_wigner_negativity_max,
            f"negativity <= {cfg.max_wigner_negativity}",
            _fmt_float(best["wigner_negativity"]),
            (not cfg.use_wigner_negativity_max) or float(best["wigner_negativity"]) <= cfg.max_wigner_negativity + 1e-8,
        )

        lines.append("\nFidelity + negativity-survival objective\n")
        lines.append("---------------------------------------\n")
        _constraint_line(
            "Joint survival objective",
            cfg.use_negativity_survival_objective,
            "score = weighted fidelity + weighted survival ratio",
            f"F={_fmt_float(best['objective_fidelity'])}, R={_fmt_float(best.get('objective_negativity_survival_ratio', best.get('negativity_survival_ratio', 0.0)))}, score={_fmt_float(best.get('selection_score', best['objective_fidelity']))}",
        )
        _constraint_line(
            "Input negativity floor for survival",
            cfg.use_negativity_survival_objective,
            f"N_in >= {cfg.min_input_negativity_for_survival}",
            _fmt_float(best["wigner_negativity"]),
            (not cfg.use_negativity_survival_objective) or float(best["wigner_negativity"]) >= cfg.min_input_negativity_for_survival - 1e-8,
        )
        lines.append(f"fidelity weight = {cfg.negativity_survival_fidelity_weight}, survival weight = {cfg.negativity_survival_ratio_weight}, ratio clip = {cfg.survival_ratio_clip}\n")

        lines.append("\nRobust-objective settings\n")
        lines.append("-------------------------\n")
        _constraint_line(
            "Robust average objective",
            cfg.use_robust_average_objective,
            f"average over r=[{cfg.robust_r_min},{cfg.robust_r_max}], gain=[{cfg.robust_gain_min},{cfg.robust_gain_max}]",
            f"samples = {len(objective_channels) if objective_channels else 1}",
        )
        _constraint_line(
            "Robust worst-case objective",
            cfg.use_robust_worstcase_objective,
            f"worst case over r=[{cfg.robust_r_min},{cfg.robust_r_max}], gain=[{cfg.robust_gain_min},{cfg.robust_gain_max}]",
            f"samples = {len(objective_channels) if objective_channels else 1}",
        )
        lines.append(f"robust_r_samples = {cfg.robust_r_samples}, robust_gain_samples = {cfg.robust_gain_samples}, robust_phase_samples = {cfg.robust_phase_samples}\n")
        lines.append(f"robust_phase_max = {cfg.robust_phase_max}\n")

        lines.append("\nEffective channel parameters\n")
        lines.append("----------------------------\n")
        if params is not None:
            n_xp = float(getattr(params, "noise_xp", 0.0))
            det_y = float(params.noise_x2 * params.noise_p2 - n_xp ** 2)
            corr_y = n_xp / np.sqrt(max(params.noise_x2 * params.noise_p2, 1e-300))
            lines.append(f"gain = {params.gain}\n")
            lines.append(f"noise_x2 = {params.noise_x2}\n")
            lines.append(f"noise_p2 = {params.noise_p2}\n")
            lines.append(f"noise_xp = {n_xp}\n")
            lines.append(f"det(noise covariance) = {det_y}\n")
            lines.append(f"effective noise correlation rho = {corr_y}\n")
            lines.append(f"displacement drift = ({getattr(params, 'displacement_x', 0.0)}, {getattr(params, 'displacement_p', 0.0)})\n")
            lines.append(f"phase_sigma = {params.phase_sigma}\n")
            lines.append(f"n_phase_quad = {params.n_phase_quad}\n")
        lines.append(f"finite_squeezing_noise = {_fmt_bool(cfg.use_finite_squeezing_noise)}, r = {cfg.r}\n")
        lines.append(f"anisotropic_noise = {_fmt_bool(cfg.use_anisotropic_noise)}, extra x2 = {cfg.anisotropic_noise_x2}, extra p2 = {cfg.anisotropic_noise_p2}\n")
        lines.append(f"thermal_noise = {_fmt_bool(cfg.use_thermal_noise)}, nbar = {cfg.thermal_nbar}, strength = {cfg.thermal_noise_strength}\n")
        lines.append(f"detector_noise = {_fmt_bool(cfg.use_detector_noise)}, eta = {cfg.detector_efficiency_eta}, strength = {cfg.detector_noise_strength}\n")
        lines.append(f"extra_noise = {_fmt_bool(cfg.use_extra_additive_noise)}, x2 = {cfg.extra_noise_x2}, p2 = {cfg.extra_noise_p2}\n")
        lines.append(f"correlated_epr_noise = {_fmt_bool(cfg.use_correlated_epr_noise)}, rho = {cfg.correlated_epr_rho}\n")
        lines.append(f"rotated_asymmetric_epr_noise = {_fmt_bool(cfg.use_rotated_asymmetric_epr_noise)}, major = {cfg.rotated_epr_noise_major2}, minor = {cfg.rotated_epr_noise_minor2}, angle_deg = {cfg.rotated_epr_angle_degrees}\n")
        lines.append(f"loss_channel_noise = {_fmt_bool(cfg.use_loss_channel_noise)}, T = {cfg.loss_transmissivity}, bath_nbar = {cfg.loss_thermal_nbar}\n")
        lines.append(f"displacement_drift = {_fmt_bool(cfg.use_displacement_drift)}, dx = {cfg.drift_x}, dp = {cfg.drift_p}\n")
        lines.append(f"auto_enforce_cp_noise = {_fmt_bool(cfg.auto_enforce_cp_noise)}\n")

        lines.append("\nNumerical settings snapshot\n")
        lines.append("---------------------------\n")
        lines.append(f"Ncut = {cfg.Ncut}, target_energy = {cfg.target_energy}, x_max = {cfg.x_max}, grid_points = {cfg.grid_points}\n")
        lines.append(f"n_starts = {cfg.n_starts}, maxiter = {cfg.maxiter}, ftol = {cfg.ftol}, random_seed = {cfg.random_seed}\n")
        lines.append(f"live_update_every = {cfg.live_update_every}\n")

        lines.append("\nRaw configuration values\n")
        lines.append("------------------------\n")
        for field_name, value in cfg.__dict__.items():
            lines.append(f"{field_name} = {value}\n")

        conv = best.get("convergence_report", None)
        lines.append("\nConvergence diagnostics\n")
        lines.append("-----------------------\n")
        if conv is None:
            lines.append("No convergence diagnostics were run.\n")
        elif "error" in conv:
            lines.append(f"Convergence diagnostics failed: {conv['error']}\n")
        else:
            lines.append(f"Verdict: {conv.get('verdict', '')}\n")
            lines.append(f"Compactness proxy active: {bool(conv.get('has_compactifying_constraint', False))}\n")
            warnings_list = conv.get("warnings", [])
            if warnings_list:
                lines.append("Warnings:\n")
                for warning in warnings_list:
                    lines.append(f"  - {warning}\n")
            else:
                lines.append("Warnings: none\n")

            support = conv.get("support", {})
            if support:
                lines.append("Support diagnostics:\n")
                lines.append(f"  P(top 1 level)             = {_fmt_float(support.get('p_top1', 0.0))}\n")
                lines.append(f"  P(last tail window)        = {_fmt_float(support.get('p_tail', 0.0))}\n")
                lines.append(f"  energy in tail window      = {_fmt_float(support.get('e_tail', 0.0))}\n")
                lines.append(f"  P(upper quarter cutoff)    = {_fmt_float(support.get('p_upper_quarter', 0.0))}\n")
                lines.append(f"  energy upper quarter       = {_fmt_float(support.get('e_upper_quarter', 0.0))}\n")
                lines.append(f"  max n with P_n > 1e-4      = {_fmt_float(support.get('max_n_prob_gt_1e_minus_4', 0.0))}\n")
                lines.append(f"  max n with P_n > 1e-6      = {_fmt_float(support.get('max_n_prob_gt_1e_minus_6', 0.0))}\n")
                lines.append(f"  inverse participation dim  = {_fmt_float(support.get('inverse_participation_dim', 0.0))}\n")
                lines.append(f"  <n^2>                      = {_fmt_float(support.get('n2', 0.0))}\n")

            escape = conv.get("escape_scan", {})
            if escape.get("available"):
                esc_best = escape.get("best", {})
                lines.append("Two-level escape-family scan:\n")
                lines.append(
                    f"  best state |{int(esc_best.get('low', 0))}>/|{int(esc_best.get('high', 0))}> "
                    f"phase={_fmt_float(esc_best.get('phase', 0.0))}\n"
                )
                lines.append(f"  escape objective fidelity  = {_fmt_float(esc_best.get('objective_fidelity', 0.0))}\n")
                lines.append(f"  escape selection score     = {_fmt_float(esc_best.get('selection_score', 0.0))}\n")
                lines.append(f"  score gap best-escape      = {_fmt_float(float(best.get('selection_score', best['objective_fidelity'])) - float(esc_best.get('selection_score', 0.0)))}\n")
                lines.append(f"  high/Ncut fraction         = {_fmt_float(escape.get('best_high_fraction', 0.0))}\n")
                lines.append(f"  last-window score slope    = {_fmt_float(escape.get('last_window_score_slope', 0.0))}\n")
            else:
                lines.append(f"Two-level escape-family scan: not available ({escape.get('reason', 'not run')})\n")

            local_probe = conv.get("local_probe", None)
            if local_probe is not None:
                lines.append("Local optimality probe:\n")
                lines.append(f"  completed trials           = {local_probe.get('completed', 0)} / {local_probe.get('trials', 0)}\n")
                lines.append(f"  successful SLSQP trials    = {local_probe.get('successes', 0)}\n")
                lines.append(f"  max score improvement      = {_fmt_float(local_probe.get('max_improvement', 0.0))}\n")
                lines.append(f"  median score improvement   = {_fmt_float(local_probe.get('median_improvement', 0.0))}\n")
                lines.append(f"  locally stable             = {bool(local_probe.get('locally_stable', False))}\n")
            else:
                lines.append("Local optimality probe: not run\n")

            cutoff_scan = conv.get("cutoff_scan", None)
            if cutoff_scan is not None:
                lines.append("Automatic Ncut scan:\n")
                lines.append(f"  verdict                    = {cutoff_scan.get('verdict', '')}\n")
                lines.append(f"  last-three score range     = {_fmt_float(cutoff_scan.get('last_three_score_range', ''))}\n")
                for row in cutoff_scan.get("rows", []):
                    if "error" in row:
                        lines.append(f"  Ncut={row.get('Ncut')}: ERROR {row.get('error')}\n")
                    else:
                        lines.append(
                            f"  Ncut={int(row['Ncut']):3d}: score={float(row['selection_score']):.9f}, "
                            f"F={float(row['fidelity']):.9f}, Var(n)={float(row['var_n']):.5g}, "
                            f"tail={float(row['tail_probability']):.2e}, "
                            f"maxn(1e-4)={float(row['max_n_gt_1e_minus_4']):.0f}, "
                            f"success={bool(row['success'])}\n"
                        )
            else:
                lines.append("Automatic Ncut scan: not run\n")

        lines.append("\nOptimized Fock coefficients\n")
        lines.append("---------------------------\n")
        for n, cn in enumerate(c):
            if abs(cn) > 1e-6:
                lines.append(
                    f"c[{n:2d}] = {cn.real:+.10f} {cn.imag:+.10f}j"
                    f"     |c_n|^2 = {abs(cn) ** 2:.10f}\n"
                )

        lines.append("\nReference states\n")
        lines.append("----------------\n")
        coherent_F = None
        for row in refs:
            lines.append(
                f"{row['state']:22s}  "
                f"F={float(row['fidelity']):.10f}  "
                f"E={float(row['energy']):.8f}  "
                f"<a>={complex(row['a']):.3e}  "
                f"Var(n)={float(row['n_variance']):.5f}  "
                f"Q={float(row['mandel_q']):+.5f}  "
                f"Vx={float(row['x_var']):.5f}  "
                f"Vp={float(row['p_var']):.5f}  "
                f"Tail={float(row.get('tail_probability', float('nan'))):.3e}  "
                f"allowed={bool(row['allowed'])}  "
                f"viol={float(row.get('max_violation', 0.0)):.1e}\n"
            )
            if row["state"] == "coherent":
                coherent_F = float(row["fidelity"])

        if coherent_F is not None:
            diff = float(best["fidelity"]) - coherent_F
            lines.append("\nComparison to coherent reference\n")
            lines.append("--------------------------------\n")
            lines.append(f"F_best - F_coherent = {diff:+.12f}\n")
            if diff > 1e-5:
                lines.append("Result: optimized state beats coherent for this configured problem.\n")
            elif abs(diff) <= 1e-5:
                lines.append("Result: optimized state is essentially tied with coherent.\n")
            else:
                lines.append("Result: optimized state does not beat unconstrained coherent.\n")

        lines.append("\nStart summary\n")
        lines.append("-------------\n")
        for item in best.get("all_results", []):
            lines.append(
                f"start {int(item['start_index']):2d}: "
                f"F={float(item['fidelity']):.10f}, "
                f"Obj={float(item['objective_fidelity']):.10f}, "
                f"Score={float(item.get('selection_score', item['objective_fidelity'])):.10f}, "
                f"Surv={float(item.get('negativity_survival_ratio', 0.0)):.4f}, "
                f"Tail={float(item.get('tail_probability', 0.0)):.2e}, "
                f"E={float(item['energy']):.8f}, "
                f"feasible={bool(item.get('feasible', False))}, "
                f"viol={float(item.get('constraint_violation', 0.0)):.1e}, "
                f"success={bool(item['success'])}, "
                f"message={item['message']}\n"
            )

        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "".join(lines))
        self._style_result_report()

    def _style_result_report(self) -> None:
        """Apply lightweight typography to the text report without changing content."""
        text_widget = self.result_text
        content = text_widget.get("1.0", "end-1c")
        lines = content.splitlines()

        for line_index, line in enumerate(lines, start=1):
            start = f"{line_index}.0"
            end = f"{line_index}.end"
            next_line = lines[line_index] if line_index < len(lines) else ""
            if next_line and set(next_line) == {"="}:
                text_widget.tag_add("report_title", start, end)
            elif next_line and set(next_line) == {"-"}:
                text_widget.tag_add("report_section", start, end)
            elif line.startswith("This section") or line.startswith("The final state") or line.startswith("Compared") or line.startswith("The coherent") or line.startswith("The survival") or line.startswith("The objective"):
                text_widget.tag_add("report_note", start, end)

        for token, tag in [
            ("OK", "report_ok"),
            ("VIOLATION", "report_bad"),
            ("CAUTION", "report_warn"),
            ("Warnings", "report_warn"),
            ("not feasible", "report_bad"),
            ("feasible", "report_ok"),
        ]:
            pos = "1.0"
            while True:
                pos = text_widget.search(token, pos, stopindex="end", nocase=False)
                if not pos:
                    break
                end = f"{pos}+{len(token)}c"
                text_widget.tag_add(tag, pos, end)
                pos = end

    def _log(self, text: str) -> None:
        self.status.insert("end", text + "\n")
        self.status.see("end")


def main() -> None:
    root = tk.Tk()
    FidelityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
