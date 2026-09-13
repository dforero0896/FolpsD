import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Native numpy for testing assertions and mutable fixtures.  Under the JAX
# backend, `np` resolves to jax.numpy (immutable arrays, no `testing` module),
# so anything that needs real numpy arrays or `np.testing.*` must use this.
import numpy as _np_native

# Make JAX use double precision to match the NumPy pipeline behavior.
jax.config.update("jax_enable_x64", True)

# Force a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")

# Select JAX backend before importing folps internals.
os.environ["FOLPS_BACKEND"] = "jax"

from cosmo_class import run_class
from folps import (
    BispectrumCalculator,
    MatrixCalculator,
    NonLinearPowerSpectrumCalculator,
    RSDMultipolesPowerSpectrumCalculator,
    extrapolate_pklin,
    get_pknow_jax,
    get_rsd_pkell_marg_const,
    get_rsd_pkell_marg_derivatives,
)
# === GEO-FPT imports ===
from folps import (
    BispectrumCalculator_Geo,
    BispectrumCalculator_fk_Geo,
    F_VALS_FULL,
    geo_fac,
    geo_fac_pade,
    interpolate_geo_coeffs,
)


# ======================================================================================
# Explicit test coefficient tables (kept inside the test file so the assertions do
# not depend on whatever the real F_VALS_FULL calibration happens to contain)
# ======================================================================================

# Identity table: geo_fac == 1 exactly for any triangle.  Used to (a) check that the
# geo class reduces to the standard calculator when both the P swap and the shape
# correction are disabled, and (b) isolate the effect of the P swap alone.
_IDENTITY_GEO_COEFFS = _np_native.array([
    [1.0, 1.0, 1.0],   # f1
    [0.0, 0.0, 0.0],   # f2
    [0.0, 0.0, 0.0],   # f3
    [0.0, 0.0, 0.0],   # f4
    [0.0, 0.0, 0.0],   # f5
])


def _load_linear_pk() -> dict:
    """Load linear P(k) either from CLASS or from the local fallback file."""
    try:
        from classy import Class as _Class  # noqa: F401

        return run_class(
            h=0.6711,
            ombh2=0.022,
            omch2=0.122,
            omnuh2=0.0006442,
            As=2e-9,
            ns=0.965,
            z=0.3,
            z_scale=[0.97],
            N_ur=2.0328,
            khmin=0.0001,
            khmax=2.0,
            nbk=1000,
            spectra="cb",
        )
    except Exception:
        data_path = Path(__file__).resolve().parent / "inputpkT.txt"
        k_arr, pk_arr = np.loadtxt(data_path, unpack=True)
        return {"k": k_arr, "pk": pk_arr}


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")


def _extract_pknow_array(pknow_result):
    """Accept either get_pknow_jax outputs: (k, pnow) or pnow-only."""
    if isinstance(pknow_result, tuple):
        if len(pknow_result) != 2:
            raise ValueError("Unexpected tuple format for pknow result.")
        return pknow_result[1]
    return pknow_result


def _prepare_pknow_on_target_k(k_target: np.ndarray, pknow_result) -> jnp.ndarray:
    """Return pknow sampled on k_target to avoid shape mismatches in FOLPS internals."""
    if isinstance(pknow_result, tuple):
        if len(pknow_result) != 2:
            raise ValueError("Unexpected tuple format for pknow result.")
        k_pknow = np.asarray(pknow_result[0], dtype=np.float64)
        pknow_arr = np.asarray(pknow_result[1], dtype=np.float64)
    else:
        k_pknow = np.asarray(k_target, dtype=np.float64)
        pknow_arr = np.asarray(pknow_result, dtype=np.float64)

    k_target = np.asarray(k_target, dtype=np.float64)
    if pknow_arr.shape[0] != k_target.shape[0] or k_pknow.shape[0] != k_target.shape[0]:
        pknow_arr = np.interp(k_target, k_pknow, pknow_arr)

    return jnp.asarray(pknow_arr)


def _print_timing_table(rows: list[tuple[str, str]]) -> None:
    title = "[test_folps_jax] JAX JIT timing summary"
    metric_width = max(len("Metric"), max(len(metric) for metric, _ in rows))
    value_width = max(len("Value"), max(len(value) for _, value in rows))

    top = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    header = f"| {'Metric'.ljust(metric_width)} | {'Value'.rjust(value_width)} |"

    print(title)
    print(top)
    print(header)
    print(top)
    for metric, value in rows:
        print(f"| {metric.ljust(metric_width)} | {value.rjust(value_width)} |")
    print(top)


def _jit_speedup(first_seconds: float, cached_seconds: float) -> tuple[float, float]:
    if cached_seconds <= 0:
        return np.inf, 100.0
    speedup = first_seconds / cached_seconds
    gain = (1.0 - cached_seconds / first_seconds) * 100.0 if first_seconds > 0 else 0.0
    return speedup, gain


def _plot_power_spectrum(
    k: np.ndarray,
    p0: np.ndarray,
    p2: np.ndarray,
    p4: np.ndarray,
    p0_marg: np.ndarray,
    p2_marg: np.ndarray,
    p4_marg: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel(r"$k\, [h\, \mathrm{Mpc}^{-1}]$", fontsize=14)
    ax.set_ylabel(r"$k\, P_{\ell}(k)\, [h^{-1}\, \mathrm{Mpc}]^2$", fontsize=14)

    ax.plot(k, k * p0, color="navy", ls="-", label=r"$\ell = 0$")
    ax.plot(k, k * p2, color="maroon", ls="-", label=r"$\ell = 2$")
    ax.plot(k, k * p4, color="darkgreen", ls="-", label=r"$\ell = 4$")

    ax.plot(k, k * p0_marg, color="navy", ls=":", lw=3, label=r"$\ell = 0$ (marginalized)")
    ax.plot(k, k * p2_marg, color="maroon", ls=":", lw=3)
    ax.plot(k, k * p4_marg, color="darkgreen", ls=":", lw=3)

    ax.set_xlim([k[0], 0.2])
    ax.set_ylim([-250, 1900])
    leg = ax.legend(loc="best")
    leg.get_frame().set_linewidth(0.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_bispectrum(
    k_ev: np.ndarray,
    b000: np.ndarray,
    b110: np.ndarray,
    b220: np.ndarray,
    b202: np.ndarray,
    b022: np.ndarray,
    b112: np.ndarray,
    outpath: Path,
) -> None:
    xmax = 0.2
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].set_ylabel(r"$k^2\, B(k,k)\, [h^{-1}\, \mathrm{Mpc}]^4$", fontsize=14)
    axs[0].plot(k_ev, k_ev**2 * b000, label="B000", ls="-", color="red")
    axs[0].plot(k_ev, k_ev**2 * b110, label="B110", ls="-", color="blue")
    axs[0].plot(k_ev, k_ev**2 * b220, label="B220", ls="-", color="green")
    axs[0].set_xlim([0, xmax])
    axs[0].legend(fontsize=12, loc="best")
    axs[0].set_title("B_l1l2L * H_l1l2L", fontsize=16, pad=15)

    axs[1].set_xlabel(r"$k\, [h\, \mathrm{Mpc}^{-1}]$", fontsize=14)
    axs[1].set_ylabel(r"$k^2\, B(k,k)\, [h^{-1}\, \mathrm{Mpc}]^4$", fontsize=14)
    axs[1].plot(k_ev, k_ev**2 * b202, label="B202", ls="-", color="red")
    axs[1].plot(k_ev, k_ev**2 * b022, label="B022", ls="--", color="green")
    axs[1].plot(k_ev, k_ev**2 * b112, label="B112", ls="-", color="blue")
    axs[1].set_xlim([0, xmax])
    axs[1].legend(fontsize=12, loc="best")
    axs[1].set_title("B_l1l2L * H_l1l2L", fontsize=16, pad=15)

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_bispectrum_geo(
    k_ev: np.ndarray,
    curves,
    outpath: Path,
) -> None:
    """Compare Sugiyama-Bell bispectrum multipoles across models.

    Layout mirrors ``_plot_bispectrum``: left panel shows B000/B110/B220,
    right panel shows B202/B022/B112.  Colors encode the multipole (following
    the example notebook), linestyles encode the model.  Two legends are drawn
    per panel: one for the multipole→color mapping and one for the
    model→linestyle mapping.

    Parameters
    ----------
    k_ev : array
        Wavenumber grid for the diagonal (k1 = k2) Sugiyama-Bell multipoles.
    curves : list of dicts
        Each entry has keys:
          - ``label`` : legend label for the model
          - ``ls``    : matplotlib linestyle identifying the model
          - ``B``     : length-6 sequence
                        [B000, B110, B220, B202, B022, B112]
        All ``B`` arrays must have the same length as ``k_ev``.
    outpath : Path
        Output figure path.
    """
    xmax = 0.2
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: notebook colors for B000, B110, B220
    left_specs = [
        {"name": "B000", "idx": 0, "color": "red"},
        {"name": "B110", "idx": 1, "color": "blue"},
        {"name": "B220", "idx": 2, "color": "green"},
    ]
    # Right panel: notebook colors for B202, B022, B112
    right_specs = [
        {"name": "B202", "idx": 3, "color": "red"},
        {"name": "B022", "idx": 4, "color": "green"},
        {"name": "B112", "idx": 5, "color": "blue"},
    ]

    def draw_panel(ax, specs, panel_title):
        # Curves: color = multipole, linestyle = model
        for spec in specs:
            for c in curves:
                B = np.asarray(c["B"][spec["idx"]])
                ax.plot(
                    k_ev, k_ev**2 * B,
                    color=spec["color"],
                    ls=c["ls"],
                    lw=1.5,
                    alpha=0.9,
                )

        # Legend #1: multipole -> color
        color_handles = [
            Line2D([0], [0], color=s["color"], lw=2.2,
                   label=rf"${s['name']}$")
            for s in specs
        ]
        leg_colors = ax.legend(
            handles=color_handles, loc="upper left",
            fontsize=10, title="Multipole", frameon=True,
        )
        leg_colors.get_title().set_fontsize(10)
        ax.add_artist(leg_colors)

        # Legend #2: model -> linestyle
        style_handles = [
            Line2D([0], [0], color="black", ls=c["ls"], lw=2.0,
                   label=c["label"])
            for c in curves
        ]
        leg_styles = ax.legend(
            handles=style_handles, loc="upper right",
            fontsize=9, title="Model", frameon=True,
        )
        leg_styles.get_title().set_fontsize(9)

        ax.set_xlabel(r"$k\, [h\, \mathrm{Mpc}^{-1}]$", fontsize=14)
        ax.set_ylabel(r"$k^2\, B(k,k)\, [h^{-1}\, \mathrm{Mpc}]^4$", fontsize=14)
        ax.set_title(panel_title, fontsize=15, pad=12)
        ax.set_xlim([0, xmax])
        ax.axhline(0.0, color="grey", lw=0.6, alpha=0.5)

    draw_panel(axs[0], left_specs, "B000, B110, B220")
    draw_panel(axs[1], right_specs, "B202, B022, B112")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_scoccimarro_comparison(
    k_grid,
    curves,
    outpath,
    k_max=0.12,
    triangle_label="equilateral",
):
    """Plot Scoccimarro B0, B2, B4 for one or more models.

    Mirrors the plotting style of ``example_folps_jax.ipynb`` (cells 27/30):
    semilogy y-axis, ``k1*k2*k3*B`` scaling (equal to ``k^3 * B`` for the
    equilateral triangles used here), and marker + line style.

    Parameters
    ----------
    k_grid : array
        Wavenumbers along the k1 axis (equilateral: k1 = k2 = k3 = k_grid).
    curves : list of dicts
        Each entry has keys ``label``, ``color``, ``ls``, ``marker``, and
        ``b0``, ``b2``, ``b4``.  All ``b*`` arrays must have the same length
        as ``k_grid``.
    outpath : Path
        Output figure path.
    k_max : float
        Upper x-limit (default 0.12, matching the notebook).
    triangle_label : str
        Text placed in the subplot titles, e.g. ``"equilateral"``.
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ells = [0, 2, 4]
    for ax, ell in zip(axs, ells):
        for c in curves:
            y = np.asarray(c[f"b{ell}"])
            # Notebook scaling: k1 * k2 * k3 * B  ==  k^3 * B for equilateral
            scaled = k_grid**3 * y
            # semilogy needs strictly positive values; render nonpositive points
            # as NaN so a gap appears in the curve instead of the plot crashing.
            plot_y = np.where(scaled > 0, scaled, np.nan)
            ax.semilogy(
                k_grid, plot_y,
                color=c["color"],
                ls=c.get("ls", "-"),
                marker=c.get("marker", "o"),
                ms=4, lw=1.4,
                label=c["label"],
            )
        ax.set_xlabel(r"$k\, [h\,\mathrm{Mpc}^{-1}]$", fontsize=13)
        ax.set_ylabel(
            rf"$k_1 k_2 k_3\, B_{{{ell}}}\ \ [(h^{{-1}}\,\mathrm{{Mpc}})^3]$",
            fontsize=13,
        )
        ax.set_title(rf"$B_{{{ell}}}$ ({triangle_label})", fontsize=14)
        ax.set_xlim([k_grid[0], min(k_max, k_grid[-1])])
        ax.grid(True, which="both", alpha=0.3)
        if ell == 0:
            ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ======================================================================================
# Main test
# ======================================================================================

def run_test_folps_jax() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "test_outputs_jax"
    output_dir.mkdir(parents=True, exist_ok=True)

    classy = _load_linear_pk()

    kwargs = {
        "z": 0.3,
        "h": 0.6711,
        "Omega_m": 0.3211636237981114,
        "f0": np.float64(0.6880638641959066),
        "fnu": 0.004453689063655854,
    }

    b1 = 1.645
    b2 = -0.46
    bs2 = -4.0 / 7.0 * (b1 - 1.0)
    b3nl = 32.0 / 315.0 * (b1 - 1.0)
    alpha0, alpha2, alpha4 = 3.0, -28.9, 0.0
    ctilde = 0.0
    pshot_pk = 1.0 / 0.0002118763
    alphashot0, alphashot2 = 0.08, -8.1
    x_fog_pk = 1.0

    pars_pk = jnp.asarray(
        [
            b1, b2, bs2, b3nl,
            alpha0, alpha2, alpha4, ctilde,
            alphashot0, alphashot2, pshot_pk, x_fog_pk,
        ]
    )

    k_jax = jnp.asarray(classy["k"])
    pk_jax = jnp.asarray(classy["pk"])
    k_np = np.asarray(classy["k"], dtype=np.float64)

    t0 = time.perf_counter()
    k_extrap, pk_extrap = extrapolate_pklin(k=np.asarray(classy["k"]), pk=np.asarray(classy["pk"]))
    pknow_result = get_pknow_jax(k=jnp.asarray(k_extrap), pk=jnp.asarray(pk_extrap), h=kwargs["h"])
    pknow_jax = _prepare_pknow_on_target_k(k_np, pknow_result)
    t_pknow = time.perf_counter() - t0

    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = time.perf_counter()
        matrix = MatrixCalculator(A_full=True, save_dir='output_matrices')
        mmatrices = matrix.get_mmatrices()
        t_matrix = time.perf_counter() - t0

        nonlinear = NonLinearPowerSpectrumCalculator(mmatrices=mmatrices, kernels="fk", **kwargs)

        # Setup run used for marginalization terms and bispectrum inputs.
        t0 = time.perf_counter()
        table, table_now = nonlinear.calculate_loop_table(
            k=k_jax,
            pklin=pk_jax,
            pknow=pknow_jax,
            cosmo=None,
            **kwargs,
        )
        t_loop_setup = time.perf_counter() - t0

        p0_c, p2_c, p4_c = get_rsd_pkell_marg_const(
            kobs=k_jax,
            qpar=1.0,
            qper=1.0,
            pars=pars_pk,
            table=table,
            table_now=table_now,
            bias_scheme="folps",
            damping="lor",
            model="FOLPSD",
        )
        p0_i, p2_i, p4_i = get_rsd_pkell_marg_derivatives(
            kobs=k_jax,
            qpar=1.0,
            qper=1.0,
            pars=pars_pk,
            table=table,
            table_now=table_now,
            bias_scheme="folps",
            damping="lor",
            model="FOLPSD",
        )

    p0_marg = p0_c + (alpha0 * p0_i[0] + alpha2 * p0_i[1] + alpha4 * p0_i[2] + alphashot0 * p0_i[3] + alphashot2 * p0_i[4])
    p2_marg = p2_c + (alpha0 * p2_i[0] + alpha2 * p2_i[1] + alpha4 * p2_i[2] + alphashot0 * p2_i[3] + alphashot2 * p2_i[4])
    p4_marg = p4_c + (alpha0 * p4_i[0] + alpha2 * p4_i[1] + alpha4 * p4_i[2] + alphashot0 * p4_i[3] + alphashot2 * p4_i[4])

    @jax.jit
    def compute_pkells_jit(k, pklin, pknow, qpar, qper, pars, kw):
        nonlinear_local = NonLinearPowerSpectrumCalculator(
            mmatrices=mmatrices,
            kernels="fk",
            **kw,
        )
        table_local, table_now_local = nonlinear_local.calculate_loop_table(
            k=k,
            pklin=pklin,
            pknow=pknow,
            cosmo=None,
            **kw,
        )
        multipoles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD")
        p0_local, p2_local, p4_local = multipoles.get_rsd_pkell(
            kobs=k,
            qpar=qpar,
            qper=qper,
            pars=pars,
            table=table_local,
            table_now=table_now_local,
            bias_scheme="folps",
            damping="lor",
        )
        return p0_local, p2_local, p4_local

    t0 = time.perf_counter()
    p0_jit_1, p2_jit_1, p4_jit_1 = compute_pkells_jit(k_jax, pk_jax, pknow_jax, 1.0, 1.0, pars_pk, kwargs)
    p0_jit_1 = jax.block_until_ready(p0_jit_1)
    p2_jit_1 = jax.block_until_ready(p2_jit_1)
    p4_jit_1 = jax.block_until_ready(p4_jit_1)
    t_pk_jit_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    p0_jit_2, p2_jit_2, p4_jit_2 = compute_pkells_jit(k_jax, pk_jax, pknow_jax, 1.0, 1.0, pars_pk, kwargs)
    p0_jit_2 = jax.block_until_ready(p0_jit_2)
    p2_jit_2 = jax.block_until_ready(p2_jit_2)
    p4_jit_2 = jax.block_until_ready(p4_jit_2)
    t_pk_jit_cached = time.perf_counter() - t0

    pk_speedup, pk_gain = _jit_speedup(t_pk_jit_first, t_pk_jit_cached)

    k_ev = np.linspace(0.01, 0.2, num=40)
    k1k2_pairs = np.vstack([k_ev, k_ev]).T
    k1k2_pairs_jax = jnp.asarray(k1k2_pairs)

    pshot_bk = 0.0
    bshot = 0.0
    c1, c2 = 0.0, 0.0
    x_fog_bk = 1.0
    f0 = nonlinear.f0
    pars_bk = jnp.asarray([b1, b2, bs2, c1, c2, bshot, pshot_bk, x_fog_bk])

    k_pkl_pklnw = jnp.asarray(np.array([np.asarray(table[0]), np.asarray(table[1]), np.asarray(table_now[1])]))

    @jax.jit
    def compute_bispectrum_jit(bpars, f, qpar, qper):
        bispectrum = BispectrumCalculator(model="FOLPSD")
        return bispectrum.Sugiyama_Bell(
            f=f,
            bpars=bpars,
            k_pkl_pklnw=k_pkl_pklnw,
            k1k2pairs=k1k2_pairs_jax,
            qpar=qpar,
            qper=qper,
            precision=[10, 10, 10],
            damping="lor",
            multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
            renormalize=True,
            interpolation_method="linear",
            bias_scheme="folps",
        )

    t0 = time.perf_counter()
    b000_1, b110_1, b220_1, b202_1, b022_1, b112_1 = compute_bispectrum_jit(pars_bk, f0, 1.0, 1.0)
    b000_1 = jax.block_until_ready(b000_1)
    b110_1 = jax.block_until_ready(b110_1)
    b220_1 = jax.block_until_ready(b220_1)
    b202_1 = jax.block_until_ready(b202_1)
    b022_1 = jax.block_until_ready(b022_1)
    b112_1 = jax.block_until_ready(b112_1)
    t_bk_jit_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    b000_2, b110_2, b220_2, b202_2, b022_2, b112_2 = compute_bispectrum_jit(pars_bk, f0, 1.0, 1.0)
    b000_2 = jax.block_until_ready(b000_2)
    b110_2 = jax.block_until_ready(b110_2)
    b220_2 = jax.block_until_ready(b220_2)
    b202_2 = jax.block_until_ready(b202_2)
    b022_2 = jax.block_until_ready(b022_2)
    b112_2 = jax.block_until_ready(b112_2)
    t_bk_jit_cached = time.perf_counter() - t0

    bk_speedup, bk_gain = _jit_speedup(t_bk_jit_first, t_bk_jit_cached)

    p0_plot = np.asarray(p0_jit_2)
    p2_plot = np.asarray(p2_jit_2)
    p4_plot = np.asarray(p4_jit_2)

    b000_plot = np.asarray(b000_2)
    b110_plot = np.asarray(b110_2)
    b220_plot = np.asarray(b220_2)
    b202_plot = np.asarray(b202_2)
    b022_plot = np.asarray(b022_2)
    b112_plot = np.asarray(b112_2)

    _assert_finite("P0 (JIT)", p0_plot)
    _assert_finite("P2 (JIT)", p2_plot)
    _assert_finite("P4 (JIT)", p4_plot)
    _assert_finite("B000 (JIT)", b000_plot)
    _assert_finite("B202 (JIT)", b202_plot)

    power_fig = output_dir / "power_spectrum_jax.png"
    bispec_fig = output_dir / "bispectrum_jax.png"
    results_npz = output_dir / "results_jax.npz"

    _plot_power_spectrum(
        np.asarray(k_jax),
        p0_plot, p2_plot, p4_plot,
        np.asarray(p0_marg), np.asarray(p2_marg), np.asarray(p4_marg),
        power_fig,
    )
    _plot_bispectrum(
        k_ev,
        b000_plot, b110_plot, b220_plot, b202_plot, b022_plot, b112_plot,
        bispec_fig,
    )

    # ==================================================================================
    # ============================   GEO-FPT TESTS   ===================================
    # ==================================================================================
    print()
    print("=" * 90)
    print("[geo] Starting GEO-FPT test suite")
    print("=" * 90)

    # ----------------------------------------------------------------------------------
    # (1) Build the nonlinear P table for GEO-FPT
    # ----------------------------------------------------------------------------------
    t0 = time.perf_counter()
    k_pkl_pklnw_nl_np = nonlinear.get_geofpt_pk_tables(
        k=k_jax,
        pklin=pk_jax,
        pknow=pknow_jax,
        cosmo=None,
        **kwargs,
    )
    t_geo_pk_table = time.perf_counter() - t0

    k_pkl_pklnw_nl = jnp.asarray(np.array([
        np.asarray(k_pkl_pklnw_nl_np[0]),
        np.asarray(k_pkl_pklnw_nl_np[1]),
        np.asarray(k_pkl_pklnw_nl_np[2]),
    ]))

    # Sanity: the nonlinear P should differ from the linear one
    pk_lin_arr = np.asarray(k_pkl_pklnw[1])
    pk_nl_arr  = np.asarray(k_pkl_pklnw_nl[1])
    pk_nl_nw   = np.asarray(k_pkl_pklnw_nl[2])
    assert pk_nl_arr.shape == pk_lin_arr.shape, "P_nl shape mismatch"
    assert pk_nl_nw.shape == pk_lin_arr.shape, "P_nl_no_wiggle shape mismatch"
    max_rel_diff = np.max(np.abs(pk_nl_arr / pk_lin_arr - 1.0))
    assert max_rel_diff > 1e-6, (
        f"get_geofpt_pk_tables returns P_nl == P_lin (max rel diff {max_rel_diff:.2e}). "
        "The 1-loop contribution should be nonzero."
    )
    print(f"[geo] P_nl vs P_lin: max relative difference = {max_rel_diff:.4e}")

    # ----------------------------------------------------------------------------------
    # (2) Standard calculator reference
    # ----------------------------------------------------------------------------------
    bisc_std = BispectrumCalculator(model="FOLPSD")
    Bstd = bisc_std.Sugiyama_Bell(
        f=f0,
        bpars=pars_bk,
        k_pkl_pklnw=k_pkl_pklnw,
        k1k2pairs=k1k2_pairs_jax,
        qpar=1.0,
        qper=1.0,
        precision=[10, 10, 10],
        damping="lor",
        multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
        renormalize=True,
        interpolation_method="linear",
        bias_scheme="folps",
    )
    Bstd = tuple(np.asarray(b) for b in Bstd)

    # ----------------------------------------------------------------------------------
    # (3) Backwards compatibility:
    #     geo with identity coeffs and LINEAR P == standard
    # ----------------------------------------------------------------------------------
    bisc_geo_compat = BispectrumCalculator_Geo(
        model="FOLPSD",
        geo_expansion="pade",
        fi_vals=_IDENTITY_GEO_COEFFS,
        k_pkl_pklnw_nl=k_pkl_pklnw,     # feed LINEAR P as the "nonlinear" one
        z=kwargs["z"],
    )
    Bgeo_compat = bisc_geo_compat.Sugiyama_Bell(
        f=f0,
        bpars=pars_bk,
        k_pkl_pklnw=k_pkl_pklnw,
        k1k2pairs=k1k2_pairs_jax,
        qpar=1.0,
        qper=1.0,
        precision=[10, 10, 10],
        damping="lor",
        multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
        renormalize=True,
        interpolation_method="linear",
        bias_scheme="folps",
    )
    Bgeo_compat = tuple(np.asarray(b) for b in Bgeo_compat)

    labels = ["B000", "B110", "B220", "B202", "B022", "B112"]
    for name, ref, got in zip(labels, Bstd, Bgeo_compat):
        np.testing.assert_allclose(
            ref, got, rtol=1e-9, atol=1e-13,
            err_msg=(
                f"[geo-compat] GEO-FPT with identity coeffs and linear P must match "
                f"the standard calculator, but {name} differs. "
                f"max|Δ|={np.max(np.abs(ref - got)):.3e}"
            ),
        )
    print("[geo] Backwards-compat check: GEO(identity, linear P) == standard (PASS)")

    # ----------------------------------------------------------------------------------
    # (4) Nonlinear-P swap changes the result (F_VALS_FULL, so shape correction is on)
    # ----------------------------------------------------------------------------------
    bisc_geo_nl = BispectrumCalculator_Geo(
        model="FOLPSD",
        geo_expansion="pade",
        fi_vals=F_VALS_FULL,
        k_pkl_pklnw_nl=k_pkl_pklnw_nl,
        z=kwargs["z"],
    )
    Bgeo_nl = bisc_geo_nl.Sugiyama_Bell(
        f=f0,
        bpars=pars_bk,
        k_pkl_pklnw=k_pkl_pklnw,
        k1k2pairs=k1k2_pairs_jax,
        qpar=1.0,
        qper=1.0,
        precision=[10, 10, 10],
        damping="lor",
        multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
        renormalize=True,
        interpolation_method="linear",
        bias_scheme="folps",
    )
    Bgeo_nl = tuple(np.asarray(b) for b in Bgeo_nl)

    rel_swap = np.max(np.abs(Bgeo_nl[0] - Bstd[0]) / (np.abs(Bstd[0]) + 1e-30))
    assert rel_swap > 1e-4, (
        f"[geo-nl] GEO with nonlinear P should differ from standard, "
        f"but max rel diff on B000 is {rel_swap:.2e}"
    )
    print(f"[geo] Nonlinear-P swap changes B000 by max relative {rel_swap:.4e} (PASS)")

    # ----------------------------------------------------------------------------------
    # (5) Doubling the shape coeffs changes the result
    # ----------------------------------------------------------------------------------
    bisc_geo_shape = BispectrumCalculator_Geo(
        model="FOLPSD",
        geo_expansion="pade",
        fi_vals=F_VALS_FULL * 2,
        k_pkl_pklnw_nl=k_pkl_pklnw_nl,
        z=kwargs["z"],
    )
    Bgeo_shape = bisc_geo_shape.Sugiyama_Bell(
        f=f0,
        bpars=pars_bk,
        k_pkl_pklnw=k_pkl_pklnw,
        k1k2pairs=k1k2_pairs_jax,
        qpar=1.0,
        qper=1.0,
        precision=[10, 10, 10],
        damping="lor",
        multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
        renormalize=True,
        interpolation_method="linear",
        bias_scheme="folps",
    )
    Bgeo_shape = tuple(np.asarray(b) for b in Bgeo_shape)

    rel_shape = np.max(np.abs(Bgeo_shape[0] - Bgeo_nl[0]) / (np.abs(Bgeo_nl[0]) + 1e-30))
    assert rel_shape > 1e-4, (
        f"[geo-shape] Doubling F_VALS_FULL should change B000, but max rel diff is {rel_shape:.2e}"
    )
    print(f"[geo] Doubling shape coeffs changes B000 by max relative {rel_shape:.4e} (PASS)")

    # ----------------------------------------------------------------------------------
    # (6) State helpers: set_pk_nl and per-call kwargs give the same result
    # ----------------------------------------------------------------------------------
    bisc_geo_state = BispectrumCalculator_Geo(
        model="FOLPSD",
        geo_expansion="pade",
        fi_vals=F_VALS_FULL,
    )
    bisc_geo_state.set_pk_nl(k_pkl_pklnw_nl=k_pkl_pklnw_nl, z=kwargs["z"])
    Bgeo_state = bisc_geo_state.Sugiyama_Bell(
        f=f0,
        bpars=pars_bk,
        k_pkl_pklnw=k_pkl_pklnw,
        k1k2pairs=k1k2_pairs_jax,
        qpar=1.0,
        qper=1.0,
        precision=[10, 10, 10],
        damping="lor",
        multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
        renormalize=True,
        interpolation_method="linear",
        bias_scheme="folps",
    )
    Bgeo_state = tuple(np.asarray(b) for b in Bgeo_state)
    for name, ref, got in zip(labels, Bgeo_nl, Bgeo_state):
        np.testing.assert_allclose(
            ref, got, rtol=1e-10, atol=1e-13,
            err_msg=f"[geo-state] set_pk_nl path differs from construction-time state on {name}",
        )
    print("[geo] State consistency (set_pk_nl == constructor kwargs): PASS")

    # ----------------------------------------------------------------------------------
    # (7) JAX / JIT for the geo calculator
    # ----------------------------------------------------------------------------------
    bisc_geo_jit = BispectrumCalculator_Geo(
        model="FOLPSD",
        geo_expansion="pade",
        fi_vals=F_VALS_FULL,
        # no state: k_pkl_pklnw_nl and z are passed per-call so they can be traced
    )

    @jax.jit
    def compute_bispectrum_geo_jit(bpars, f, k_pkl_pklnw_arg, k_pkl_pklnw_nl_arg,
                                    z_arg, k1k2pairs_arg):
        return bisc_geo_jit.Sugiyama_Bell(
            f=f,
            bpars=bpars,
            k_pkl_pklnw=k_pkl_pklnw_arg,
            k1k2pairs=k1k2pairs_arg,
            qpar=1.0,
            qper=1.0,
            precision=[10, 10, 10],
            damping="lor",
            multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
            renormalize=True,
            interpolation_method="linear",
            bias_scheme="folps",
            k_pkl_pklnw_nl=k_pkl_pklnw_nl_arg,
            z=z_arg,
        )

    t0 = time.perf_counter()
    Bgeo_jit_first = compute_bispectrum_geo_jit(
        pars_bk, f0, k_pkl_pklnw, k_pkl_pklnw_nl, jnp.float64(kwargs["z"]),
        k1k2_pairs_jax,
    )
    Bgeo_jit_first = tuple(jax.block_until_ready(b) for b in Bgeo_jit_first)
    t_geo_jit_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    Bgeo_jit_cached = compute_bispectrum_geo_jit(
        pars_bk, f0, k_pkl_pklnw, k_pkl_pklnw_nl, jnp.float64(kwargs["z"]),
        k1k2_pairs_jax,
    )
    Bgeo_jit_cached = tuple(jax.block_until_ready(b) for b in Bgeo_jit_cached)
    t_geo_jit_cached = time.perf_counter() - t0

    geo_speedup, geo_gain = _jit_speedup(t_geo_jit_first, t_geo_jit_cached)

    # Correctness: JIT result must equal the eager result
    for name, eager, jitted in zip(labels, Bgeo_nl, Bgeo_jit_cached):
        np.testing.assert_allclose(
            eager, np.asarray(jitted), rtol=1e-10, atol=1e-13,
            err_msg=f"[geo-jit] JIT result differs from eager for {name}",
        )
    print(f"[geo] JAX JIT geo speedup: {geo_speedup:.2f}x (gain {geo_gain:.1f}%)")

    # ----------------------------------------------------------------------------------
    # (8) Scoccimarro: geo vs non-geo on equilateral triangles, k in [0.02, 0.12]
    #     Includes pade vs poly comparison of the GEO expansion.
    # ----------------------------------------------------------------------------------
    k_scoc_max = 0.12
    k_scoc = np.linspace(0.02, k_scoc_max, num=25)
    triplets_eq = np.column_stack([k_scoc, k_scoc, k_scoc])
    triplets_eq_jax = jnp.asarray(triplets_eq)

    # --- (a) Standard calculator, linear P ----------------------------------------
    B0_scoc_std, B2_scoc_std, B4_scoc_std, _ = bisc_std.Scoccimarro_Bell(
        triplets_eq_jax, f0, pars_bk,
        qpar=1.0, qperp=1.0,
        k_pkl_pklnw=k_pkl_pklnw,
        precision=[10, 10], damping="lor",
        interpolation_method="cubic",
    )
    B0_scoc_std = np.asarray(B0_scoc_std)
    B2_scoc_std = np.asarray(B2_scoc_std)
    B4_scoc_std = np.asarray(B4_scoc_std)

    # --- (b) GEO-FPT, nonlinear P, identity coeffs (isolates the P swap) -----------
    bisc_geo_identity = BispectrumCalculator_Geo(
        model="FOLPSD",
        geo_expansion="pade",
        fi_vals=_IDENTITY_GEO_COEFFS,
        k_pkl_pklnw_nl=k_pkl_pklnw_nl,
        z=kwargs["z"],
    )
    B0_scoc_geo_id, B2_scoc_geo_id, B4_scoc_geo_id, _ = bisc_geo_identity.Scoccimarro_Bell(
        triplets_eq_jax, f0, pars_bk,
        qpar=1.0, qperp=1.0,
        k_pkl_pklnw=k_pkl_pklnw,
        precision=[10, 10], damping="lor",
        interpolation_method="cubic",
    )
    B0_scoc_geo_id = np.asarray(B0_scoc_geo_id)
    B2_scoc_geo_id = np.asarray(B2_scoc_geo_id)
    B4_scoc_geo_id = np.asarray(B4_scoc_geo_id)

    # --- (c) GEO-FPT, nonlinear P, real F_VALS_FULL, PADE expansion ----------------
    bisc_geo_real = BispectrumCalculator_Geo(
        model="FOLPSD",
        geo_expansion="pade",
        fi_vals=F_VALS_FULL,
        k_pkl_pklnw_nl=k_pkl_pklnw_nl,
        z=kwargs["z"],
    )
    B0_scoc_geo_real, B2_scoc_geo_real, B4_scoc_geo_real, _ = bisc_geo_real.Scoccimarro_Bell(
        triplets_eq_jax, f0, pars_bk,
        qpar=1.0, qperp=1.0,
        k_pkl_pklnw=k_pkl_pklnw,
        precision=[10, 10], damping="lor",
        interpolation_method="cubic",
    )
    B0_scoc_geo_real = np.asarray(B0_scoc_geo_real)
    B2_scoc_geo_real = np.asarray(B2_scoc_geo_real)
    B4_scoc_geo_real = np.asarray(B4_scoc_geo_real)

    # --- (d) GEO-FPT, nonlinear P, real F_VALS_FULL, POLY expansion ----------------
    bisc_geo_real_poly = BispectrumCalculator_Geo(
        model="FOLPSD",
        geo_expansion="poly",
        fi_vals=F_VALS_FULL,
        k_pkl_pklnw_nl=k_pkl_pklnw_nl,
        z=kwargs["z"],
    )
    B0_scoc_geo_real_poly, B2_scoc_geo_real_poly, B4_scoc_geo_real_poly, _ = \
        bisc_geo_real_poly.Scoccimarro_Bell(
            triplets_eq_jax, f0, pars_bk,
            qpar=1.0, qperp=1.0,
            k_pkl_pklnw=k_pkl_pklnw,
            precision=[10, 10], damping="lor",
            interpolation_method="cubic",
        )
    B0_scoc_geo_real_poly = np.asarray(B0_scoc_geo_real_poly)
    B2_scoc_geo_real_poly = np.asarray(B2_scoc_geo_real_poly)
    B4_scoc_geo_real_poly = np.asarray(B4_scoc_geo_real_poly)

    # --- Finiteness ----------------------------------------------------------------
    for name, arr in [
        ("B0 std", B0_scoc_std), ("B2 std", B2_scoc_std), ("B4 std", B4_scoc_std),
        ("B0 geo(id)", B0_scoc_geo_id), ("B2 geo(id)", B2_scoc_geo_id),
        ("B4 geo(id)", B4_scoc_geo_id),
        ("B0 geo(real,pade)", B0_scoc_geo_real), ("B2 geo(real,pade)", B2_scoc_geo_real),
        ("B4 geo(real,pade)", B4_scoc_geo_real),
        ("B0 geo(real,poly)", B0_scoc_geo_real_poly),
        ("B2 geo(real,poly)", B2_scoc_geo_real_poly),
        ("B4 geo(real,poly)", B4_scoc_geo_real_poly),
    ]:
        _assert_finite(f"Scoccimarro {name}", arr)

    # --- Sanity: nonlinear-P swap changes B0 ---------------------------------------
    rel_scoc0 = np.max(
        np.abs(B0_scoc_geo_id - B0_scoc_std) / (np.abs(B0_scoc_std) + 1e-30)
    )
    assert rel_scoc0 > 1e-4, (
        f"[geo-scoc] Scoccimarro B0 with nonlinear P should differ from standard, "
        f"got max rel diff {rel_scoc0:.2e}"
    )
    print(f"[geo] Scoccimarro B0 nonlinear-P swap: max relative {rel_scoc0:.4e} (PASS)")

    # --- Sanity: real F_VALS_FULL shape correction changes B0 (pade) ---------------
    rel_scoc_real = np.max(
        np.abs(B0_scoc_geo_real - B0_scoc_geo_id) / (np.abs(B0_scoc_geo_id) + 1e-30)
    )
    if rel_scoc_real < 1e-6:
        print(f"[geo] NOTE: Scoccimarro B0 with F_VALS_FULL (pade) matches identity "
              f"(max rel diff {rel_scoc_real:.2e}); the calibration may be flat.")
    else:
        print(f"[geo] Scoccimarro B0 shape correction (F_VALS_FULL pade vs identity): "
              f"max relative {rel_scoc_real:.4e}")

    # --- Sanity: pade vs poly difference -------------------------------------------
    rel_pade_poly = np.max(
        np.abs(B0_scoc_geo_real_poly - B0_scoc_geo_real) /
        (np.abs(B0_scoc_geo_real) + 1e-30)
    )
    print(f"[geo] Scoccimarro B0 pade vs poly (F_VALS_FULL): "
          f"max relative {rel_pade_poly:.4e}")

    # --- Plot ----------------------------------------------------------------------
    # Style follows the example notebook (cells 27/30): semilogy, k1*k2*k3*B,
    # marker+line, k-range capped at 0.12.
    scoc_fig = output_dir / "scoccimarro_geo_vs_std.png"
    _plot_scoccimarro_comparison(
        k_scoc,
        curves=[
            {"label": "standard (linear P)",
             "color": "black", "ls": "-", "marker": "o",
             "b0": B0_scoc_std, "b2": B2_scoc_std, "b4": B4_scoc_std},
            {"label": "GEO-FPT, nonlinear P, identity coeffs",
             "color": "tab:red", "ls": "--", "marker": "s",
             "b0": B0_scoc_geo_id, "b2": B2_scoc_geo_id, "b4": B4_scoc_geo_id},
            {"label": "GEO-FPT, nonlinear P, F_VALS_FULL (pade)",
             "color": "tab:blue", "ls": ":", "marker": "^",
             "b0": B0_scoc_geo_real, "b2": B2_scoc_geo_real, "b4": B4_scoc_geo_real},
            {"label": "GEO-FPT, nonlinear P, F_VALS_FULL (poly)",
             "color": "tab:green", "ls": "-.", "marker": "d",
             "b0": B0_scoc_geo_real_poly, "b2": B2_scoc_geo_real_poly,
             "b4": B4_scoc_geo_real_poly},
        ],
        outpath=scoc_fig,
        k_max=k_scoc_max,
        triangle_label="equilateral",
    )
    print(f"  Scoccimarro figure:    {scoc_fig}")

    # ----------------------------------------------------------------------------------
    # (9) Geo helper sanity checks
    #       (a) identity coeffs -> geo_fac == 1  (structural property)
    #       (b) real F_VALS_FULL -> finite, symmetric, shape-dependent (physics-facing)
    # ----------------------------------------------------------------------------------
    # (a) Identity coeffs -> geo_fac == 1 exactly
    af_id = _np_native.asarray(interpolate_geo_coeffs(0.5, _IDENTITY_GEO_COEFFS))
    assert af_id.shape == (5,), f"interpolate_geo_coeffs returned wrong shape {af_id.shape}"
    _np_native.testing.assert_allclose(
        af_id, _np_native.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        rtol=1e-12, atol=1e-14,
        err_msg="Identity table should give af = [1, 0, 0, 0, 0]")

    k_eq = _np_native.array([0.1, 0.1, 0.1])
    gf_poly_id = float(_np_native.asarray(geo_fac(k_eq[0], k_eq[1], k_eq[2], af_id)))
    gf_pade_id = float(_np_native.asarray(geo_fac_pade(k_eq[0], k_eq[1], k_eq[2], af_id)))
    _np_native.testing.assert_allclose(
        gf_poly_id, 1.0, rtol=1e-12, atol=1e-14,
        err_msg="geo_fac with identity coeffs must be exactly 1")
    _np_native.testing.assert_allclose(
        gf_pade_id, 1.0, rtol=1e-12, atol=1e-14,
        err_msg="geo_fac_pade with identity coeffs must be exactly 1")

    # (b) Real F_VALS_FULL: finite, symmetric, shape-dependent
    af_real = _np_native.asarray(interpolate_geo_coeffs(kwargs["z"], F_VALS_FULL))
    assert af_real.shape == (5,), f"F_VALS_FULL interpolation gave shape {af_real.shape}"
    assert _np_native.all(_np_native.isfinite(af_real)), \
        "F_VALS_FULL interpolation returned non-finite values"

    gf_eq_real = float(_np_native.asarray(geo_fac(0.10, 0.10, 0.10, af_real)))
    gf_sq_real = float(_np_native.asarray(geo_fac(0.05, 0.05, 0.099, af_real)))
    assert _np_native.isfinite(gf_eq_real) and _np_native.isfinite(gf_sq_real), \
        "geo_fac with F_VALS_FULL returned non-finite values"

    # Symmetry check: geo_fac must be invariant under permutation of (ka, kb, kc)
    gf_perm_real = float(_np_native.asarray(geo_fac(0.05, 0.099, 0.05, af_real)))
    _np_native.testing.assert_allclose(
        gf_perm_real, gf_sq_real, rtol=1e-12, atol=1e-14,
        err_msg="geo_fac is not symmetric under permutation of (ka, kb, kc)")

    print("[geo] Module-level helpers (interpolate_geo_coeffs / geo_fac / geo_fac_pade): PASS")
    print(f"[geo] F_VALS_FULL(z={kwargs['z']}) = {af_real}")

    # ----------------------------------------------------------------------------------
    # (10) GEO-FPT Sugiyama-Bell figure — all six multipoles, matching the
    #      layout and units of _plot_bispectrum.
    # ----------------------------------------------------------------------------------
    geo_fig = output_dir / "bispectrum_geo_jax.png"
    _plot_bispectrum_geo(
        k_ev,
        curves=[
            {"label": "standard (linear P)",
             "ls": "-",
             "B": Bstd},
            {"label": "GEO-FPT, nonlinear P, F_VALS_FULL",
             "ls": "--",
             "B": Bgeo_nl},
            {"label": r"GEO-FPT, nonlinear P, $2\times$F_VALS_FULL",
             "ls": ":",
             "B": Bgeo_shape},
        ],
        outpath=geo_fig,
    )

    # ----------------------------------------------------------------------------------
    # (11) Save results and print timing
    # ----------------------------------------------------------------------------------
    np.savez(
        results_npz,
        k=np.asarray(k_jax),
        p0=p0_plot, p2=p2_plot, p4=p4_plot,
        p0_marg=np.asarray(p0_marg),
        p2_marg=np.asarray(p2_marg),
        p4_marg=np.asarray(p4_marg),
        k_bis=np.asarray(k_ev),
        b000=b000_plot, b110=b110_plot, b220=b220_plot,
        b202=b202_plot, b022=b022_plot, b112=b112_plot,
        # === GEO-FPT ===
        k_pkl_pklnw_nl_k=np.asarray(k_pkl_pklnw_nl[0]),
        k_pkl_pklnw_nl_pk=np.asarray(k_pkl_pklnw_nl[1]),
        k_pkl_pklnw_nl_pknw=np.asarray(k_pkl_pklnw_nl[2]),
        # All six geo multipoles for both variants
        b000_geo_pl=Bgeo_nl[0], b110_geo_pl=Bgeo_nl[1], b220_geo_pl=Bgeo_nl[2],
        b202_geo_pl=Bgeo_nl[3], b022_geo_pl=Bgeo_nl[4], b112_geo_pl=Bgeo_nl[5],
        b000_geo_sh=Bgeo_shape[0], b110_geo_sh=Bgeo_shape[1], b220_geo_sh=Bgeo_shape[2],
        b202_geo_sh=Bgeo_shape[3], b022_geo_sh=Bgeo_shape[4], b112_geo_sh=Bgeo_shape[5],
        # Scoccimarro: equilateral triangles up to k = 0.12
        k_scoc=k_scoc,
        b0_scoc_std=B0_scoc_std,
        b2_scoc_std=B2_scoc_std,
        b4_scoc_std=B4_scoc_std,
        b0_scoc_geo_id=B0_scoc_geo_id,
        b2_scoc_geo_id=B2_scoc_geo_id,
        b4_scoc_geo_id=B4_scoc_geo_id,
        b0_scoc_geo_real=B0_scoc_geo_real,
        b2_scoc_geo_real=B2_scoc_geo_real,
        b4_scoc_geo_real=B4_scoc_geo_real,
        b0_scoc_geo_real_poly=B0_scoc_geo_real_poly,
        b2_scoc_geo_real_poly=B2_scoc_geo_real_poly,
        b4_scoc_geo_real_poly=B4_scoc_geo_real_poly,
    )

    _print_timing_table(
        [
            ("Pnow precompute", f"{t_pknow:.3f} s"),
            ("Matrix build", f"{t_matrix:.3f} s"),
            ("Loop table setup", f"{t_loop_setup:.3f} s"),
            ("PK JIT first run (compile+exec)", f"{t_pk_jit_first:.3f} s"),
            ("PK JIT cached run", f"{t_pk_jit_cached:.3f} s"),
            ("PK speedup", f"{pk_speedup:.2f}x"),
            ("PK improvement", f"{pk_gain:.1f}%"),
            ("BK JIT first run (compile+exec)", f"{t_bk_jit_first:.3f} s"),
            ("BK JIT cached run", f"{t_bk_jit_cached:.3f} s"),
            ("BK speedup", f"{bk_speedup:.2f}x"),
            ("BK improvement", f"{bk_gain:.1f}%"),
            # === GEO-FPT ===
            ("GEO P_nl table build", f"{t_geo_pk_table:.3f} s"),
            ("GEO BK JIT first run (compile+exec)", f"{t_geo_jit_first:.3f} s"),
            ("GEO BK JIT cached run", f"{t_geo_jit_cached:.3f} s"),
            ("GEO BK speedup", f"{geo_speedup:.2f}x"),
            ("GEO BK improvement", f"{geo_gain:.1f}%"),
        ]
    )
    print(f"  Numeric results:        {results_npz}")
    print(f"  Power spectrum figure:  {power_fig}")
    print(f"  Bispectrum figure:      {bispec_fig}")
    print(f"  GEO-FPT figure:         {geo_fig}")
    print(f"  Scoccimarro figure:     {scoc_fig}")
    print()
    print("[geo] All GEO-FPT tests passed.")


if __name__ == "__main__":
    run_test_folps_jax()