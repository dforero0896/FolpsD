"""
Standalone script version of the FOLPS-JAX example notebook (bispectrum section).

Reproduces the bispectrum workflow from ``example_folps_jax.ipynb`` and extends
it with GEO-FPT support (``pade`` and ``poly`` expansions of the shape
correction).  Every figure is saved to ``example_outputs/`` as a PNG instead of
being shown interactively.

Main plots are accompanied by a ratio panel showing GEO-FPT / standard.

Usage:

    python example_folps_jax_script.py

Assumes the following input files live next to this script (as in the notebook):

    - pk_linear_simtocmass.txt   (linear P(k), Mpc/h units)
    - k1k2k3.txt                 (Scoccimarro triangle triplets)
    - ../folps/...               (folps package, imported via sys.path)
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Point Python at YOUR LOCAL folps checkout instead of the site-packages one.
#
# Priority order for the local path (first match wins):
#   1. FOLPS_LOCAL_PATH env var
#   2. ../  relative to THIS SCRIPT's directory
#   3. the script's own directory
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent

_env_path = os.environ.get("FOLPS_LOCAL_PATH")
_candidates = []
if _env_path:
    _candidates.append(Path(_env_path).expanduser().resolve())
_candidates.append((_script_dir / "..").resolve())
_candidates.append(_script_dir)

FOLPS_LOCAL_PATH = None
for cand in _candidates:
    if (cand / "folps").is_dir() or (cand / "folps.py").is_file():
        FOLPS_LOCAL_PATH = cand
        break

if FOLPS_LOCAL_PATH is None:
    raise FileNotFoundError(
        "Could not locate a local folps checkout. Checked:\n"
        + "\n".join(f"  {c}" for c in _candidates)
        + "\nSet FOLPS_LOCAL_PATH to the directory containing 'folps/'."
    )

sys.path.insert(0, str(FOLPS_LOCAL_PATH))

# Purge any cached folps modules so the next import resolves to the local copy.
for _mod in [m for m in list(sys.modules) if m == "folps" or m.startswith("folps.")]:
    del sys.modules[_mod]

# Backend must be set before importing folps internals.
os.environ["FOLPS_BACKEND"] = "jax"

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless: save to disk, no interactive windows
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import folps as FOLPS

_imported_from = Path(FOLPS.__file__).resolve().parent
print(f"[folps] imported from: {_imported_from}")
if "site-packages" in str(_imported_from):
    print(
        "[folps] WARNING: still importing from site-packages! "
        "Set FOLPS_LOCAL_PATH explicitly to your local checkout, or run:\n"
        "    pip uninstall folps\n"
        "    pip install -e /path/to/your/folps"
    )

from folps import BispectrumCalculator_Geo, F_VALS_FULL


# ===========================================================================
# Paths
# ===========================================================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "example_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(fig, name):
    """Save a matplotlib figure to OUTPUT_DIR/<name>.png and close it."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path.name}")


# ===========================================================================
# Small plotting helpers
# ===========================================================================
def _safe_ratio(num, den, min_abs_den=1e-30):
    """Elementwise num/den with points masked (NaN) where |den| is too small."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    good = np.abs(den) > min_abs_den
    out[good] = num[good] / den[good]
    return out


def _safe_positive(y):
    """Return y with nonpositive values replaced by NaN (for semilogy)."""
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y, np.nan)


def _set_robust_ylim(ax, arrays, pad_frac=0.05):
    """Set y-limits from the 1st/99th percentile of the finite data.

    Keeps ratio panels readable when a few points spike due to a near-zero
    denominator.
    """
    finite = [np.asarray(a, dtype=float)[np.isfinite(a)] for a in arrays]
    finite = [a for a in finite if a.size > 0]
    if not finite:
        return
    all_vals = np.concatenate(finite)
    lo, hi = np.percentile(all_vals, [1.0, 99.0])
    if hi <= lo:
        lo, hi = lo - 0.1, hi + 0.1
    span = hi - lo
    ax.set_ylim(lo - pad_frac * span, hi + pad_frac * span)


# ===========================================================================
# 1. Cosmology (CMASS NGC)
# ===========================================================================
omega_b = 0.02242
omega_cdm = 0.117
omega_ncdm = 0.0
h = 0.67
z_pk = 0.57

CosmoParams = [z_pk, omega_b, omega_cdm, omega_ncdm, h]

kwargs = {
    "z": z_pk,
    "h": h,
    "Omega_m": (omega_cdm + omega_b + omega_ncdm) / h**2,
    "fnu": 0.02,
}

f0 = FOLPS.get_f0(z_pk, kwargs["Omega_m"])
print(f"[cosmo] Omega_m = {kwargs['Omega_m']:.6f},  f0 = {f0:.6f}")


# ===========================================================================
# 2. Linear power spectrum
# ===========================================================================
data_path = ROOT / "pk_linear_simtocmass.txt"
if not data_path.exists():
    alt = ROOT / "inputpkT.txt"
    if alt.exists():
        data_path = alt
    else:
        raise FileNotFoundError(
            f"Could not find pk_linear_simtocmass.txt or inputpkT.txt in {ROOT}"
        )

k_arr, pk_arr = np.loadtxt(data_path, unpack=True)
classy = {"k": k_arr, "pk": pk_arr}
print(
    f"[pk]    loaded {data_path.name}: {len(k_arr)} points, "
    f"k in [{k_arr.min():.3e}, {k_arr.max():.3e}] h/Mpc"
)


# ===========================================================================
# 3. Nuisance parameters (power spectrum)
# ===========================================================================
b1 = 1.9
b2 = 8 / 21 * (b1 - 1)
bs2 = -4 / 7 * (b1 - 1)
b3nl = 32 / 315 * (b1 - 1)

alpha0, alpha2, alpha4 = 0, 0, 0
ctilde = 0
X_FoG = 0

alphashot0 = 0
alphashot2 = 0
PshotP = 0

NuisanParams = [
    b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde,
    alphashot0, alphashot2, PshotP, X_FoG,
]


# ===========================================================================
# 4. M matrices (cosmology-independent, computed once)
# ===========================================================================
print("[matrix] calculating M matrices...")
matrix = FOLPS.MatrixCalculator(A_full=True, save_dir="output_matrices")
mmatrices = matrix.get_mmatrices()


# ===========================================================================
# 5. AP parameters (identity here since Omfid < 0)
# ===========================================================================
Omfid = -1
qpar, qperp = FOLPS.qpar_qperp(
    Omega_fid=Omfid, Omega_m=kwargs["Omega_m"], z_pk=kwargs["z"]
)
print(f"[AP]    qpar = {qpar}, qperp = {qperp}")


# ===========================================================================
# 6. Nonlinear P setup: pknow and loop table
# ===========================================================================
nonlinear = FOLPS.NonLinearPowerSpectrumCalculator(
    mmatrices=mmatrices, kernels="fk", **kwargs
)

k_, pk_ = FOLPS.extrapolate_pklin(k=classy["k"], pk=classy["pk"])
pknow_result = FOLPS.get_pknow_jax(k=k_, pk=pk_, h=kwargs["h"])
if isinstance(pknow_result, tuple):
    k_pknow, pknow = pknow_result
else:
    k_pknow, pknow = k_, pknow_result
if len(pknow) != len(classy["k"]):
    pknow = np.interp(classy["k"], np.asarray(k_pknow), np.asarray(pknow))
pknow = jnp.asarray(pknow)
print("[pknow] done")

print("[loop]  computing loop table...")
table, table_nonwiggles = nonlinear.calculate_loop_table(
    k=classy["k"], pklin=classy["pk"], pknow=pknow, cosmo=None, **kwargs
)
print("[loop]  done")


# ===========================================================================
# 7. Bispectrum nuisance parameters and linear P triplets
# ===========================================================================
Pshot = 0  # same as PshotP * alphashot0 if the bispectrum were 1-loop
Bshot = 0
c1 = 0
c2 = 0
X_FoG_bk = 1
bpars = [b1, b2, bs2, c1, c2, Bshot, Pshot, X_FoG_bk]

linear = nonlinear.get_linear(
    classy["k"], classy["pk"], pknow=None, cosmo=None, **kwargs
)
k_pkl_pklnw = np.array([linear["k"], linear["pk_l"], linear["pk_l_NW"]])


# ===========================================================================
# 8. GEO-FPT nonlinear P table
# ===========================================================================
print("[geo]   building nonlinear P table...")
k_pkl_pklnw_nl = nonlinear.get_geofpt_pk_tables(
    k=classy["k"],
    pklin=classy["pk"],
    pknow=pknow,
    cosmo=None,
    **kwargs,
)

pk_lin = np.asarray(k_pkl_pklnw[1])
pk_nl = np.asarray(k_pkl_pklnw_nl[1])
max_rel_diff = np.max(np.abs(pk_nl / pk_lin - 1.0))
print(f"[geo]   max |P_nl/P_lin - 1| = {max_rel_diff:.4e}")


# ===========================================================================
# 9. Sugiyama-Bell multipoles: standard, geo-pade, geo-poly
# ===========================================================================
print("[sugi]  computing Sugiyama-Bell multipoles...")
k_ev = np.linspace(0.01, 0.2, num=40)
k1k2T = np.vstack([k_ev, k_ev]).T  # k1 = k2 pairs

bispectrum_std = FOLPS.BispectrumCalculator(model="FOLPSD")

sugi_kwargs = dict(
    f=f0,
    bpars=bpars,
    k_pkl_pklnw=k_pkl_pklnw,
    k1k2pairs=k1k2T,
    qpar=1,
    qper=1,
    precision=[8, 10, 10],
    renormalize=True,
    damping="lor",
    interpolation_method="linear",
    bias_scheme="folps",
    multipoles=["B000", "B110", "B220", "B202", "B022", "B112"],
)

B_std = bispectrum_std.Sugiyama_Bell(**sugi_kwargs)

bispectrum_geo_pade = BispectrumCalculator_Geo(
    model="FOLPSD",
    geo_expansion="pade",
    fi_vals=F_VALS_FULL,
    k_pkl_pklnw_nl=k_pkl_pklnw_nl,
    z=z_pk,
)
B_pade = bispectrum_geo_pade.Sugiyama_Bell(**sugi_kwargs)

bispectrum_geo_poly = BispectrumCalculator_Geo(
    model="FOLPSD",
    geo_expansion="poly",
    fi_vals=F_VALS_FULL,
    k_pkl_pklnw_nl=k_pkl_pklnw_nl,
    z=z_pk,
)
B_poly = bispectrum_geo_poly.Sugiyama_Bell(**sugi_kwargs)
print("[sugi]  done")


# ---------------------------------------------------------------------------
# Plot: Sugiyama-Bell multipoles (top row) + GEO/std ratio (bottom row)
#   Colors encode multipole; linestyles encode model.
# ---------------------------------------------------------------------------
xmax = 0.2
left_specs = [("B000", 0, "red"), ("B110", 1, "blue"), ("B220", 2, "green")]
right_specs = [("B202", 3, "red"), ("B022", 4, "green"), ("B112", 5, "blue")]

models = [
    {"label": "standard (linear P)",         "ls": "-",  "B": B_std,  "ref": False},
    {"label": "GEO-FPT (pade, nonlinear P)", "ls": "--", "B": B_pade, "ref": True},
    {"label": "GEO-FPT (poly, nonlinear P)", "ls": ":",  "B": B_poly, "ref": True},
]

fig, axs = plt.subplots(
    2, 2, figsize=(14, 9),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    sharex=True,
)


def _draw_sugi_panel(ax, ax_ratio, specs, title):
    # --- main panel: k^2 * B(k,k) ------------------------------------------
    for name, idx, color in specs:
        for m in models:
            ax.plot(
                k_ev, k_ev**2 * np.asarray(m["B"][idx]),
                color=color, ls=m["ls"], lw=1.5,
            )
    ax.axhline(0.0, color="grey", lw=0.6, alpha=0.5)

    # Legend 1: multipole -> color
    c_handles = [
        Line2D([0], [0], color=col, lw=2.2, label=rf"${nm}$")
        for nm, _, col in specs
    ]
    leg_c = ax.legend(
        handles=c_handles, loc="upper left",
        fontsize=10, title="Multipole", frameon=True,
    )
    leg_c.get_title().set_fontsize(10)
    ax.add_artist(leg_c)

    # Legend 2: model -> linestyle
    s_handles = [
        Line2D([0], [0], color="black", ls=m["ls"], lw=2.0, label=m["label"])
        for m in models
    ]
    leg_s = ax.legend(
        handles=s_handles, loc="upper right",
        fontsize=9, title="Model", frameon=True,
    )
    leg_s.get_title().set_fontsize(9)

    ax.set_ylabel(r"$k^2 B(k,k) \, [h^{-1}\, Mpc]^4$", fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim([k_ev[0], xmax])

    # --- ratio panel: GEO / standard --------------------------------------
    ratio_arrays = []
    for name, idx, color in specs:
        std = np.asarray(B_std[idx], dtype=float)
        for m in models:
            if not m["ref"]:
                continue
            num = np.asarray(m["B"][idx], dtype=float)
            ratio = _safe_ratio(num, std)
            ratio_arrays.append(ratio)
            ax_ratio.plot(
                k_ev, ratio,
                color=color, ls=m["ls"], lw=1.4,
            )
    ax_ratio.axhline(1.0, color="grey", lw=0.8, ls="-", alpha=0.7)
    ax_ratio.set_ylabel(r"GEO / std", fontsize=12)
    ax_ratio.set_xlabel(r"$k \, [h\, Mpc^{-1}]$", fontsize=13)
    ax_ratio.set_xlim([k_ev[0], xmax])
    ax_ratio.grid(True, which="both", alpha=0.25)
    _set_robust_ylim(ax_ratio, ratio_arrays)
    # Hide top x-ticks on the ratio panel is unnecessary since sharex=True


_draw_sugi_panel(axs[0, 0], axs[1, 0], left_specs,  "B000, B110, B220")
_draw_sugi_panel(axs[0, 1], axs[1, 1], right_specs, "B202, B022, B112")
plt.tight_layout()
_savefig(fig, "sugiyama_multipoles")


# ===========================================================================
# 10. Scoccimarro basis
# ===========================================================================
print("[scoc]  loading triangle triplets...")
triplets_path = ROOT / "k1k2k3.txt"
if not triplets_path.exists():
    raise FileNotFoundError(f"Triangle file k1k2k3.txt not found in {ROOT}")

k1T, k2T, k3T = np.loadtxt(triplets_path, unpack=True)
k1k2k3triplets = np.column_stack((k1T, k2T, k3T))
print(f"[scoc]  {len(k1T)} triplets loaded")

print("[scoc]  computing Scoccimarro multipoles (standard)...")
B0_std, B2_std, B4_std, _ = bispectrum_std.Scoccimarro_Bell(
    k1k2k3triplets, f0, bpars, qpar, qperp, k_pkl_pklnw,
    precision=[10, 10], damping="lor", interpolation_method="cubic",
)

print("[scoc]  computing Scoccimarro multipoles (geo pade)...")
B0_pade, B2_pade, B4_pade, _ = bispectrum_geo_pade.Scoccimarro_Bell(
    k1k2k3triplets, f0, bpars, qpar, qperp, k_pkl_pklnw,
    precision=[10, 10], damping="lor", interpolation_method="cubic",
)

print("[scoc]  computing Scoccimarro multipoles (geo poly)...")
B0_poly, B2_poly, B4_poly, _ = bispectrum_geo_poly.Scoccimarro_Bell(
    k1k2k3triplets, f0, bpars, qpar, qperp, k_pkl_pklnw,
    precision=[10, 10], damping="lor", interpolation_method="cubic",
)

B0_std, B2_std, B4_std = map(np.asarray, (B0_std, B2_std, B4_std))
B0_pade, B2_pade, B4_pade = map(np.asarray, (B0_pade, B2_pade, B4_pade))
B0_poly, B2_poly, B4_poly = map(np.asarray, (B0_poly, B2_poly, B4_poly))


# ---------------------------------------------------------------------------
# Plot: Scoccimarro B0, B2, B4 vs triangle index, capped at k = 0.12.
#   Top row: k1*k2*k3*B (semilogy, marker+line).
#   Bottom row: GEO/std ratio.
# ---------------------------------------------------------------------------
k_cap = 0.12
mask = np.max(np.column_stack([k1T, k2T, k3T]), axis=1) <= k_cap
print(
    f"[scoc]  keeping {np.count_nonzero(mask)}/{len(k1T)} triangles "
    f"with max(k1,k2,k3) <= {k_cap}"
)

x_plot = np.arange(np.count_nonzero(mask))
kfac = (k1T * k2T * k3T)[mask]

curves_scoc = [
    {
        "label": "standard (linear P)",
        "color": "black", "ls": "-", "marker": "o",
        "B": (B0_std[mask], B2_std[mask], B4_std[mask]),
        "ref": True,
    },
    {
        "label": "GEO-FPT (pade, nonlinear P)",
        "color": "tab:blue", "ls": "--", "marker": "^",
        "B": (B0_pade[mask], B2_pade[mask], B4_pade[mask]),
        "ref": False,
    },
    {
        "label": "GEO-FPT (poly, nonlinear P)",
        "color": "tab:green", "ls": "-.", "marker": "d",
        "B": (B0_poly[mask], B2_poly[mask], B4_poly[mask]),
        "ref": False,
    },
]

fig, axs = plt.subplots(
    2, 3, figsize=(18, 9),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    sharex=True,
)

for col, (ell, col_idx) in enumerate(zip([0, 2, 4], [0, 1, 2])):
    ax = axs[0, col]
    ax_ratio = axs[1, col]

    # --- main panel --------------------------------------------------------
    std_vals = kfac * np.asarray(curves_scoc[0]["B"][col_idx])
    for c in curves_scoc:
        y = kfac * np.asarray(c["B"][col_idx])
        ax.semilogy(
            x_plot, _safe_positive(y),
            color=c["color"], ls=c["ls"], marker=c["marker"],
            ms=4, lw=1.4, label=c["label"],
        )
    ax.set_ylabel(rf"$k_1 k_2 k_3 B_{{{ell}}} \ \ [(Mpc/h)^3]$", fontsize=16)
    ax.set_title(rf"$B_{{{ell}}}$   (triangles with $k \leq {k_cap}$)", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    if ell == 0:
        ax.legend(fontsize=10, loc="best")

    # --- ratio panel -------------------------------------------------------
    ratio_arrays = []
    for c in curves_scoc:
        if c["ref"]:
            continue
        y = kfac * np.asarray(c["B"][col_idx])
        ratio = _safe_ratio(y, std_vals)
        ratio_arrays.append(ratio)
        ax_ratio.plot(
            x_plot, ratio,
            color=c["color"], ls=c["ls"], marker=c["marker"],
            ms=3, lw=1.3,
        )
    ax_ratio.axhline(1.0, color="grey", lw=0.8, ls="-", alpha=0.7)
    ax_ratio.set_ylabel(r"GEO / std", fontsize=12)
    ax_ratio.set_xlabel(r"Triangle index", fontsize=13)
    ax_ratio.grid(True, which="both", alpha=0.25)
    _set_robust_ylim(ax_ratio, ratio_arrays)

plt.tight_layout()
_savefig(fig, "scoccimarro_triangle_index")


# ===========================================================================
# 11. Equilateral / isosceles extraction
# ===========================================================================
equilateral_n = []
isosceles_n = []
for i, (a, b, c) in enumerate(k1k2k3triplets):
    if a == b == c:
        equilateral_n.append(i)
    elif a == b or a == c or b == c:
        isosceles_n.append(i)
equilateral_n = np.array(equilateral_n)
isosceles_n = np.array(isosceles_n)
print(f"[tri]   {len(equilateral_n)} equilateral, {len(isosceles_n)} isosceles")


def _stack(idx_arr, values):
    """Return a 2xN array (index, value) for the given triangle indices."""
    return np.stack((idx_arr, values[idx_arr]))


B0_eq_std, B2_eq_std, B4_eq_std = (
    _stack(equilateral_n, B0_std),
    _stack(equilateral_n, B2_std),
    _stack(equilateral_n, B4_std),
)
B0_eq_pade, B2_eq_pade, B4_eq_pade = (
    _stack(equilateral_n, B0_pade),
    _stack(equilateral_n, B2_pade),
    _stack(equilateral_n, B4_pade),
)
B0_eq_poly, B2_eq_poly, B4_eq_poly = (
    _stack(equilateral_n, B0_poly),
    _stack(equilateral_n, B2_poly),
    _stack(equilateral_n, B4_poly),
)


# ---------------------------------------------------------------------------
# Plot: one figure per multipole, with a GEO/std ratio panel underneath.
#   Full triangle-index range (as in the notebook), equilateral markers
#   highlighted with open circles/squares/diamonds.
# ---------------------------------------------------------------------------
kfac_full = k1T * k2T * k3T
x = np.arange(len(B0_std))


for ell, (B_std_full, B_pade_full, B_poly_full,
          B_std_eq, B_pade_eq, B_poly_eq) in zip(
    [0, 2, 4],
    [
        (B0_std, B0_pade, B0_poly, B0_eq_std, B0_eq_pade, B0_eq_poly),
        (B2_std, B2_pade, B2_poly, B2_eq_std, B2_eq_pade, B2_eq_poly),
        (B4_std, B4_pade, B4_poly, B4_eq_std, B4_eq_pade, B4_eq_poly),
    ],
):
    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(15, 10),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    # --- main panel --------------------------------------------------------
    y_std = kfac_full * B_std_full
    y_pade = kfac_full * B_pade_full
    y_poly = kfac_full * B_poly_full

    ax.semilogy(x, _safe_positive(y_std),
                "o-", color="black", label="standard (linear P)")
    ax.semilogy(x, _safe_positive(y_pade),
                "^--", color="tab:blue", label="GEO-FPT (pade, nonlinear P)")
    ax.semilogy(x, _safe_positive(y_poly),
                "d-.", color="tab:green", label="GEO-FPT (poly, nonlinear P)")

    # Overlay equilateral triangles as open markers
    ax.semilogy(
        B_std_eq[0], kfac_full[equilateral_n] * B_std_eq[1],
        "o", color="black", ms=7, mfc="none", mew=1.5,
    )
    ax.semilogy(
        B_pade_eq[0], kfac_full[equilateral_n] * B_pade_eq[1],
        "^", color="tab:blue", ms=7, mfc="none", mew=1.5,
    )
    ax.semilogy(
        B_poly_eq[0], kfac_full[equilateral_n] * B_poly_eq[1],
        "d", color="tab:green", ms=7, mfc="none", mew=1.5,
    )

    ax.set_ylabel(rf"$k_1 k_2 k_3 B_{{{ell}}} \quad [(Mpc/h)^3]$", fontsize=20)
    ax.set_title(rf"$B_{{{ell}}}$ — all triangles", fontsize=15, pad=10)
    ax.legend(fontsize=11, loc="best")

    # --- ratio panel -------------------------------------------------------
    ratio_pade = _safe_ratio(y_pade, y_std)
    ratio_poly = _safe_ratio(y_poly, y_std)

    ax_ratio.plot(x, ratio_pade, "^--", color="tab:blue",  ms=4, lw=1.3,
                  label="pade / std")
    ax_ratio.plot(x, ratio_poly, "d-.", color="tab:green", ms=4, lw=1.3,
                  label="poly / std")
    ax_ratio.axhline(1.0, color="grey", lw=0.8, ls="-", alpha=0.7)
    # Highlight equilateral positions on the ratio panel too
    ax_ratio.plot(
        equilateral_n, ratio_pade[equilateral_n],
        "^", color="tab:blue", ms=8, mfc="none", mew=1.5,
    )
    ax_ratio.plot(
        equilateral_n, ratio_poly[equilateral_n],
        "d", color="tab:green", ms=8, mfc="none", mew=1.5,
    )
    ax_ratio.set_ylabel(r"GEO / std", fontsize=16)
    ax_ratio.set_xlabel(r"Triangle index", fontsize=14)
    ax_ratio.grid(True, which="both", alpha=0.25)
    ax_ratio.legend(fontsize=10, loc="best", ncols=2)
    _set_robust_ylim(ax_ratio, [ratio_pade, ratio_poly])

    plt.tight_layout()
    _savefig(fig, f"scoccimarro_B{ell}_equilateral")


print()
print("=" * 70)
print(f"All figures saved to: {OUTPUT_DIR}")
print("=" * 70)