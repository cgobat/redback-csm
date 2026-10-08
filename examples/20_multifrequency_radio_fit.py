"""
20_multifrequency_radio_fit.py — Fit multi-frequency radio data in one call.

Demonstrates:
  - Passing a per-point ``frequency`` array (one value per data point, same
    length as ``time``) so each likelihood evaluation is a single hydro run,
    including epochs where several frequencies share the same phase
  - The optional spectral-shape keywords of the radio models:
      * ``cooling`` / ``log_nu_c_scale``   synchrotron cooling break
      * ``ssa_alpha_thick`` / ``ssa_smoothing``  broadened SSA turnover
      * ``log_tau_ff`` / ``ff_time_index``  external free-free absorption
  - A maximum-a-posteriori fit with redback on synthetic data

Run:
    python examples/20_multifrequency_radio_fit.py

The script generates synthetic data in memory; no files are read.
"""

import numpy as np
import matplotlib.pyplot as plt
import bilby
import redback

from redback_csm.models import static_spline_csm_bpl_radio

MODEL = "static_spline_csm_bpl_radio"
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic observing campaign: several frequencies per epoch, so phases repeat
# ---------------------------------------------------------------------------
epochs = {                       # phase (d): frequencies (GHz)
    10.0: [10, 22, 33, 100, 250],
    60.0: [3, 6, 10, 33, 100],
    250.0: [1.5, 3, 6, 10, 22],
    800.0: [1.5, 3, 6, 10, 22, 100],
    1500.0: [1.5, 3, 6, 10, 15, 22, 33],
    2500.0: [1.5, 3, 6, 10, 15, 22],
}
time = np.concatenate([np.full(len(f), t) for t, f in epochs.items()])
frequency = np.concatenate([np.asarray(f, float) for f in epochs.values()]) * 1e9

# Truth: a steady-wind-like CSM (rho ~ r^-2) with a density enhancement near
# 1e17 cm, plus a cooling break and a broadened (inhomogeneous) SSA turnover.
# The SSA peak moves from ~30 GHz at 10 d to ~1 GHz at late times.
log_r_nodes = np.linspace(14.0, 17.7, 8)
truth = dict(
    redshift=0.004, log_r_inner=14.0, log_r_outer=17.7,
    **{f"log_rho_{i}": -15.0 - 2.0 * (lr - 14.0) for i, lr in enumerate(log_r_nodes)},
    delta_sn=1.0, nn_sn=10.0, mej_sn=3.0, esn=1.0, eff=0.5,
    logepsb=-1.5, logepse=-1.0, p=2.6,
    cooling=True, log_nu_c_scale=0.0, ssa_alpha_thick=1.2, ssa_smoothing=2.0,
)
truth["log_rho_6"] += 0.8

# One call evaluates every (time, frequency) pair, repeated phases included.
flux_true = static_spline_csm_bpl_radio(time=time, frequency=frequency, **truth)

# Sanity check: identical to calling the model once per frequency.
for nu in np.unique(frequency):
    sel = frequency == nu
    single = static_spline_csm_bpl_radio(time=time[sel], frequency=nu, **truth)
    np.testing.assert_allclose(flux_true[sel], single, rtol=1e-10)

flux_err = 0.1 * flux_true + 0.02
flux_obs = flux_true + rng.normal(0.0, flux_err)

# ---------------------------------------------------------------------------
# Fit with redback (MAP).  Fix the outer density nodes and ejecta structure to
# keep the demo quick; free the parameters that set the spectral shape.
# ---------------------------------------------------------------------------
transient = redback.transient.Supernova(
    name="multifrequency_radio_demo", data_mode="flux_density",
    time=time, flux_density=flux_obs, flux_density_err=flux_err,
    frequency=frequency, optical_data=False,
)

priors = redback.priors.get_priors(model=MODEL)
for key in ("redshift", "log_r_inner", "log_r_outer", "delta_sn", "nn_sn",
            "mej_sn", "esn", "eff", "logepse", "vej_max_ratio"):
    if key in truth:
        priors[key] = truth[key]
priors["vej_max_ratio"] = 3.0
for i in range(8):
    priors[f"log_rho_{i}"] = bilby.core.prior.Uniform(
        truth[f"log_rho_{i}"] - 1.5, truth[f"log_rho_{i}"] + 1.5, f"log_rho_{i}")
priors["cooling"] = True
priors["log_nu_c_scale"] = bilby.core.prior.Uniform(-2.0, 2.0, "log_nu_c_scale")
priors["ssa_alpha_thick"] = bilby.core.prior.Uniform(0.0, 2.5, "ssa_alpha_thick")
priors["ssa_smoothing"] = bilby.core.prior.LogUniform(0.3, 30.0, "ssa_smoothing")

model_kwargs = {"frequency": frequency, "output_format": "flux_density"}
result = redback.fit_model(
    transient=transient, model=MODEL, prior=priors, model_kwargs=model_kwargs,
    fit_method="map", outdir="examples/multifrequency_radio_fit",
    label="demo", plot=False, clean=True,
    optimizer_kwargs={"seed": 1, "maxiter": 300, "popsize": 15, "tol": 1e-6},
)
best = dict(truth)
best.update({key: float(result.posterior[key].iloc[0])
             for key in result.posterior if key in truth and key != "cooling"})
for key in ("p", "logepsb", "log_nu_c_scale", "ssa_alpha_thick", "ssa_smoothing"):
    print(f"{key:>16}: truth {truth[key]:7.3f}   fit {best[key]:7.3f}")

# ---------------------------------------------------------------------------
# Plot: light curves per frequency, and SEDs at each epoch
# ---------------------------------------------------------------------------
fig, (ax_lc, ax_sed) = plt.subplots(1, 2, figsize=(11, 4.5))
grid = np.geomspace(5.0, 3000.0, 300)
freqs = np.unique(frequency)
colors = plt.cm.viridis(np.linspace(0, 0.95, freqs.size))
for nu, c in zip(freqs, colors):
    sel = frequency == nu
    ax_lc.errorbar(time[sel], flux_obs[sel], flux_err[sel], fmt="o", ms=3, color=c)
    ax_lc.plot(grid, static_spline_csm_bpl_radio(time=grid, frequency=nu, **best),
               color=c, lw=1.2, label=f"{nu/1e9:g} GHz")
ax_lc.set(xscale="log", yscale="log", xlabel="Time (observer days)",
          ylabel="Flux density (mJy)", ylim=(1e-2, None))
ax_lc.legend(fontsize=7, ncol=2)

nu_grid = np.geomspace(1e9, 3e11, 200)
for (t, _), c in zip(epochs.items(), plt.cm.plasma(np.linspace(0, 0.85, len(epochs)))):
    sel = time == t
    ax_sed.errorbar(frequency[sel] / 1e9, flux_obs[sel], flux_err[sel], fmt="o", ms=3, color=c)
    ax_sed.plot(nu_grid / 1e9,
                static_spline_csm_bpl_radio(time=np.full_like(nu_grid, t),
                                            frequency=nu_grid, **best),
                color=c, lw=1.2, label=f"{t:g} d")
ax_sed.set(xscale="log", yscale="log", xlabel="Frequency (GHz)",
           ylabel="Flux density (mJy)", ylim=(1e-2, None))
ax_sed.legend(fontsize=7)

fig.tight_layout()
fig.savefig("examples/multifrequency_radio_fit.png", dpi=150)
print("Saved: examples/multifrequency_radio_fit.png")
