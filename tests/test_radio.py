import numpy as np

from redback_csm.models import static_spline_csm_bpl_radio
from redback_csm.radio import synchrotron_flux_density


def _shock_inputs(n=200):
    time = np.geomspace(5.0, 3000.0, n)
    vshell = np.full(n, 1.0e9)
    return dict(
        time_days=time,
        vshell_cgs=vshell,
        rho_csm_cgs=1.0e-20 * (time / 100.0) ** -2,
        redshift=0.004,
        logepsb=-1.5,
        logepse=-1.0,
        p=2.6,
        luminosity_distance_cm=3.0e25,
        radius_cgs=vshell * time * 86400.0,
    )


def _local_index(fn, nu, **kwargs):
    lo = fn(frequency=nu / 1.05, **kwargs)
    hi = fn(frequency=nu * 1.05, **kwargs)
    return np.log(hi / lo) / np.log(1.05**2)


def test_explicit_defaults_match_default_spectrum():
    kwargs = _shock_inputs()
    nu = np.geomspace(1.0e8, 3.0e11, kwargs["time_days"].size)
    base = synchrotron_flux_density(frequency=nu, **kwargs)
    explicit = synchrotron_flux_density(
        frequency=nu, cooling=False, ssa_alpha_thick=2.5, ssa_smoothing=None,
        log_tau_ff=None, **kwargs,
    )
    np.testing.assert_array_equal(base, explicit)


def test_cooling_factor_matches_analytic_break():
    kwargs = _shock_inputs()
    nu = np.full(kwargs["time_days"].size, 1.0e11)
    base = synchrotron_flux_density(frequency=nu, **kwargs)
    cooled = synchrotron_flux_density(frequency=nu, cooling=True, log_nu_c_scale=-0.5, **kwargs)

    b_field = np.sqrt(8.0 * np.pi * 10**kwargs["logepsb"] * kwargs["rho_csm_cgs"]
                      * kwargs["vshell_cgs"] ** 2)
    t_src_s = kwargs["time_days"] / 1.004 * 86400.0
    nu_c = (18.0 * np.pi * 9.109e-28 * 2.998e10 * 4.803e-10
            / (6.6524e-25**2 * b_field**3 * t_src_s**2)) * 10**-0.5
    expected = base * (1.0 + nu * 1.004 / nu_c) ** -0.5
    np.testing.assert_allclose(cooled, expected, rtol=1e-10)  # 100 GHz is optically thin here


def test_cooling_steepens_thin_slope_by_half_far_above_break():
    kwargs = _shock_inputs()
    kwargs["time_days"] = kwargs["time_days"][:20]  # early, high-B epochs
    for key in ("vshell_cgs", "rho_csm_cgs", "radius_cgs"):
        kwargs[key] = kwargs[key][:20]
    index = _local_index(
        synchrotron_flux_density, 1.0e13, cooling=True, log_nu_c_scale=-3.0, **kwargs
    )
    np.testing.assert_allclose(index, -kwargs["p"] / 2.0, atol=1e-3)


def test_large_smoothing_recovers_sharp_break():
    kwargs = _shock_inputs()
    nu = np.geomspace(1.0e8, 3.0e11, kwargs["time_days"].size)
    sharp = synchrotron_flux_density(frequency=nu, **kwargs)
    smooth = synchrotron_flux_density(frequency=nu, ssa_smoothing=1.0e4, **kwargs)
    np.testing.assert_allclose(smooth, sharp, rtol=1e-3)


def test_free_free_absorption_factor():
    kwargs = _shock_inputs()
    nu = np.full(kwargs["time_days"].size, 5.0e9)
    base = synchrotron_flux_density(frequency=nu, **kwargs)
    absorbed = synchrotron_flux_density(
        frequency=nu, log_tau_ff=0.0, ff_nu_ref_ghz=5.0 * 1.004,
        ff_t_ref_days=100.0 / 1.004, ff_time_index=1.0, **kwargs,
    )
    t_src = kwargs["time_days"] / 1.004
    expected = base * np.exp(-(t_src / (100.0 / 1.004)) ** -1.0)
    np.testing.assert_allclose(absorbed, expected, rtol=1e-12)


def test_spectral_options_reach_model_wrapper():
    kwargs = dict(
        redshift=0.004, log_r_inner=14.0, log_r_outer=17.7,
        delta_sn=1.0, nn_sn=10.0, mej_sn=3.0, esn=1.0, eff=0.5,
        logepsb=-1.5, logepse=-1.0, p=2.6,
        **{f"log_rho_{i}": -17.0 - 0.6 * i for i in range(8)},
    )
    time = np.array([20.0, 20.0, 500.0, 500.0])
    frequency = np.array([5.0e9, 1.0e11, 5.0e9, 1.0e11])
    base = static_spline_csm_bpl_radio(time=time, frequency=frequency, **kwargs)
    cooled = static_spline_csm_bpl_radio(
        time=time, frequency=frequency, cooling=True, **kwargs
    )
    assert np.all(base > 0)
    assert np.all(cooled <= base)
    assert np.any(cooled < 0.99 * base)


def test_cooling_keeps_sharp_ssa_break_continuous():
    kwargs = _shock_inputs(n=1)
    kwargs["time_days"] = np.full(4000, 20.0)
    for key in ("vshell_cgs", "rho_csm_cgs", "radius_cgs"):
        kwargs[key] = np.full(4000, kwargs[key][0])
    nu = np.geomspace(1.0e8, 1.0e12, 4000)
    flux = synchrotron_flux_density(frequency=nu, cooling=True, log_nu_c_scale=-4.0, **kwargs)
    # adjacent grid points differ by 0.1%; slopes are at most 2.5, so no step > 1%
    assert np.max(np.abs(np.diff(np.log(flux)))) < 0.01


_SPLINE_KWARGS = dict(
    redshift=0.004, log_r_inner=14.0, log_r_outer=17.7,
    delta_sn=1.0, nn_sn=10.0, mej_sn=3.0, esn=1.0, eff=0.5,
    logepsb=-1.5, logepse=-1.0, p=2.6,
    **{f"log_rho_{i}": -15.0 - 0.53 * i for i in range(8)},
)


def test_homologous_csm_density_and_velocity_follow_hydro_expansion():
    from redback_csm.core import (
        DAY, _call_csm, _get_csm_velocity_at_shock, _get_rho_csm_at_shock,
        _rho_generic_spline_at_r,
    )

    kw = {k: v for k, v in _SPLINE_KWARGS.items()
          if k not in ("redshift", "logepsb", "logepse", "p")}
    kw["interval_sn"] = 3652.5
    lc = _call_csm("generic_spline_csm_bpl", **kw)
    r_sh = lc.rshock
    s = 1.0 + lc.time / (kw["interval_sn"] * DAY)

    rho = _get_rho_csm_at_shock("generic_spline_csm_bpl", lc, **kw)
    np.testing.assert_allclose(rho, _rho_generic_spline_at_r(r_sh / s, **kw) / s**3)
    v_csm = _get_csm_velocity_at_shock("generic_spline_csm_bpl", lc, **kw)
    np.testing.assert_allclose(v_csm, r_sh / (lc.time + kw["interval_sn"] * DAY))
    assert np.all(v_csm < lc.vshell)


def test_slowly_expanding_homologous_csm_matches_static_radio():
    from redback_csm.models import generic_spline_csm_bpl_radio

    time = np.array([20.0, 200.0, 1000.0, 2500.0])
    frequency = np.full(time.size, 6.0e9)
    static = static_spline_csm_bpl_radio(time=time, frequency=frequency, **_SPLINE_KWARGS)
    slow = generic_spline_csm_bpl_radio(
        time=time, frequency=frequency, interval_sn=3.65e7, **_SPLINE_KWARGS
    )
    np.testing.assert_allclose(slow, static, rtol=0.05)


def test_exponential_wind_runs():
    from redback_csm.models import exponential_wind_bolometric

    lbol = exponential_wind_bolometric(
        np.geomspace(1.0, 300.0, 20), mexp=3.0, eexp=1.0, mdot=1e-3, vwind=50.0, eff=0.5
    )
    assert np.all(np.isfinite(lbol)) and np.any(lbol > 0)


def test_wind_driven_and_previous_explosion_radio_are_nonzero():
    from redback_csm.models import bpl_bpl_radio, bpl_wind_radio

    time = np.geomspace(5.0, 500.0, 8)
    sync = dict(redshift=0.01, logepsb=-2.0, logepse=-1.0, p=2.8, frequency=5e9)
    wind_driven = bpl_wind_radio(time=time, delta=1.0, nn=10.0, mexp=3.0, eexp=1.0,
                                 mdot=1e-3, vwind=3000.0, eff=0.5, **sync)
    previous = bpl_bpl_radio(time=time, delta=1.0, nn=10.0, mexp=1.0, eexp=0.1,
                             delta_out=1.0, nn_out=10.0, mexp_out=5.0, eexp_out=1.0,
                             interval=365.0, eff=0.5, **sync)
    assert np.all(np.isfinite(wind_driven)) and np.count_nonzero(wind_driven) > 0
    assert np.all(np.isfinite(previous)) and np.count_nonzero(previous) > 0


def test_homologous_swept_mass_matches_trajectory_integral_for_slow_csm():
    from redback_csm.core import (
        _call_csm, _get_rho_csm_at_shock, _homologous_swept_csm_mass, DAY,
    )
    from redback_csm.xray import cumulative_swept_csm_mass

    kw = {k: v for k, v in _SPLINE_KWARGS.items()
          if k not in ("redshift", "logepsb", "logepse", "p")}
    kw["interval_sn"] = 3.65e7
    lc = _call_csm("generic_spline_csm_bpl", **kw)
    exact = _homologous_swept_csm_mass(
        "generic_spline_csm_bpl", lc, kw["interval_sn"] * DAY, **kw
    )
    trajectory = cumulative_swept_csm_mass(
        lc.rshock, _get_rho_csm_at_shock("generic_spline_csm_bpl", lc, **kw)
    )
    late = lc.time > 10 * DAY
    np.testing.assert_allclose(exact[late], trajectory[late], rtol=0.02)
