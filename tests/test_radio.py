import numpy as np

from redback_csm.models import static_spline_csm_bpl_radio


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
