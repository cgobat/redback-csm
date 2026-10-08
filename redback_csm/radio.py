"""
radio.py — Self-contained synchrotron radio emission physics for CSM interaction models.

All calculations are in CGS units. Output is in mJy.

The key scaling relation (Chevalier 1998) correctly gives:
    L_nu  ∝  epsilon_B^{(p+1)/4}
which is positive for all p > -1.  This corrects the common error of confusing the
spectral-index exponent (1-p)/2 with the epsilon_B dependence.
"""

import numpy as _np

# Physical constants (CGS)
_QE     = 4.803e-10     # esu (elementary charge)
_ME     = 9.109e-28     # g  (electron mass)
_C      = 2.998e10      # cm/s
_SIGMAT = 6.6524e-25    # cm^2 (Thomson cross-section)
_MP     = 1.6726e-24    # g  (proton mass)
_DAY    = 86400.0       # s


# Optional spectral-shape keywords accepted by ``synchrotron_flux_density``;
# the model wrappers pop these before the hydro call.
RADIO_SPECTRAL_OPTIONS = (
    "cooling",
    "log_nu_c_scale",
    "ssa_alpha_thick",
    "ssa_smoothing",
    "log_tau_ff",
    "ff_time_index",
    "ff_nu_ref_ghz",
    "ff_t_ref_days",
)


def synchrotron_flux_density(
    time_days,
    vshell_cgs,
    rho_csm_cgs,
    redshift,
    logepsb,
    logepse,
    p,
    frequency,
    luminosity_distance_cm,
    radius_cgs=None,
    cooling=False,
    log_nu_c_scale=0.0,
    ssa_alpha_thick=2.5,
    ssa_smoothing=None,
    log_tau_ff=None,
    ff_time_index=3.0,
    ff_nu_ref_ghz=1.0,
    ff_t_ref_days=100.0,
):
    """
    Compute synchrotron radio flux density from a CSM-interaction shock.

    Uses the Chevalier (1998) formalism with self-absorption.  All input arrays
    should be on the same time grid (the Fortran output grid).

    The optional spectral extensions below are all off by default, in which
    case the output is identical to the original sharp-break SSA spectrum.

    Parameters
    ----------
    time_days : array_like
        Observer-frame time in days.
    vshell_cgs : array_like
        Shock velocity relative to the upstream CSM in cm/s.
    rho_csm_cgs : array_like
        Upstream CSM mass density at the shock in g/cm^3.
    radius_cgs : array_like, optional
        Shock radius in cm. If omitted, the function falls back to
        ``vshell_cgs * time`` for backwards compatibility.
    redshift : float
        Source redshift z.
    logepsb : float
        log10(epsilon_B); magnetic energy fraction of shock ram pressure.
    logepse : float
        log10(epsilon_e); electron energy fraction.
    p : float
        Electron power-law index (typically 2–4).
    frequency : float or array_like
        Observing frequency in Hz (observer frame).  Can be a scalar or array of
        the same length as time_days.
    luminosity_distance_cm : float
        Luminosity distance in cm.
    cooling : bool, optional
        If True, steepen the optically thin spectrum by 1/2 above the
        synchrotron cooling frequency
        ``nu_c = 18 pi m_e c q_e / (sigma_T^2 B^3 t^2)``, using a smooth
        ``(1 + nu/nu_c)^{-1/2}`` factor.
    log_nu_c_scale : float, optional
        log10 multiplier on ``nu_c`` (default 0).  A negative value mimics
        additional (e.g. inverse-Compton) cooling.
    ssa_alpha_thick : float, optional
        Spectral index of the optically thick asymptote (default 5/2).
        Values below 5/2 mimic an inhomogeneous / clumpy absorber.
    ssa_smoothing : float or None, optional
        If None (default) the thick and thin asymptotes are joined with a
        sharp break at ``nu_SSA``.  Otherwise they are joined smoothly,
        ``F = (F_thick^{-s} + F_thin^{-s})^{-1/s}``; smaller ``s`` gives a
        broader peak and ``s -> inf`` recovers the sharp break.
    log_tau_ff : float or None, optional
        If given, multiply by ``exp(-tau_ff)`` for an external free-free
        absorber, ``tau_ff = 10**log_tau_ff (nu/ff_nu_ref_ghz)^{-2.1}
        (t/ff_t_ref_days)^{-ff_time_index}`` (rest-frame nu and t).
    ff_time_index : float, optional
        Time decay index of ``tau_ff`` (default 3, a steady wind swept by a
        constant-velocity shock).
    ff_nu_ref_ghz, ff_t_ref_days : float, optional
        Reference frequency (GHz) and time (days) for ``log_tau_ff``.

    Returns
    -------
    flux_mJy : ndarray
        Flux density in mJy, same shape as time_days.
    """
    time_days = _np.asarray(time_days, dtype=float)
    vshell    = _np.asarray(vshell_cgs, dtype=float)
    rho       = _np.asarray(rho_csm_cgs, dtype=float)

    eps_b = 10.0 ** logepsb
    eps_e = 10.0 ** logepse
    dl    = luminosity_distance_cm
    z     = redshift

    # K-correction: source-frame quantities
    t_src   = time_days / (1.0 + z)          # days, rest frame
    t_src_s = t_src * _DAY                   # seconds

    # Broadcast frequency to the time grid
    freq_obs = _np.asarray(frequency, dtype=float)
    if freq_obs.ndim == 0:
        freq_obs = _np.full_like(time_days, float(frequency))
    freq_src = freq_obs * (1.0 + z)          # Hz, rest frame

    if radius_cgs is None:
        r_s = vshell * t_src_s
    else:
        r_s = _np.asarray(radius_cgs, dtype=float)

    # Magnetic field energy density: u_B = eps_B * rho * v^2
    u_b = eps_b * rho * vshell ** 2          # erg/cm^3

    # Magnetic field strength
    B = _np.sqrt(8.0 * _np.pi * u_b)        # Gauss

    # Larmor (cyclotron) frequency
    nu_L = _QE * B / (2.0 * _np.pi * _ME * _C)  # Hz

    # Number density of radiating electrons: n_e = eps_e * (rho / m_p)
    n_e = eps_e * (rho / _MP)               # cm^-3

    # Total number of electrons in a sphere of radius r_s
    N_0 = (4.0 / 3.0) * _np.pi * r_s ** 3 * n_e  # dimensionless count

    # Synchrotron power normalisaton (erg/s/Hz at nu = nu_L)
    #   C_0 = (4/3) N_0 sigma_T c u_B
    C_0 = (4.0 / 3.0) * N_0 * _SIGMAT * _C * u_b  # erg/s

    # Optically-thin spectral luminosity (erg/s/Hz)
    #   L_nu = C_0 / (2 nu_L) * (nu / nu_L)^{(1-p)/2}
    #
    # epsilon_B dependence check:
    #   C_0   ∝  u_B  ∝  eps_B
    #   nu_L  ∝  B    ∝  eps_B^{1/2}
    #   (nu/nu_L)^{(1-p)/2}  ∝  nu_L^{(p-1)/2}  ∝  eps_B^{(p-1)/4}
    #   Overall: L_nu  ∝  eps_B * eps_B^{-1/2} * eps_B^{(p-1)/4}
    #           = eps_B^{1 - 1/2 + (p-1)/4}  =  eps_B^{(p+1)/4}  ✓
    L_nu = C_0 / (2.0 * nu_L) * (freq_src / nu_L) ** ((1.0 - p) / 2.0)  # erg/s/Hz

    # Flux density at observer (Jy)
    flux_Jy = L_nu / (4.0 * _np.pi * dl ** 2) / 1.0e-23  # Jy

    # Self-absorption frequency and suppression
    # From Chevalier (1998), the SSA flux peak at nu_SSA is F_SSA, and
    # below nu_SSA the spectrum rises as nu^{5/2}.
    #
    # Spectral peak flux density in Jy (at nu_L)
    F_peak_Jy = C_0 / (2.0 * nu_L) / (4.0 * _np.pi * dl ** 2) / 1.0e-23

    # SSA optical depth ~ 1 condition gives nu_SSA.
    # Following Chevalier (1998) Eq. 9 / Björnsson & Fransson (2004):
    #   nu_SSA = [ (3^1.5 * q_e^0.5 * B^0.5 * F_peak_Jy * 1e-23 * nu_L^{beta-1} * dl^2)
    #              / (4 pi^1.5 * r_s^2 * c^0.5 * m_e^1.5) ]^{2/(2 beta + 3)}
    # where beta = (p+4)/2 — 1 = (p+2)/2 ... actually we use beta = (p-1)/2 + 1 = (p+1)/2
    # More precisely, using the convention in the prototype code: beta = 1 - (1-p)/2 = (p+1)/2
    beta = (p + 1.0) / 2.0

    # F_peak in CGS flux units for SSA formula (erg/s/Hz/cm^2)
    F_peak_cgs = F_peak_Jy * 1.0e-23

    numerator   = dl ** 2 * 3.0 ** 1.5 * _QE ** 0.5 * B ** 0.5 * F_peak_cgs * nu_L ** (beta - 1.0)
    denominator = 4.0 * _np.pi ** 1.5 * r_s ** 2 * _C ** 0.5 * _ME ** 1.5
    nu_ssa = (numerator / denominator) ** (2.0 / (2.0 * beta + 3.0))  # Hz (rest frame)

    # SSA flux at nu_SSA
    F_ssa_Jy = F_peak_Jy * (nu_ssa / nu_L) ** (1.0 - beta)

    # Optically thin flux, optionally steepened above the cooling break
    flux_mJy = flux_Jy * 1.0e3  # convert to mJy
    if cooling:
        nu_c = (18.0 * _np.pi * _ME * _C * _QE / (_SIGMAT ** 2 * B ** 3 * t_src_s ** 2)
                * 10.0 ** log_nu_c_scale)
        flux_mJy = flux_mJy * (1.0 + freq_src / nu_c) ** -0.5

    if ssa_smoothing is None:
        if cooling:
            # Anchor the sharp optically thick branch to the cooled thin
            # spectrum at nu_SSA so the two branches stay continuous.
            F_ssa_Jy = F_ssa_Jy * (1.0 + nu_ssa / nu_c) ** -0.5
        # Apply SSA suppression where freq_src < nu_ssa
        ssa_mask = freq_src < nu_ssa
        if _np.any(ssa_mask):
            flux_mJy[ssa_mask] = (
                F_ssa_Jy[ssa_mask] * (freq_src[ssa_mask] / nu_ssa[ssa_mask]) ** ssa_alpha_thick
                * 1.0e3   # Jy → mJy
            )
    else:
        # Smoothly broken power law between the thick and thin asymptotes,
        # evaluated in log space to avoid overflow.
        with _np.errstate(divide="ignore", invalid="ignore"):
            log_thick = _np.log(F_ssa_Jy * 1.0e3) + ssa_alpha_thick * _np.log(freq_src / nu_ssa)
            log_thin = _np.log(flux_mJy)
            s = float(ssa_smoothing)
            flux_mJy = _np.exp(-_np.logaddexp(-s * log_thick, -s * log_thin) / s)

    if log_tau_ff is not None:
        tau_ff = (10.0 ** log_tau_ff
                  * (freq_src / (ff_nu_ref_ghz * 1.0e9)) ** -2.1
                  * (t_src / ff_t_ref_days) ** -ff_time_index)
        flux_mJy = flux_mJy * _np.exp(-_np.minimum(tau_ff, 700.0))

    # Guard against non-physical negative values (can occur at very early times when
    # r_s ~ 0 and numerical noise dominates)
    flux_mJy = _np.where(_np.isfinite(flux_mJy) & (flux_mJy > 0.0), flux_mJy, 0.0)

    return flux_mJy
