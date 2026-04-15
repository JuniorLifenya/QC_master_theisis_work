import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import patches as mpatches

def fig_hydrogen_radial_overlap():
    """We want to visualize really the 1s->3d transition
    and why it dominates graviton absorption (Bough & Rothman 2006)
    We show the probability density and quadrupole overlap integrand
    """

    r = np.linspace(0.001, 28, 2000)

    # Exact hydrogenic radial wavefunctions (not normalised for display)
    R_1s = 2* np.exp(-r)
    R_3d = (2/(81*np.sqrt(30))) * r**2 * np.exp(-r/3)

    # Radial probability densities
    P_1s = R_1s**2 * r**2
    P_3d = R_3d**2 * r**2

    # Quadrupole integrand:
    # The r^4 weighting strongly suppresses short-range
    integrand_raw = R_3d * r**2 *R_1s * r**2
    integrand = integrand_raw/np.max(np.abs(integrand_raw))

    # Normalise densities for display
    P_1s /= np.max(P_1s)
    P_3d /= np.max(P_3d)

    # Also compute 2p for comparison( dipole-forbidden by GW)
    R_2p = (1/2*np.sqrt(6)) * r * np.exp(-r/2)
    P_2p = R_2p**2 * r**2 / np.max(R_2p**2 * r**2)

    fig, (ax_main,ax_int) = plt.subplots(2,1, figsize = (9, 7.5), gridspec_kw = {'height_ratios': [1.5,1], 'hspace': 0.35})
    ax = ax_main
    ax.plot(r,P_1s, color = '#3a7bd5', lw = 2.2, label = r'$|R_{1s}|^2 r^2$ (initial state)')
    ax.plot(r, P_3d, color= '#e05a2b', lw = 2.2, label = r'$|R_{3d}|^2 r^2$ (final state, GW-allowed)')
    ax.plot(r, P_2p, color='#888898',  lw = 1.5, ls ='--', alpha=0.6, label=r'$|R_{2p}|^2 r^2$ (EM dipole -- not driven by GW)')