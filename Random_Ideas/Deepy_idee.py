import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# ========================================================================
# FYSISKE KONSTANTER (SI-ENHETER)
# ========================================================================
c = 3e8  # Lyshastighet [m/s]
k = 8.99e9  # Coulombs konstant [N·m²/C²]
q_proton = 1.6e-19  # Protonladning [C]
r0 = 1e-10  # Initial avstand [m] (~atomær skala)

# Gravitasjonsbølgeparametre
h_gw = 1e-20  # Dimensjonsløs amplitude (typisk for LIGO)
f_gw = 1000  # Frekvens [Hz] (i deteksjonsområdet for NV-sentre)
omega_gw = 2 * np.pi * f_gw  # Vinkelfrekvens

# ========================================================================
# GRAVITASJONSBØLGEMETRIKK (TT-gauge)
# ========================================================================
def gw_metric(t, h=h_gw, omega=omega_gw):
    """Returnerer metrikkforstyrrelsen h_μν(t)"""
    return h * np.cos(omega * t)

def distance_perturbation(t, r0=r0):
    """GW-indusert avstandsmodulasjon"""
    return r0 * gw_metric(t)

# ========================================================================
# RETARDASJONSBEREGNING MED GW-EFFEKT
# ========================================================================
def retarded_time(t, r0, h_gw, omega_gw, c, max_iter=100):
    """
    Løser den implisitte retardasjonslikningen under GW-påvirkning:
    t_ret = t - [r0 + δr(t_ret)] / c
    """
    # Hjelpefunksjon for Newton-Raphson
    def equation(t_ret):
        delta_r = distance_perturbation(t_ret, r0)
        return t_ret + (r0 + delta_r)/c - t
    
    # Derivert for numerisk løsning
    def derivative(t_ret):
        d_delta_r = -r0 * h_gw * omega_gw * np.sin(omega_gw * t_ret)
        return 1 + d_delta_r/c
    
    # Startgjett: klassisk retardert tid
    t_ret_guess = t - r0/c
    
    # Løs den implisitte likningen
    try:
        return newton(equation, t_ret_guess, fprime=derivative, maxiter=max_iter)
    except:
        return t_ret_guess  # Fallback ved konvergensproblemer

# ========================================================================
# SIMULERINGSPARAMETRE
# ========================================================================
t_max = 5e-3  # Simuleringstid [s] (5 ms)
num_points = 5000  # Antall tidspunkter
t = np.linspace(0, t_max, num_points)

# Beregn retardert tid og felt med/uten GW
t_ret_gw = np.zeros_like(t)
t_ret_no_gw = np.zeros_like(t)
E_field_gw = np.zeros_like(t)
E_field_no_gw = np.zeros_like(t)

for i, t_i in enumerate(t):
    # Beregn retardert tid
    t_ret_gw[i] = retarded_time(t_i, r0, h_gw, omega_gw, c)
    t_ret_no_gw[i] = t_i - r0/c  # Uten GW-effekt
    
    # Beregn avstand på retardert tidspunkt
    r_ret_gw = r0 + distance_perturbation(t_ret_gw[i])
    r_ret_no_gw = r0  # Konstant uten GW
    
    # Coulombs lov med retardert posisjon
    E_field_gw[i] = k * q_proton / (r_ret_gw**2)
    E_field_no_gw[i] = k * q_proton / (r_ret_no_gw**2)

# Beregn GW-indusert feltforskjell
delta_E = E_field_gw - E_field_no_gw

# ========================================================================
# VISUALISERING
# ========================================================================
plt.figure(figsize=(14, 10))

# Gravitasjonsbølgemodulasjon
plt.subplot(3, 1, 1)
plt.plot(t, gw_metric(t), 'purple')
plt.title('Gravitasjonsbølgemodulasjon')
plt.ylabel('Metrikkforstyrrelse $h_{\\mu\\nu}$')
plt.grid(alpha=0.3)

# Retardasjonsforskyvning
plt.subplot(3, 1, 2)
plt.plot(t, (t_ret_gw - t_ret_no_gw)*1e12, 'teal')
plt.title('GW-indusert retardasjonsforskyvning')
plt.ylabel('$\\Delta t_{ret}$ [ps]')
plt.grid(alpha=0.3)

# Elektrisk felt med/uten GW
plt.subplot(3, 1, 3)
plt.plot(t, E_field_no_gw, 'gray', label='Uten GW')
plt.plot(t, E_field_gw, 'red', label='Med GW')
plt.plot(t, delta_E, 'blue', label='Feltdifferanse $\Delta E$')
plt.title('Retardert elektrisk felt')
plt.xlabel('Tid [s]')
plt.ylabel('Feltstyrke [V/m]')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================================================
# ANALYSE AV NV-CENTER SIGNAL
# ========================================================================
# Energiforskyvning i NV-senter (~GHz spinnresonans)
gamma_e = 28e9  # Gyromagnetisk forhold [Hz/T]
B_field = 0.01  # [T] (eksternt felt)
delta_E_spin = gamma_e * B_field * (delta_E / max(E_field_no_gw))

plt.figure(figsize=(12, 6))
plt.plot(t, delta_E_spin, 'darkorange')
plt.title('GW-indusert spinnenergiforskyvning i NV-senter')
plt.xlabel('Tid [s]')
plt.ylabel('$\\Delta E_{spin}$ [Hz]')
plt.grid(alpha=0.3)
plt.show()