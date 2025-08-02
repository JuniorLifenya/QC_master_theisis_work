import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Konstanter
c = 3e8  # Lyshastighet i m/s
L0 = 300  # Baseline avstand i meter
h0 = 1e-20  # Amplitude til gravitasjonsbølgen
f_gw = 100  # Frekvens til gravitasjonsbølgen i Hz
omega = 2 * np.pi * f_gw

# Tidsintervall for simulering (sekunder)
t_send = np.linspace(0, 0.1, 5000)

# Gravitasjonsbølge strain h(t)
def h(t):
    return h0 * np.cos(omega * t)

# Effektiv avstand L(t)
def L(t):
    return L0 * (1 + h(t))

# Funksjon for å finne mottakstid t_r gitt sendetid t_s
def find_reception_time(t_s):
    func = lambda t_r: t_r - t_s - L(t_r)/c
    t_r_guess = t_s + L0/c  # Startgjetning
    t_r_solution, = fsolve(func, t_r_guess)
    return t_r_solution

# Beregn mottakstider for alle sendetidspunkter
t_receive = np.array([find_reception_time(ts) for ts in t_send])

# Beregn forsinkelse med og uten gravitasjonsbølge
delay_with_gw = t_receive - t_send
delay_without_gw = L0 / c * np.ones_like(t_send)

# Plot resultatene
plt.figure(figsize=(10, 6))
plt.plot(t_send, delay_with_gw * 1e6, label='Retardasjonstid med GW (μs)')
plt.plot(t_send, delay_without_gw * 1e6, '--', label='Retardasjonstid uten GW (μs)')
plt.xlabel('Sendetid $t_s$ (s)')
plt.ylabel('Retardasjonstid $t_r - t_s$ (mikrosekunder)')
plt.title('Effekt av gravitasjonsbølge på retardasjonstid for elektromagnetisk signal')
plt.legend()
plt.grid(True)
plt.show()
