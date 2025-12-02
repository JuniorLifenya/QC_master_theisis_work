import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

def run_cpmg_simulation():
    # --- 1. System Parameters ---
    # Frequencies in MHz (arbitrary units for demo, scale to physical values later)
    omega_gw = 2 * np.pi * 50e-3  # GW frequency (e.g., 50 kHz)
    h_strain = 0.5                # Effective coupling strength (amplitude)
    
    # Pulse Parameters
    n_pulses = 8                  # Number of pi-pulses (CPMG-8)
    # The total time must match the GW periods for maximum sensitivity.
    # For CPMG, pulses happen at: tau/2, 3tau/2, 5tau/2...
    # We tune the inter-pulse delay (tau) to be half the GW period.
    tau = (1 / (omega_gw / (2 * np.pi))) / 2  # Resonance condition
    total_time = n_pulses * tau
    
    # --- 2. Operators (Spin-1 Subspace |0> and |-1>) ---
    # We treat the NV center as an effective qubit for the sensing transition
    # Basis: |0> = [1,0], |-1> = [0,1]
    Sz = qt.sigmaz()
    Sx = qt.sigmax()
    Sy = qt.sigmay()
    
    # Initial State: Superposition (|0> + |-1>)/sqrt(2) created by first pi/2 pulse
    psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

    # --- 3. Hamiltonians ---
    # H_gw: The Gravitational Wave Interaction
    # In the rotating frame, this looks like a time-dependent detuning (Sz)
    H_gw = h_strain * Sz
    
    # H_control: The Control Pulses (Driving Sx)
    # We use a very strong amplitude for short pulses (Hard Pulse Limit)
    Omega_rabi = 500.0  # Rabi frequency (MHz)
    H_control = Omega_rabi * Sx

    # --- 4. Time-Dependent Functions ---
    
    # GW signal: h(t) = A * cos(omega * t + phase)
    def gw_signal(t, args):
        return np.cos(args['omega_gw'] * t)

    # Pulse Sequence Function
    # Returns 1.0 when pulse is ON, 0.0 when OFF
    def pulse_sequence(t, args):
        tau = args['tau']
        pulse_width = args['width']
        # CPMG timing: pulses at t = tau/2, 3tau/2, 5tau/2...
        # Check if t is close to any pulse center
        for k in range(args['n_pulses']):
            center_time = (k + 0.5) * tau
            if abs(t - center_time) < (pulse_width / 2):
                return 1.0
        return 0.0

    args = {
        'omega_gw': omega_gw, 
        'tau': tau, 
        'n_pulses': n_pulses,
        'width': np.pi / (2 * Omega_rabi) # Width for a pi-pulse (pi / Omega)
    }

    # --- 5. Run Simulation ---
    # H = [H_gw * gw(t)] + [H_control * pulse(t)]
    H_td = [[H_gw, gw_signal], [H_control, pulse_sequence]]
    
    tlist = np.linspace(0, total_time, 1000)
    
    # We observe the expectation value of Sx and Sy (coherence)
    # The signal will appear as a rotation in the XY plane.
    result = qt.mesolve(H_td, psi0, tlist, [], [Sx, Sy, Sz], args=args)

    # --- 6. Plotting ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot the Pulse Sequence (to verify timing)
    pulses = [pulse_sequence(t, args) for t in tlist]
    ax[0].plot(tlist, pulses, color='orange', label='Control Pulses (CPMG)')
    ax[0].set_ylabel('Rabi Drive')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot the Spin Dynamics
    # If the GW is detected, the spin vector will rotate away from X axis
    ax[1].plot(tlist, result.expect[0], label='<Sx> (Coherence)', linewidth=2)
    ax[1].plot(tlist, result.expect[1], label='<Sy> (Signal)', linestyle='--')
    ax[1].set_ylabel('Spin Expectation')
    ax[1].set_xlabel('Time (us)')
    ax[1].legend()
    ax[1].set_title(f'GW Sensing with CPMG-{n_pulses}')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("plots/cpmg_sensing.png")
    print("Simulation complete. Check 'plots/cpmg_sensing.png'.")

if __name__ == "__main__":
    run_cpmg_simulation()