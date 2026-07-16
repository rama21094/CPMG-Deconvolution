import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# 1. Independent Variables & Constants
gamma_H = 2.6752e8       # [cite: 137]
gamma_N = -2.7120e7      # [cite: 138]
B0 = 14.1                # Tesla (approx 600 MHz)
J_NH = 92.0              # Hz [cite: 120]
k_ex = 500.0             # s^-1 
p_B = 0.05               # Population of minor state 
dw = 200.0 * 2 * np.pi   # Rad/s, chemical shift difference between A and B
R2_intrinsic = 15.0      # s^-1, baseline relaxation

# Derived populations and rates
p_A = 1.0 - p_B
k_AB = k_ex * p_B
k_BA = k_ex * p_A

import scipy.constants as const

# --- 1. Physical Constants and Spin System Parameters ---
mu_0 = const.mu_0                     # Permeability of free space [cite: 24]
hbar = const.hbar                     # Reduced Planck's constant
gamma_I = 267.522e6                   # 1H gyromagnetic ratio (rad s^-1 T^-1)
gamma_S = -27.116e6                   # 15N gyromagnetic ratio (rad s^-1 T^-1)
B0 = 14.09                            # Static magnetic field (T) (~600 MHz 1H)

r_IS = 1.02e-10                       # N-H bond length (m) [cite: 24]
csa_S = -160e-6                       # 15N CSA (ppm) [cite: 52]
theta_CSA_DD = 22.0 * (np.pi / 180.0) # Angle between CSA unique axis and N-H bond [cite: 32, 51]

# Dynamic Parameters (Lipari-Szabo) [cite: 29, 50]
tau_m = 5.0e-9                        # Overall correlation time (s)
tau_e = 50.0e-12                      # Internal correlation time (s)
S2 = 0.8                              # Generalized order parameter [cite: 33, 50]

# Frequencies (rad / s)
omega_I = -gamma_I * B0
omega_S = -gamma_S * B0

# --- 2. Spectral Density and Coupling Constants ---
def J_comp(omega):
    """
    Calculates the analytical spectral density using the Lipari-Szabo approach[cite: 29, 30].
    """
    tau_inv = (1.0 / tau_m) + (1.0 / tau_e)
    tau = 1.0 / tau_inv
    
    term1 = (S2 * tau_m) / (1.0 + (omega * tau_m)**2)
    term2 = ((1.0 - S2) * tau) / (1.0 + (omega * tau)**2)
    
    # Factor of 2/5 is often absorbed into the interaction constants in standard Redfield formulations,
    # but we apply it here for exactness relative to the standard J(w) definition.
    return (2.0 / 5.0) * (term1 + term2)

# Interaction Constants [cite: 23]
# Dipolar coupling constant squared (rad^2 s^-2)
d2 = (mu_0 / (4.0 * np.pi))**2 * (hbar * gamma_I * gamma_S / r_IS**3)**2 # [cite: 24]

# CSA coupling constant squared for S spin (rad^2 s^-2)
c2 = (1.0 / 3.0) * (omega_S * csa_S)**2 # [cite: 25]

# DD-CSA cross-correlation prefactor [cite: 31]
P2_cos = 0.5 * (3.0 * np.cos(theta_CSA_DD)**2 - 1.0)
dc = np.sqrt(d2) * np.sqrt(c2) * P2_cos

# --- 3. Rigorous Matrix Builder ---
def build_16x16_rigorous(omega_offset_I, omega_offset_S, J_IS):
    """
    Constructs the exact 16x16 Homogeneous Master Equation matrix[cite: 8, 35].
    Basis Order:
    0: E/2,  1: Ix,   2: Iy,   3: Iz
    4: Sx,   5: Sy,   6: Sz,   7: 2IxSz, 8: 2IySz
    9: 2IzSx, 10: 2IzSy, 11: 2IxSx, 12: 2IxSy, 13: 2IySx, 14: 2IySy, 15: 2IzSz
    """
    L = np.zeros((16, 16), dtype=complex)
    
    # Pre-calculate spectral densities at required transitions
    j0 = J_comp(0)
    jI = J_comp(omega_I)
    jS = J_comp(omega_S)
    jI_minus_S = J_comp(omega_I - omega_S)
    jI_plus_S = J_comp(omega_I + omega_S)
    
    # --- Redfield Relaxation Rates ---
    # Autorelaxation of S (e.g., 15N)
    R1_S = (d2 / 4.0) * (jI_minus_S + 3*jS + 6*jI_plus_S) + c2 * jS
    R2_S = (d2 / 8.0) * (4*j0 + jI_minus_S + 3*jS + 6*jI + 6*jI_plus_S) + (c2 / 6.0) * (4*j0 + 3*jS)
    
    # Autorelaxation of I (e.g., 1H)
    R1_I = (d2 / 4.0) * (jI_minus_S + 3*jI + 6*jI_plus_S)
    R2_I = (d2 / 8.0) * (4*j0 + jI_minus_S + 3*jI + 6*jS + 6*jI_plus_S)
    
    # Anti-phase relaxation (e.g., 2IzSx)
    R2_IzSx = (d2 / 8.0) * (4*j0 + jI_minus_S + 3*jS + 6*jI + 6*jI_plus_S) + \
              (d2 / 4.0) * (jI_minus_S + 3*jI + 6*jI_plus_S) + \
              (c2 / 6.0) * (4*j0 + 3*jS)
              
    # Multiple Quantum relaxation (e.g., 2IxSx)
    R_MQ = (d2 / 8.0) * (jI_minus_S + 3*jI + 3*jS + 6*jI_plus_S) + (c2 / 6.0) * (4*j0 + 3*jS)

    # Cross-correlated relaxation rates (DD-CSA interference) 
    eta_z = dc * jS
    eta_xy = (dc / 6.0) * (4*j0 + 3*jS)

    # --- Apply Relaxation to Diagonal ---
    L[1,1] = L[2,2] = -R2_I
    L[3,3] = -R1_I
    L[4,4] = L[5,5] = -R2_S
    L[6,6] = -R1_S
    L[7,7] = L[8,8] = -R2_IzSx # Symmetry assumes similar rates for IxSz and IySz
    L[9,9] = L[10,10] = -R2_IzSx
    L[11,11] = L[12,12] = L[13,13] = L[14,14] = -R_MQ
    L[15,15] = -(R1_I + R1_S)

    # --- Apply DD-CSA Cross-Correlation (Off-diagonal) ---
    # S_x <-> 2IzSx and S_y <-> 2IzSy
    L[4,9] = L[9,4] = -eta_xy
    L[5,10] = L[10,5] = -eta_xy
    
    # S_z <-> 2IzSz
    L[6,15] = L[15,6] = -eta_z

    # --- Coherent Evolution (Hamiltonian Commutators) ---
    # Chemical Shift (I spin)
    L[1, 2] = -omega_offset_I; L[2, 1] = omega_offset_I
    L[7, 8] = -omega_offset_I; L[8, 7] = omega_offset_I
    L[11, 13] = -omega_offset_I; L[13, 11] = omega_offset_I
    L[12, 14] = -omega_offset_I; L[14, 12] = omega_offset_I

    # Chemical Shift (S spin)
    L[4, 5] = -omega_offset_S; L[5, 4] = omega_offset_S
    L[9, 10] = -omega_offset_S; L[10, 9] = omega_offset_S
    L[11, 12] = -omega_offset_S; L[12, 11] = omega_offset_S
    L[13, 14] = -omega_offset_S; L[14, 13] = omega_offset_S

    # Scalar Coupling J_IS (e.g., Sx <-> 2SyIz)
    pi_J = np.pi * J_IS
    # I spin coupling
    L[1, 8] = -pi_J; L[8, 1] = pi_J
    L[2, 7] = pi_J; L[7, 2] = -pi_J
    # S spin coupling
    L[4, 10] = -pi_J; L[10, 4] = pi_J
    L[5, 9] = pi_J; L[9, 5] = -pi_J
    
    # Multiple Quantum coupling
    L[11, 14] = -pi_J; L[14, 11] = pi_J # 2IxSx <-> 2IySy
    L[12, 13] = pi_J; L[13, 12] = -pi_J # 2IxSy <-> 2IySx

    return L

def build_16x16_block(omega_offset, J, R2):
    """
    Constructs the 16x16 Homogeneous Master Equation matrix for a single state.
    Includes Coherent (Hamiltonian) and Relaxation (Liouvillian) terms.
    """
    L = np.zeros((16, 16), dtype=complex)
    
    # Basis indices (simplified representation):
    # 0: E/2, 1: Ix, 2: Iy, 3: Iz, 4: Sx, 5: Sy, 6: Sz, 7: 2IxSz, 8: 2IySz ...
    
    # Intrinsic Relaxation (Diagonal)
    np.fill_diagonal(L, -R2)
    L[0, 0] = 0.0 # E/2 does not relax
    
    # Coherent Evolution (Hamiltonian Commutators) [cite: 48, 87]
    # Chemical Shift (Sx -> Sy evolution)
    L[4, 5] = -omega_offset
    L[5, 4] = omega_offset
    
    # Scalar Coupling J (Sx <-> 2SyIz) [cite: 31, 87, 124]
    pi_J = np.pi * J
    L[4, 8] = -pi_J   # Sx -> 2IySz (antiphase)
    L[8, 4] = pi_J
    L[5, 7] = pi_J    # Sy -> 2IxSz (antiphase)
    L[7, 5] = -pi_J
    
    return L

def build_32x32_liouvillian(J_val):
    """
    Builds the full exchanging two-spin system using Bloch-McConnell. 
    """
    L_full = np.zeros((32, 32), dtype=complex)
    
    # Build sub-blocks
    L_A = build_16x16_rigorous(0.0, 0.0, J_val) # build_16x16_block(0.0, J_val, R2_intrinsic) 
    L_B = build_16x16_rigorous(0.0, dw, J_val) # build_16x16_block(dw, J_val, R2_intrinsic) 
    
    # Insert sub-blocks into the 32x32 matrix
    L_full[0:16, 0:16] = L_A
    L_full[16:32, 16:32] = L_B
    
    # Add Exchange Kinetics (Bloch-McConnell) 
    for i in range(16):
        if i == 0: continue # Unity operator unaffected by exchange
        
        # -k_AB * A + k_BA * B
        L_full[i, i] -= k_AB
        L_full[i, i+16] += k_BA
        
        # k_AB * A - k_BA * B
        L_full[i+16, i] += k_AB
        L_full[i+16, i+16] -= k_BA
        
    return L_full

def simulate_cpmg_profile(J_val, ncyc_range, T_relax=0.04):
    """
    Simulates the CPMG sequence: (tau - 180 - tau)_ncyc
    and calculates R2_eff.
    """
    L = build_32x32_liouvillian(J_val)
    
    # Define the 180 degree pulse on S spin (assumed ideal for simplicity)
    # Inverts Sy and Sz, leaves Sx unchanged.
    P_180 = np.eye(32)
    for state_offset in [0, 16]:
        P_180[state_offset+5, state_offset+5] = -1 # Sy -> -Sy
        P_180[state_offset+6, state_offset+6] = -1 # Sz -> -Sz
        P_180[state_offset+7, state_offset+7] = -1 # 2IxSz -> -2IxSz
        P_180[state_offset+8, state_offset+8] = -1 # 2IySz -> -2IySz
    
    R2_eff_list = []
    nu_cp_list = []
    
    # Initial state: In-phase transverse magnetization on State A (Sx) 
    rho_0 = np.zeros(32)
    rho_0[4] = p_A 
    rho_0[20] = p_B 
    
    for ncyc in ncyc_range:
        tau_cp = T_relax / (4.0 * ncyc)
        nu_cp = 1.0 / (4.0 * tau_cp)
        nu_cp_list.append(nu_cp)
        
        # Propagator for free precession during tau_cp 
        U_tau = la.expm(L * tau_cp)
        
        # CPMG building block: tau - 180 - tau - tau - 180 - tau
        U_block = U_tau @ P_180 @ U_tau @ U_tau @ P_180 @ U_tau
        
        # Apply N cycles
        U_total = np.linalg.matrix_power(U_block, int(ncyc))
        rho_t = U_total @ rho_0
        
        # Extract observable magnetization (Sx_A + Sx_B) 
        mag_x = np.real(rho_t[4] + rho_t[20])
        
        # Calculate R2_eff
        R2_eff = -np.log(mag_x / (p_A + p_B)) / T_relax
        R2_eff_list.append(R2_eff)
        
    return nu_cp_list, R2_eff_list

# 4. Execute and Visualize
ncyc_range = np.arange(1, 40, 2)

# Simulate WITH Scalar Coupling (J = 92 Hz)
nu_cp, r2eff_with_J = simulate_cpmg_profile(J_NH, ncyc_range)

# Simulate WITHOUT Scalar Coupling (J = 0 Hz)
_, r2eff_no_J = simulate_cpmg_profile(0.0, ncyc_range)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(nu_cp, r2eff_with_J, '-o', label='With J-coupling (92 Hz)')
plt.plot(nu_cp, r2eff_no_J, '--s', label='No J-coupling (0 Hz)')
plt.xlabel(r'$\nu_{cp}$ (Hz)')
plt.ylabel(r'$R_{2,eff}$ ($s^{-1}$)')
plt.title('Simulated CPMG Dispersion Profile')
plt.legend()
plt.grid(True)
plt.show()