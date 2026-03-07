# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 20:42:47 2026

@author: clher
"""

import numpy as np
from scipy.constants import hbar, c, epsilon_0, atomic_mass, k as k_B
import pandas as pd

"""
Calculate parameters for the dipole force on Rb87 atoms
Based on Hernández-Cedillo et al., Phys. Rev. A 87, 023404 (2013)

Force equation from paper (Eq. 1):
f = ħ * Γ * p / (1 + p) * [β - δω*α]

where:
- p is the saturation parameter
- α is the logarithmic gradient of intensity
- β is the gradient of phase
- δω is the detuning
- Γ is the Einstein coefficient (decay rate)
"""

print("="*80)
print("DIPOLE FORCE PARAMETERS FOR Rb87 ATOMS")
print("Hernández-Cedillo et al., Phys. Rev. A 87, 023404 (2013)")
print("="*80)

# ============================================================================
# ATOMIC CONSTANTS FOR Rb87
# ============================================================================

print("\n### ATOMIC CONSTANTS ###\n")

# Rubidium-87 mass
m_Rb87 = 87 * atomic_mass  # kg
print(f"Rb87 atomic mass: m = {m_Rb87:.6e} kg")
print(f"                    = {m_Rb87/atomic_mass:.1f} u")

# D2 line transition (5S_1/2 -> 5P_3/2)
wavelength = 780.368e-9  # m (from paper)
omega_0 = 2 * np.pi * c / wavelength  # rad/s (transition frequency)
f_0 = omega_0 / (2 * np.pi)  # Hz

print(f"\nD2 Line Transition (5S_1/2 -> 5P_3/2):")
print(f"  Wavelength: λ₀ = {wavelength*1e9:.3f} nm")
print(f"  Frequency: f₀ = {f_0:.6e} Hz")
print(f"  Angular frequency: ω₀ = {omega_0:.6e} rad/s")

# Decay rate (Einstein A coefficient)
Gamma = 2 * np.pi * 6.06e6  # rad/s (from paper)
tau = 1 / Gamma  # lifetime
print(f"\nSpontaneous emission:")
print(f"  Decay rate: Γ = 2π × {Gamma/(2*np.pi):.2e} s⁻¹")
print(f"            = {Gamma:.6e} rad/s")
print(f"  Lifetime: τ = {tau*1e9:.2f} ns")

# ============================================================================
# LASER PARAMETERS
# ============================================================================

print("\n### LASER PARAMETERS ###\n")

# Laser wavelength (red detuned from D2 line)
lambda_laser = 780.368e-9 + 0.127e-9  # m (red detuning of 0.127 nm)
omega_laser = 2 * np.pi * c / lambda_laser  # rad/s
f_laser = omega_laser / (2 * np.pi)  # Hz

print(f"Laser wavelength: λ = {lambda_laser*1e9:.3f} nm")
print(f"Laser frequency: f = {f_laser:.6e} Hz")

# Detuning
delta_lambda = lambda_laser - wavelength  # m
delta_omega = omega_laser - omega_0  # rad/s (negative for red detuning)
delta_f = delta_omega / (2 * np.pi)  # Hz

print(f"\nDetuning:")
print(f"  Δλ = {delta_lambda*1e9:.3f} nm (red detuned)")
print(f"  δω = {delta_omega:.6e} rad/s")
print(f"  Δf = {delta_f:.6e} Hz")
print(f"  δω/Γ = {delta_omega/Gamma:.2f}")

# Transverse wave number (from paper)
k_perp = 0.375e6  # m^-1
k = 2 * np.pi / wavelength  # wave number
print(f"\nWave vectors:")
print(f"  k = 2π/λ = {k:.6e} m⁻¹")
print(f"  k_⊥ = {k_perp:.3e} m⁻¹ (from paper)")

# ============================================================================
# DIPOLE MOMENT AND COUPLING
# ============================================================================

print("\n### DIPOLE MOMENT AND COUPLING ###\n")

# For D2 line: 5S_1/2 -> 5P_3/2
# Reduced dipole matrix element for D2 line
# Can be calculated from lifetime or taken from literature

# From decay rate Γ and transition frequency:
# Γ = (ω₀³ |d|²) / (3π ε₀ ħ c³)
# Solving for |d|²:

d_squared = (3 * np.pi * epsilon_0 * hbar * c**3 * Gamma) / (omega_0**3)
d = np.sqrt(d_squared)  # Dipole moment magnitude

print(f"Dipole matrix element:")
print(f"  |μ₁₂|² = {d_squared:.6e} C²·m²")
print(f"  |μ₁₂| = {d:.6e} C·m")
print(f"  |μ₁₂| = {d/(1.602e-19 * 1e-10):.3f} e·Å")

# Electric field for a given intensity
# I = (1/2) * c * ε₀ * |E|²
# For I = I_sat, we get E_sat

# Saturation intensity for Rb87 D2 line
# I_sat = (π h c Γ) / (3 λ³)
I_sat = (np.pi * hbar * c * Gamma) / (3 * wavelength**3)
print(f"\nSaturation intensity:")
print(f"  I_sat = {I_sat:.3f} W/m²")
print(f"        = {I_sat/10:.3f} mW/cm²")

# Saturation electric field
E_sat = np.sqrt(2 * I_sat / (c * epsilon_0))
print(f"  E_sat = {E_sat:.3f} V/m")

# ============================================================================
# RABI FREQUENCY
# ============================================================================

print("\n### RABI FREQUENCY ###\n")

# Maximum Rabi frequency values from paper
g_values = [12.5, 18.5, 19.1, 22.8]  # in units of Γ

print("Maximum Rabi frequencies used in experiments:")
for g in g_values:
    g_rad_s = g * Gamma  # rad/s

    # Rabi frequency: Ω = d·E/ħ
    # So E = ħΩ/d
    E_field = hbar * g_rad_s / d

    # Intensity I = (1/2) * c * ε₀ * |E|²
    I = 0.5 * c * epsilon_0 * E_field**2

    print(f"  g = {g:.1f}Γ:")
    print(f"    Ω = {g_rad_s:.3e} rad/s")
    print(f"    E = {E_field:.3f} V/m")
    print(f"    I = {I:.3f} W/m² = {I/10:.3f} mW/cm²")
    print(f"    I/I_sat = {I/I_sat:.3f}")

# ============================================================================
# COUPLING FACTOR U_c
# ============================================================================

print("\n### COUPLING FACTOR U_c ###\n")

# U_c = μ₁₂ · E (dot product)
# For linearly polarized light and dipole aligned with field:
# U_c = |μ₁₂| * |E|

print("Coupling factor U_c = μ₁₂ · E:")
print(f"  U_c = d * E")
print(f"  For maximum intensity (g = 19.1Γ):")

g_max = 19.1
E_max = hbar * g_max * Gamma / d
U_c_max = d * E_max

print(f"    U_c^max = {U_c_max:.6e} J")
print(f"    U_c^max/ħ = {U_c_max/hbar:.6e} rad/s")
print(f"    U_c^max/(ħΓ) = {U_c_max/(hbar*Gamma):.2f}")

# ============================================================================
# SATURATION PARAMETER p
# ============================================================================

print("\n### SATURATION PARAMETER ###\n")

print("Saturation parameter p = 2|U_c|² / [ħ² ((Γ/2)² + δω²)]")
print()

# For different Rabi frequencies
for g in [12.5, 18.5, 19.1, 22.8]:
    U_c = hbar * g * Gamma  # Maximum coupling

    # Saturation parameter
    numerator = 2 * U_c**2
    denominator = hbar**2 * ((Gamma/2)**2 + delta_omega**2)
    p = numerator / denominator

    print(f"For g = {g:.1f}Γ:")
    print(f"  U_c = {U_c:.6e} J")
    print(f"  p = {p:.6f}")
    print(f"  1/(1+p) = {1/(1+p):.6f}")
    print()

# ============================================================================
# FORCE PREFACTOR
# ============================================================================

print("### FORCE EQUATION COMPONENTS ###\n")

print("Force: f = ħΓp/(1+p) * [β - δω·α]")
print()
print("where:")
print("  α = ∇ln(I) = (1/I)∇I  (logarithmic gradient of intensity)")
print("  β = ∇φ                (gradient of phase)")
print()

# Prefactor for different Rabi frequencies
print("Force prefactor ħΓp/(1+p):")
print()

for g in [12.5, 18.5, 19.1, 22.8]:
    U_c = hbar * g * Gamma
    numerator = 2 * U_c**2
    denominator = hbar**2 * ((Gamma/2)**2 + delta_omega**2)
    p = numerator / denominator

    prefactor = hbar * Gamma * p / (1 + p)

    print(f"For g = {g:.1f}Γ:")
    print(f"  ħΓp/(1+p) = {prefactor:.6e} N·s/m")
    print(f"  In units of ħk: {prefactor/(hbar*k):.6f}")
    print()

# ============================================================================
# GRADIENT FORCE vs RADIATION PRESSURE
# ============================================================================

print("### GRADIENT FORCE vs RADIATION PRESSURE ###\n")

print("From paper: 'the gradient force α term, which tends to guide")
print("the atoms towards the center of the illuminated lobes,")
print("dominates over the radiation pressure term β'")
print()

# Ratio of terms
print("Ratio of gradient to radiation pressure terms:")
print(f"  δω·α / β ≈ |δω|/k ≈ {abs(delta_omega/k):.3e}")
print()
print(f"Since |δω| = {abs(delta_omega):.3e} rad/s")
print(f"and k = {k:.3e} m⁻¹")
print(f"The detuning ratio δω/Γ = {delta_omega/Gamma:.2f}")
print()
print("For red detuning, atoms are attracted to high intensity regions")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE - PARAMETERS FOR FORCE CALCULATION")
print("="*80)

summary_data = []

for g in [12.5, 18.5, 19.1, 22.8]:
    U_c = hbar * g * Gamma
    E_field = hbar * g * Gamma / d
    I = 0.5 * c * epsilon_0 * E_field**2

    numerator = 2 * U_c**2
    denominator = hbar**2 * ((Gamma/2)**2 + delta_omega**2)
    p = numerator / denominator

    prefactor = hbar * Gamma * p / (1 + p)

    row = {
        'g/Γ': g,
        'Ω (MHz)': g * Gamma / (2*np.pi*1e6),
        'E (V/m)': E_field,
        'I (W/m²)': I,
        'I/I_sat': I/I_sat,
        'U_c (J)': U_c,
        'p': p,
        'ħΓp/(1+p) (N·s/m)': prefactor,
        'Prefactor/ħk': prefactor/(hbar*k)
    }
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Save to CSV
df_summary.to_csv('/mnt/user-data/outputs/force_parameters.csv',
                  index=False, float_format='%.6e')

# ============================================================================
# CONSTANTS FOR SIMULATION
# ============================================================================

print("\n" + "="*80)
print("CONSTANTS FOR SIMULATION CODE")
print("="*80)

print(f"""
# Physical constants
m_Rb87 = {m_Rb87:.6e}  # kg
wavelength = {wavelength:.6e}  # m
Gamma = {Gamma:.6e}  # rad/s (decay rate)
delta_omega = {delta_omega:.6e}  # rad/s (detuning)
k = {k:.6e}  # m^-1 (wave vector)
k_perp = {k_perp:.6e}  # m^-1
hbar = {hbar:.6e}  # J·s
d = {d:.6e}  # C·m (dipole moment)
I_sat = {I_sat:.6e}  # W/m²

# For a given Rabi frequency g (in units of Γ):
# g = 19.1  # Example
# U_c_max = hbar * g * Gamma  # J
# p = 2 * U_c_max**2 / (hbar**2 * ((Gamma/2)**2 + delta_omega**2))
# force_prefactor = hbar * Gamma * p / (1 + p)  # N·s/m

# Force calculation:
# f = force_prefactor * (beta - delta_omega * alpha)
# where:
#   alpha = grad(ln(I)) = grad(I)/I  (vector)
#   beta = grad(phase)  (vector)
""")

print("="*80)
print("All parameters calculated and saved to force_parameters.csv")
print("="*80)

# ============================================================================
# ADDITIONAL USEFUL QUANTITIES
# ============================================================================

print("\n### ADDITIONAL USEFUL QUANTITIES ###\n")

# Recoil velocity
v_recoil = hbar * k / m_Rb87
print(f"Recoil velocity: v_r = ħk/m = {v_recoil:.6e} m/s")
print(f"                       = {v_recoil*100:.6f} cm/s")

# Recoil energy
E_recoil = hbar**2 * k**2 / (2 * m_Rb87)
print(f"Recoil energy: E_r = ħ²k²/2m = {E_recoil:.6e} J")
print(f"                    = h × {E_recoil/hbar/(2*np.pi):.3f} Hz")
print(f"                    = k_B × {E_recoil/k_B*1e9:.3f} nK")

# Doppler temperature
T_Doppler = hbar * Gamma / (2 * k_B)
print(f"\nDoppler temperature: T_D = ħΓ/2k_B = {T_Doppler*1e6:.2f} μK")

# Recoil temperature
T_recoil = E_recoil / k_B
print(f"Recoil temperature: T_r = E_r/k_B = {T_recoil*1e9:.2f} nK")

# Characteristic time scales
t_Gamma = 1 / Gamma
t_recoil = m_Rb87 * v_recoil / (hbar * k**2)
print(f"\nTime scales:")
print(f"  Spontaneous emission: 1/Γ = {t_Gamma*1e9:.2f} ns")
print(f"  Recoil time: m·v_r/(ħk²) = {t_recoil*1e6:.2f} μs")

# Characteristic length scales
L_sat = c / Gamma
print(f"\nLength scale: c/Γ = {L_sat:.3f} m")

print("\n" + "="*80)
