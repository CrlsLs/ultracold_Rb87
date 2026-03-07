# %% -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:04:33 2026

@author: clher
"""

"""
Rectangular (Cartesian) to Parabolic Coordinate Transformation
===============================================================

Parabolic coordinates (σ, τ) are defined by:
    x = σ * τ
    y = (τ² - σ²) / 2

Inverse (Cartesian → Parabolic):
    σ = sqrt(sqrt(x² + y²) - y)
    τ = sqrt(sqrt(x² + y²) + y)

Note: σ ≥ 0, τ ≥ 0, and x can be negative or positive.
For x < 0, σ is negated (or the sign is tracked separately).
"""


# Convert mpmath function to a vectorized numpy function
# This allows us to pass arrays to hyp1f1
# import matplotlib
# from typing import Union, Tuple

from scipy.optimize import curve_fit, minimize
import math
import numpy as np
import mpmath
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
hyp1f1_vec = np.frompyfunc(
    lambda a, b, z: complex(mpmath.hyp1f1(a, b, z)), 3, 1)


# ── Core transformation ──────────────────────────────────────────────────────

def rect_to_parabolic(
    x: float, y: float
) -> Tuple[float, float]:
    """
    Convert rectangular coordinates (x, y) to parabolic coordinates (sigma, tau).

    Parameters
    ----------
    x : float  — Cartesian x-coordinate
    y : float  — Cartesian y-coordinate

    Returns
    -------
    (sigma, tau) : tuple of floats
        sigma ≥ 0  (with sign of x carried separately — see note)
        tau   ≥ 0

    Notes
    -----
    The standard definition uses σ ≥ 0 and τ ≥ 0.
    When x < 0, the convention sign(x) * σ is used so that the
    full plane is covered bijectively.
    """
    r = math.hypot(x, y)          # sqrt(x² + y²)
    tau = math.sqrt(r + y)      # τ = √(r + y)
    sigma = math.sqrt(r - y)      # σ = √(r - y)  (always ≥ 0)

    # Restore sign of x  →  x = σ·τ  requires sign(sigma) = sign(x)
    if x < 0:
        sigma = -sigma

    return sigma, tau


def parabolic_to_rect(
    sigma: float, tau: float
) -> Tuple[float, float]:
    """
    Convert parabolic coordinates (sigma, tau) back to rectangular (x, y).

    Parameters
    ----------
    sigma : float  — parabolic coordinate σ  (can be negative)
    tau   : float  — parabolic coordinate τ  (τ ≥ 0)

    Returns
    -------
    (x, y) : tuple of floats
    """
    x = sigma * tau
    y = (tau**2 - sigma**2) / 2
    return x, y


# ── Vectorised (NumPy) versions ──────────────────────────────────────────────

def rect_to_parabolic_array(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised version for NumPy arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    tau = np.sqrt(np.maximum(r + y, 0.0))
    sigma = np.sqrt(np.maximum(r - y, 0.0))
    sigma = np.where(x < 0, -sigma, sigma)
    return sigma, tau


def parabolic_to_rect_array(
    sigma: np.ndarray, tau: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised version for NumPy arrays."""
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau,   dtype=float)
    return sigma * tau, (tau**2 - sigma**2) / 2


# ── Utilities ────────────────────────────────────────────────────────────────

def scale_factors(sigma: float, tau: float) -> Tuple[float, float]:
    """
    Compute the scale factors h_σ and h_τ of the parabolic coordinate system.

    h_σ = h_τ = sqrt(σ² + τ²)

    These relate arc lengths to coordinate differentials:
        ds² = h_σ² dσ² + h_τ² dτ²
    """
    h = math.sqrt(sigma**2 + tau**2)
    return h, h   # h_sigma == h_tau


def jacobian(sigma: float, tau: float) -> float:
    """
    Jacobian of the transformation  ∂(x,y)/∂(σ,τ) = σ² + τ²
    (the area element dA = (σ² + τ²) dσ dτ).
    """
    return sigma**2 + tau**2


# ── Demo / CLI ───────────────────────────────────────────────────────────────

def _print_table(points):
    header = f"{'x':>10} {'y':>10} │ {'σ':>12} {
        'τ':>12} │ {'x_back':>10} {'y_back':>10}  check"
    print(header)
    print("─" * len(header))
    for x, y in points:
        s, t = rect_to_parabolic(x, y)
        xb, yb = parabolic_to_rect(s, t)
        ok = "✓" if abs(xb - x) < 1e-10 and abs(yb - y) < 1e-10 else "✗"
        print(f"{x:>10.4f} {y:>10.4f} │ {s:>12.6f} {
              t:>12.6f} │ {xb:>10.4f} {yb:>10.4f}  {ok}")


# %%
if __name__ == "__main__":
    test_points = [
        (0.0,  0.0),
        (1.0,  0.0),
        (0.0,  1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (3.0,  4.0),
        (-3.0, 4.0),
        (1000, 500),
        (-1000.0, -500),
        (1.5,  2.7),
    ]

    print("\n═══ Rectangular → Parabolic (and back) ═══\n")
    _print_table(test_points)

    print("\n═══ Scale factors & Jacobian at selected points ═══\n")
    for x, y in [(1, 0), (3, 4), (0, 1)]:
        s, t = rect_to_parabolic(x, y)
        hs, ht = scale_factors(s, t)
        J = jacobian(s, t)
        print(f"  (x,y)=({x},{y})  →  σ={s:.4f}, τ={t:.4f}  "
              f"│  h_σ=h_τ={hs:.4f}  │  J={J:.4f}")

    print()

    # %%
tel_sc = .18
yy = np.linspace(-1920 * tel_sc / 2, 1920 * tel_sc / 2, 1920)
xx = np.linspace(-1080 * tel_sc / 2, 1080 * tel_sc / 2, 1080)
# %%
x = np.linspace(-108 * tel_sc / 2, 108 * tel_sc / 2, 100)
y = np.linspace(-192 * tel_sc / 2, 192 * tel_sc / 2, 100)

[sig, tau] = rect_to_parabolic_array(x, y)

# %%

um, vm = 14.7, 7
us = 0.05
vs = us  # / 2
# u = np.arange(0, um + us, us)
# v = np.arange(0, vm + vs, vs)
# u = np.linspace(-um, um, int(um/us))
u = np.linspace(12, um, 100)
v = np.linspace(0, vm, int(vm/vs))
uneg = u[u < 0]  # -u[1:][::-1]
u_non_neg = u[u >= 0]
# uc = np.concatenate([uneg, u])

sp = 8.0                # pixel size (microns)
lambd = 0.782 / sp      # wavelength
kp = 1e-3                # transversal part of wave vector
a_param = 5.0          # parameter "a" for Weber beam
np_val = 3.0

# --- Confluent Hypergeometric Functions ---
zu = 1j * kp * u**2
zu_noneg = 1j * kp * u_non_neg**2
zv = 1j * kp * v**2

FUe_max_real = []
FUe_max_abs = []
FUo_max_real = []
FUo_max_abs = []
FUo_max_abs_real = []
u_mxs = []
a_p = []
figure()
for a_param in range(0, 25, 6):
    umx_pred = np.sqrt(2*a_param + 1.6235*a_param**0.3906)
    umx_lin = np.linspace(umx_pred-0.3, umx_pred+0.2, 100)
    zu_mx_pred = 1j*kp*umx_lin**2
    # F11Ue_nn = hyp1f1_vec(0.25 - 0.5j * a_param, 0.5, zu_noneg).astype(complex)
    F11Uo_nn = hyp1f1_vec(0.75 - 0.5j * a_param, 1.5,
                          zu_mx_pred).astype(complex)

    # FUe_max_real.append(np.argmax(np.real(F11Ue_nn)*u_non_neg))
    # FUe_max_abs.append(np.argmax(np.abs(F11Ue_nn*u_non_neg)))

    # FUo_max_real.append(np.argmax(np.real(F11Uo_nn)))
    fmax_idx = np.argmax(np.abs(F11Uo_nn))
    FUo_max_abs.append(fmax_idx)
    # u_mxs.append(u_non_neg[fmax_idx])
    u_mxs.append(umx_lin[fmax_idx])
    a_p.append(a_param)

    # FUo_max_abs_real.append(np.argmax(np.abs(np.real(F11Uo_nn))))

    plt.plot(umx_lin, np.abs(F11Uo_nn) /
             np.max(np.abs(F11Uo_nn)), '.-',  label=f"a = {a_param}")
a_p = np.array(a_p)
plt.legend()
plt.show()
# %%
figure()
plot(a_p, u_mxs, '.-')
# %%

y_40_80 = u_non_neg[FUo_max_abs]
umax_num = np.concat((y1np, y_20_40))
a = np.arange(0, 40)
x = a
y = umax_num
y[0] = 0
coeffs = np.polyfit(x, y, deg=4)
y_fit = np.polyval(coeffs, x)

plt.plot(x, y, 'o', label='data')
plt.plot(x, y_fit, label='fit')
plt.legend()
plt.show()

# %%

# s controls smoothing (s=0 interpolates exactly)
spline = UnivariateSpline(x, y, s=1)

x_fit = np.linspace(x.min(), x.max(), 200)
plt.plot(x, y, 'o', label='data')
plt.plot(x_fit, spline(x_fit), label='spline fit')
plt.legend()
plt.show()

# %%
um, vm = 7, 7
us = 0.005
vs = us  # / 2
u = np.linspace(-um, um, int(um/us))
v = np.linspace(0, vm, int(vm/vs))
uneg = u[u < 0]  # -u[1:][::-1]
u_non_neg = u[u >= 0]
# uc = np.concatenate([uneg, u])

sp = 8.0                # pixel size (microns)
lambd = 0.782 / sp      # wavelength
kp = 1.0                # transversal part of wave vector
a_param = 5.0          # parameter "a" for Weber beam

for a_param in range(0, 21):
    F11Uo_nn = hyp1f1_vec(0.75 - 0.5j * a_param, 1.5, zu_noneg).astype(complex)

    FUo_max_abs.append(np.argmax(np.abs(F11Uo_nn)))

# %%


def predict_u_max(a):
    # Using the Power Law parameters found
    return 1.765 * (a**0.445)


# Example: Predict u for a = 50
print(f"Predicted u_max for a=50: {predict_u_max(50):.4f}")
# %%
umax_pred = []
for a in range(0, 40):
    umax_pred.append(predict_u_max(a))

# %%
# Data provided
a_param = np.arange(0, 40)
u_vals = np.array([0., 1.78454065, 2.43463352, 2.90041434, 3.28046864,
                   3.61337334, 3.91198743, 4.18631233, 4.44063438, 4.67923989,
                   4.90498643, 5.12073153, 5.32504644, 5.52221746, 5.71224461,
                   5.8936991, 6.07086727, 6.24232033, 6.40948707, 6.57093871,
                   6.72953279, 6.88241177, 7.0324332, 7.17959709, 7.32390341,
                   7.46535219, 7.60251464, 7.73824832, 7.87112445, 8.0025718,
                   8.13116159, 8.25689384, 8.38119731, 8.50407201, 8.62551793,
                   8.7441063, 8.8612659, 8.97699671, 9.09129876, 9.20417202])

# Define fitting functions


def sqrt_fit(x, k, c):
    return k * np.sqrt(x) + c


def power_fit(x, k, p):
    return k * (x**p)


# Fit 1: sqrt
popt_sqrt, _ = curve_fit(sqrt_fit, a_param, u_vals)
# Fit 2: power law (skipping 0 to avoid log(0) issues if used, but curve_fit handles it)
popt_pow, _ = curve_fit(power_fit, a_param, u_vals, p0=[1.5, 0.5])

# Plotting
plt.scatter(a_param, u_vals, color='red', label='Data Points')
x_smooth = np.linspace(0, 40, 100)
plt.plot(x_smooth, sqrt_fit(x_smooth, *popt_sqrt), 'b-',
         label=f'Fit: {popt_sqrt[0]:.3f}√a + {popt_sqrt[1]:.3f}')
plt.plot(x_smooth, power_fit(x_smooth, *popt_pow), 'g--',
         label=f'Fit: {popt_pow[0]:.3f} * a^{popt_pow[1]:.3f}')

plt.title("Fitting Weber Beam Maximum $u$ vs parameter $a$")
plt.xlabel("Parameter a")
plt.ylabel("u coordinate of max abs value")
plt.legend()
plt.grid(True, alpha=0.3)
# plt.savefig('weber_fit.png')

print(f"Square root fit params: k={popt_sqrt[0]}, c={popt_sqrt[1]}")
print(f"Power fit params: k={popt_pow[0]}, p={popt_pow[1]}")

# %%
# matplotlib.use('Agg')

y = np.array([0., 1.78454065, 2.43463352, 2.90041434, 3.28046864,
              3.61337334, 3.91198743, 4.18631233, 4.44063438, 4.67923989,
              4.90498643, 5.12073153, 5.32504644, 5.52221746, 5.71224461,
              5.8936991, 6.07086727, 6.24232033, 6.40948707, 6.57093871,
              6.72953279, 6.88241177, 7.0324332, 7.17959709, 7.32390341,
              7.46535219, 7.60251464, 7.73824832, 7.87112445, 8.0025718,
              8.13116159, 8.25689384, 8.38119731, 8.50407201, 8.62551793,
              8.7441063, 8.8612659, 8.97699671, 9.09129876, 9.20417202])

x = np.arange(0, 40)
xf = x[1:]
yf = y[1:]

# Key insight: ratio ~ 1.0 + 0.294/sqrt(a) -> C ~ 1.0 exactly!
# So y ~ sqrt(2a) + 0.294*sqrt(2a)/sqrt(a) = sqrt(2a) + 0.294*sqrt(2)
# i.e., y ~ sqrt(2a) + const??  Let's check:
res_check = yf - (np.sqrt(2*xf) + 0.294*np.sqrt(2))
print("y - (sqrt(2a) + 0.294*sqrt(2)) residuals:", np.round(res_check[:10], 4))
# That doesn't work directly. Let me think:
# ratio = y/sqrt(2a) ~ 1 + 0.294/sqrt(a) = 1 + 0.294/sqrt(a)
# So y ~ sqrt(2a) + 0.294*sqrt(2a)/sqrt(a) = sqrt(2a) + 0.294*sqrt(2)
# wait: sqrt(2a)/sqrt(a) = sqrt(2) -> constant!
# So y ~ sqrt(2a) + 0.294*sqrt(2) = sqrt(2a) + 0.416


def form_sqrt2a_const(a, C):
    return np.sqrt(2*a) + C


popt_sc, _ = curve_fit(form_sqrt2a_const, xf, yf, p0=[0.4])
y_pred_sc = form_sqrt2a_const(xf, *popt_sc)
res_sc = yf - y_pred_sc
rmse_sc = np.sqrt(np.mean(res_sc**2))
print(f"\ny = sqrt(2a) + {popt_sc[0]:.6f}")
print(f"  RMSE={rmse_sc:.6f}, Max|res|={np.max(np.abs(res_sc)):.6f}")
# But this doesn't pass through y(0)=0!

# The ratio extrapolation tells us C_inf = 0.9993 ~ 1
# So for large a: y -> sqrt(2a) asymptotically ✓
# The correction term must vanish as a->inf AND go to 0 as a->0
# Best physical form: y = sqrt(2a + B*a^gamma) with gamma < 1

# Competitive forms:
# 1. sqrt(2a + B*sqrt(a)):  y(0)=0 ✓, large-a: sqrt(2a+B*sqrt(a))~sqrt(2a)*(1+B/(4a^(1/2)))
#    gives ratio ~ 1 + B/(4*sqrt(2a))... so B/(4*sqrt(2)) = 0.294 -> B = 0.294*4*sqrt(2) = 1.66
# Let's check
print(f"\nPredicted B for sqrt(a) form: {0.294*4*np.sqrt(2):.4f}")
# compare to fitted B=1.193...not matching, probably nonlinear effects

# Most important: which form fits BEST with physically motivated structure?
# Let's try: y^2 = 2a * f(a) where f->1 as a->inf, f(0)->0

# y^2:
y2 = yf**2
print(f"\ny^2 / (2a) ratios (last 5): {(y2/(2*xf))[-5:]}")
# This gives y^2/(2a) -> should go to 1


def y2_form(a, B, gamma):
    return 2*a + B*a**gamma


popt_y2f, pcov_y2f = curve_fit(y2_form, xf, y2, p0=[1.5, 0.5])
perr_y2f = np.sqrt(np.diag(pcov_y2f))
y2_pred = y2_form(xf, *popt_y2f)
res_y2 = y2 - y2_pred
rmse_y2 = np.sqrt(np.mean(res_y2**2))
print(f"\ny^2 = 2a + {popt_y2f[0]:.6f}*a^{popt_y2f[1]:.6f}")
print(f"  RMSE(y^2)={rmse_y2:.6f}")
y_from_y2 = np.sqrt(y2_form(xf, *popt_y2f))
res_y_from_y2 = yf - y_from_y2
rmse_y_from_y2 = np.sqrt(np.mean(res_y_from_y2**2))
print(
    f"  -> y = sqrt(2a + {popt_y2f[0]:.4f}*a^{popt_y2f[1]:.4f}), RMSE(y)={rmse_y_from_y2:.6f}")
print(f"  gamma ± {
      perr_y2f[1]:.6f} -> close to 1/2? diff={abs(popt_y2f[1]-0.5):.4f}")

# WINNER: y = sqrt(2a + B*a^(1/2)) -- try fixing gamma=1/2


def fixed_half(a, B):
    return np.sqrt(2*a + B*np.sqrt(a))


popt_fh, pcov_fh = curve_fit(fixed_half, xf, yf, p0=[1.2])
perr_fh = np.sqrt(np.diag(pcov_fh))
y_pred_fh = fixed_half(xf, *popt_fh)
res_fh = yf - y_pred_fh
rmse_fh = np.sqrt(np.mean(res_fh**2))
print(f"\nFIXED gamma=1/2: y = sqrt(2a + {popt_fh[0]:.6f}*sqrt(a))")
print(f"  B ± {perr_fh[0]:.6f}")
print(f"  RMSE={rmse_fh:.6f}, Max|res|={np.max(np.abs(res_fh)):.6f}")
print(f"  Residuals (x1000): {np.round(res_fh*1000, 1)}")

# %% ---- FINAL PLOT ----
figCL, axes = plt.subplots(1, 2, figsize=(16, 4))  # , num='CLfits'
x_fine = np.linspace(0.001, 40, 800)


def sqrtform2(a, B, gamma):
    return np.sqrt(2*a + B*a**gamma)


popt_sf2 = [popt_y2f[0], popt_y2f[1]]  # from y^2 fit

ax = axes[0]
ax.plot(x, y, 'ko', ms=5, zorder=6, label='Data', alpha=0.9)
ax.plot(x_fine, fixed_half(x_fine, *popt_fh), 'b-', lw=2.2,
        label=f'$\\sqrt{{2a + {popt_fh[0]:.3f}\\,\\sqrt{{a}}}}$\n  RMSE={rmse_fh:.4f}')
ax.plot(x_fine, sqrtform2(x_fine, *popt_sf2), 'r--', lw=2,
        label=f'$\\sqrt{{2a + {popt_sf2[0]:.3f}\\,a^{{{popt_sf2[1]:.3f}}}}}$\n  RMSE={rmse_y_from_y2:.4f}')
ax.plot(x_fine, np.sqrt(2*x_fine), 'm:', lw=1.8,
        label=r'$\sqrt{2a}$ (classical t.p.)')
ax.set_xlabel('$a$ parameter', fontsize=13)
ax.set_ylabel('$u_{\\rm max}$', fontsize=13)
ax.set_title(
    'Weber beam: max of $|{}_1F_1(\\frac{3}{4}-\\frac{ia}{2},\\frac{3}{2},z)|$ vs $a$', fontsize=11)
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(xf, res_fh*1.00, 'b-o', ms=4, lw=1.5,
         label=f'$\\sqrt{{2a+{popt_fh[0]:.3f}\\sqrt{{a}}}}$ (RMSE={rmse_fh:.4f})')
ax2.plot(xf, res_y_from_y2*1.00, 'r--^', ms=4, lw=.5,
         label=f'$\\sqrt{{2a+{popt_sf2[0]:.3f}\\,a^{{{popt_sf2[1]:.3f}}}}}$ (RMSE={rmse_y_from_y2:.4f})')
ax2.axhline(0, color='gray', ls='--')
ax2.set_xlabel('$a$ parameter', fontsize=13)
ax2.set_ylabel('Residual × 1.00', fontsize=13)
ax2.set_title('Fit residuals', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
# plt.savefig('/mnt/user-data/outputs/weber_fit_v2.png', dpi=150, bbox_inches='tight')
# print("\nPlot saved!")

# Summary
print("\n" + "="*55)
print("SUMMARY OF BEST FITS")
print("="*55)
print(f"1. y = sqrt(2a + {popt_fh[0]:.4f}*sqrt(a))")
print(f"   RMSE={rmse_fh:.5f}, Max|res|={np.max(np.abs(res_fh)):.5f}")
print(f"   B ± {perr_fh[0]:.4f}")
print(f"\n2. y = sqrt(2a + {popt_sf2[0]:.4f}*a^{popt_sf2[1]:.4f})")
print(f"   RMSE={rmse_y_from_y2:.5f}")
print(f"   Both pass through y(0)=0 ✓")
print(f"   Both -> sqrt(2a) as a->inf ✓")

# %% Extending values
y00 = np.array([0., 1.78454065, 2.43463352, 2.90041434, 3.28046864,
                3.61337334, 3.91198743, 4.18631233, 4.44063438, 4.67923989,
                4.90498643, 5.12073153, 5.32504644, 5.52221746, 5.71224461,
                5.8936991, 6.07086727, 6.24232033, 6.40948707, 6.57093871,
                6.72953279, 6.88241177, 7.0324332, 7.17959709, 7.32390341,
                7.46535219, 7.60251464, 7.73824832, 7.87112445, 8.0025718,
                8.13116159, 8.25689384, 8.38119731, 8.50407201, 8.62551793,
                8.7441063, 8.8612659, 8.97699671, 9.09129876, 9.20417202])

y = np.concatenate((y00, u_mxs))
x = np.concatenate((np.arange(0, 40), a_p))
xf = x[1:]
yf = y[1:]

# Key insight: ratio ~ 1.0 + 0.294/sqrt(a) -> C ~ 1.0 exactly!
# So y ~ sqrt(2a) + 0.294*sqrt(2a)/sqrt(a) = sqrt(2a) + 0.294*sqrt(2)
# i.e., y ~ sqrt(2a) + const??  Let's check:
res_check = yf - (np.sqrt(2*xf) + 0.294*np.sqrt(2))
print("y - (sqrt(2a) + 0.294*sqrt(2)) residuals:", np.round(res_check[:10], 4))
# That doesn't work directly. Let me think:
# ratio = y/sqrt(2a) ~ 1 + 0.294/sqrt(a) = 1 + 0.294/sqrt(a)
# So y ~ sqrt(2a) + 0.294*sqrt(2a)/sqrt(a) = sqrt(2a) + 0.294*sqrt(2)
# wait: sqrt(2a)/sqrt(a) = sqrt(2) -> constant!
# So y ~ sqrt(2a) + 0.294*sqrt(2) = sqrt(2a) + 0.416


def form_sqrt2a_const(a, C):
    return np.sqrt(2*a) + C


popt_sc, _ = curve_fit(form_sqrt2a_const, xf, yf, p0=[0.4])
y_pred_sc = form_sqrt2a_const(xf, *popt_sc)
res_sc = yf - y_pred_sc
rmse_sc = np.sqrt(np.mean(res_sc**2))
print(f"\ny = sqrt(2a) + {popt_sc[0]:.6f}")
print(f"  RMSE={rmse_sc:.6f}, Max|res|={np.max(np.abs(res_sc)):.6f}")
# But this doesn't pass through y(0)=0!

# The ratio extrapolation tells us C_inf = 0.9993 ~ 1
# So for large a: y -> sqrt(2a) asymptotically ✓
# The correction term must vanish as a->inf AND go to 0 as a->0
# Best physical form: y = sqrt(2a + B*a^gamma) with gamma < 1

# Competitive forms:
# 1. sqrt(2a + B*sqrt(a)):  y(0)=0 ✓, large-a: sqrt(2a+B*sqrt(a))~sqrt(2a)*(1+B/(4a^(1/2)))
#    gives ratio ~ 1 + B/(4*sqrt(2a))... so B/(4*sqrt(2)) = 0.294 -> B = 0.294*4*sqrt(2) = 1.66
# Let's check
print(f"\nPredicted B for sqrt(a) form: {0.294*4*np.sqrt(2):.4f}")
# compare to fitted B=1.193...not matching, probably nonlinear effects

# Most important: which form fits BEST with physically motivated structure?
# Let's try: y^2 = 2a * f(a) where f->1 as a->inf, f(0)->0

# y^2:
y2 = yf**2
print(f"\ny^2 / (2a) ratios (last 5): {(y2/(2*xf))[-5:]}")
# This gives y^2/(2a) -> should go to 1


def y2_form(a, B, gamma):
    return 2*a + B*a**gamma


popt_y2f, pcov_y2f = curve_fit(y2_form, xf, y2, p0=[1.5, 0.5])
perr_y2f = np.sqrt(np.diag(pcov_y2f))
y2_pred = y2_form(xf, *popt_y2f)
res_y2 = y2 - y2_pred
rmse_y2 = np.sqrt(np.mean(res_y2**2))
print(f"\ny^2 = 2a + {popt_y2f[0]:.6f}*a^{popt_y2f[1]:.6f}")
print(f"  RMSE(y^2)={rmse_y2:.6f}")
y_from_y2 = np.sqrt(y2_form(xf, *popt_y2f))
res_y_from_y2 = yf - y_from_y2
rmse_y_from_y2 = np.sqrt(np.mean(res_y_from_y2**2))
print(
    f"  -> y = sqrt(2a + {popt_y2f[0]:.4f}*a^{popt_y2f[1]:.4f}), RMSE(y)={rmse_y_from_y2:.6f}")
print(f"  gamma ± {
      perr_y2f[1]:.6f} -> close to 1/2? diff={abs(popt_y2f[1]-0.5):.4f}")

# WINNER: y = sqrt(2a + B*a^(1/2)) -- try fixing gamma=1/2


def fixed_half(a, B):
    return np.sqrt(2*a + B*np.sqrt(a))


popt_fh, pcov_fh = curve_fit(fixed_half, xf, yf, p0=[1.2])
perr_fh = np.sqrt(np.diag(pcov_fh))
y_pred_fh = fixed_half(xf, *popt_fh)
res_fh = yf - y_pred_fh
rmse_fh = np.sqrt(np.mean(res_fh**2))
print(f"\nFIXED gamma=1/2: y = sqrt(2a + {popt_fh[0]:.6f}*sqrt(a))")
print(f"  B ± {perr_fh[0]:.6f}")
print(f"  RMSE={rmse_fh:.6f}, Max|res|={np.max(np.abs(res_fh)):.6f}")
print(f"  Residuals (x1000): {np.round(res_fh*1000, 1)}")

# %% ---- FINAL PLOT ----
figCL, axes = plt.subplots(1, 2, figsize=(16, 4))  # , num='CLfits'
x_fine = xf  # np.linspace(0.001, 40, 800)


def sqrtform2(a, B, gamma):
    return np.sqrt(2*a + B*a**gamma)


popt_sf2 = [popt_y2f[0], popt_y2f[1]]  # from y^2 fit

ax = axes[0]
ax.plot(x, y, 'ko', ms=5, zorder=6, label='Data', alpha=0.9)
ax.plot(x_fine, fixed_half(x_fine, *popt_fh), 'b-', lw=2.2,
        label=f'$\\sqrt{{2a + {popt_fh[0]:.3f}\\,\\sqrt{{a}}}}$\n  RMSE={rmse_fh:.4f}')
ax.plot(x_fine, sqrtform2(x_fine, *popt_sf2), 'r--', lw=2,
        label=f'$\\sqrt{{2a + {popt_sf2[0]:.3f}\\,a^{{{popt_sf2[1]:.3f}}}}}$\n  RMSE={rmse_y_from_y2:.4f}')
ax.plot(x_fine, np.sqrt(2*x_fine), 'm:', lw=1.8,
        label=r'$\sqrt{2a}$ (classical t.p.)')
ax.set_xlabel('$a$ parameter', fontsize=13)
ax.set_ylabel('$u_{\\rm max}$', fontsize=13)
ax.set_title(
    'Weber beam: max of $|{}_1F_1(\\frac{3}{4}-\\frac{ia}{2},\\frac{3}{2},z)|$ vs $a$', fontsize=11)
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(xf, res_fh*1.00, 'b-o', ms=4, lw=1.5,
         label=f'$\\sqrt{{2a+{popt_fh[0]:.3f}\\sqrt{{a}}}}$ (RMSE={rmse_fh:.4f})')
ax2.plot(xf, res_y_from_y2*1.00, 'r--^', ms=4, lw=.5,
         label=f'$\\sqrt{{2a+{popt_sf2[0]:.3f}\\,a^{{{popt_sf2[1]:.3f}}}}}$ (RMSE={rmse_y_from_y2:.4f})')
ax2.axhline(0, color='gray', ls='--')
ax2.set_xlabel('$a$ parameter', fontsize=13)
ax2.set_ylabel('Residual × 1.00', fontsize=13)
ax2.set_title('Fit residuals', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
# plt.savefig('/mnt/user-data/outputs/weber_fit_v2.png', dpi=150, bbox_inches='tight')
# print("\nPlot saved!")

# Summary
print("\n" + "="*55)
print("SUMMARY OF BEST FITS")
print("="*55)
print(f"1. y = sqrt(2a + {popt_fh[0]:.4f}*sqrt(a))")
print(f"   RMSE={rmse_fh:.5f}, Max|res|={np.max(np.abs(res_fh)):.5f}")
print(f"   B ± {perr_fh[0]:.4f}")
print(f"\n2. y = sqrt(2a + {popt_sf2[0]:.4f}*a^{popt_sf2[1]:.4f})")
print(f"   RMSE={rmse_y_from_y2:.5f}")
print(f"   Both pass through y(0)=0 ✓")
print(f"   Both -> sqrt(2a) as a->inf ✓")

# %%
au = 5
mpmath.hyp1f1(0.75 - 0.5j * au, 1.5, sqrtform2(au, *popt_sf2)**2*1j*1e-6)
