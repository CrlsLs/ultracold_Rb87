"""
Beam Splitter Analysis
Translated from MATLAB to Python.

Dependencies: numpy, scipy, matplotlib
Install with: pip install numpy scipy matplotlib
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# Helper functions (replaces MATLAB .m files)
# ─────────────────────────────────────────────

def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    """Two-peak Gaussian (MATLAB Gauss2 model)."""
    return a1 * np.exp(-((x - b1) / c1) ** 2) + a2 * np.exp(-((x - b2) / c2) ** 2)


def single_gaussian(x, a1, b1, c1):
    """Single Gaussian (MATLAB Gauss1 model)."""
    return a1 * np.exp(-((x - b1) / c1) ** 2)


def smooth(arr, window):
    """
    Replicate MATLAB's smooth() using a moving average.
    Returns array of same length (edges padded by replication).
    """
    arr = np.asarray(arr, dtype=float)
    if window <= 1:
        return arr.copy()
    window = int(window) | 1          # force odd
    half = window // 2
    padded = np.pad(arr, half, mode='edge')
    return np.convolve(padded, np.ones(window) / window, mode='valid')


def matlab_hist(data, n_bins):
    """Replicate MATLAB hist(): centres + counts, normalised later by caller."""
    counts, edges = np.histogram(data, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(float), centres


def normfit(data):
    """MATLAB normfit equivalent: returns (mu, sigma, mu_ci, sigma_ci)."""
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    n = len(data)
    # 95 % confidence intervals (approximate)
    from scipy import stats
    se_mu = sigma / np.sqrt(n)
    se_sig = sigma / np.sqrt(2 * (n - 1))
    t = stats.t.ppf(0.975, df=n - 1)
    mu_ci = (mu - t * se_mu, mu + t * se_mu)
    sigma_ci = (sigma - t * se_sig, sigma + t * se_sig)
    return mu, sigma, mu_ci, sigma_ci


def interp_upsample(arr, factor):
    """Replicate MATLAB interp(x, factor): integer upsampling."""
    n = len(arr)
    x_old = np.arange(n)
    x_new = np.linspace(0, n - 1, n * factor)
    f = interp1d(x_old, arr, kind='linear')
    return f(x_new)


def r_squared(y_actual, y_predicted):
    """Coefficient of determination R²."""
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

salvar = 'n'


# ==========================================
# Path Configuration (Cross-Platform)
# ==========================================
# Use Path() for the base directories
if os.name == 'nt':  # Windows
    base_dir = Path(r"E:\from CL7 HDD\proy t\lab ICN SLM\programas\beam_splitter")
else:  # Linux / Pop!_OS
    # Adjust this to where your drive is mounted on Linux
    # Usually under /media/user/DRIVENAME/...
    base_dir = Path("/mnt/SharedDisk/from CL7 HDD/proy t/lab ICN SLM/programas\beam_splitter")

a = 5
if a == 5:
    mw  = [43, 36, 30]
    tof = [18, 15, 13, 11, 10, 8, 7]

dira = str(a)
lmw  = len(mw)
ltof = len(tof)

legp = [str(v)        for v in mw]          # power labels
legt = [f"{v}ms"      for v in tof]         # tof labels

# Base directory (update to your local path)
dirtemp1 = dirtemp1 = base_dir / "a5kz09989" / "newCI_roc"

# ─────────────────────────────────────────────
# Output storage
# ─────────────────────────────────────────────

d_dir    = np.zeros((lmw, ltof))
d_aj     = np.zeros((lmw, ltof))
daj_max  = np.zeros((lmw, ltof))
r2h      = np.zeros((lmw, ltof))
r2v      = np.zeros((lmw, ltof))
cmx      = np.zeros((lmw, ltof))
cmy      = np.zeros((lmw, ltof))
wy       = np.zeros((lmw, ltof))
g2t      = np.zeros(lmw)

parh = [np.zeros((ltof, 8)) for _ in range(lmw)]   # 8 cols for safety
parv = [np.zeros((ltof, 5)) for _ in range(lmw)]

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

fig100, axes100 = plt.subplots(lmw, ltof, figsize=(3 * ltof, 3 * lmw), num=100)
fig2,   axes2   = plt.subplots(lmw, ltof, figsize=(3 * ltof, 3 * lmw), num=2)

for im, mw_val in enumerate(mw):
    for it, tof_val in enumerate(tof):

        # ── Find the folder matching this mw value ──────────────────────
        pattern_fol = os.path.join(dirtemp1, f'*{mw_val}*')
        fol1s = glob.glob(pattern_fol)
        if len(fol1s) != 1:
            print(f'WARNING: Expected 1 folder, found {len(fol1s)} for mw={mw_val}')
        fol1 = fol1s[0]

        # ── Find the data file ──────────────────────────────────────────
        pattern_dat = os.path.join(fol1, f'S_*{mw_val}*t{tof_val}*.dat')
        vars_files = glob.glob(pattern_dat)
        if len(vars_files) != 1:
            print(f'WARNING: Expected 1 file, found {len(vars_files)} → mw={mw_val}, tof={tof_val}')
            input('Press Enter to continue...')
        nombre = vars_files[0]
        file   = nombre

        # ── Find the onda (wave) reference file ─────────────────────────
        donda_files = glob.glob(os.path.join(fol1, '*onda*.dat'))
        donda  = donda_files[0]
        dirg2  = donda

        # ── Load data ───────────────────────────────────────────────────
        data = np.loadtxt(file)

        # ── Load onda .DAT file — key= value format ──────────────────────
        # File layout:
        #   detuning=  -10292.435...
        #   g2=   114.246532471751        ← target value
        #   g_quantum  -8.74e-8
        #   ...
        # MATLAB: g22 = importdata(dirg2); g2 = g22.data(2)
        # We match the 'g2=' key directly instead of relying on row index.
        g2 = None
        with open(dirg2, 'r') as fh:
            for line in fh:
                if line.strip().lower().startswith('g2='):
                    g2 = float(line.split('=')[1].strip())
                    break
        if g2 is None:
            raise ValueError(f"Could not find 'g2=' in {dirg2}")

        # ── Coordinate transform ─────────────────────────────────────────
        f_scale = 0.78
        y_raw = data[:, 0] * f_scale + 140
        x_raw = data[:, 1] * f_scale
        ze_raw= data[:, 2] * f_scale

        theta = 10 * np.pi / 180
        Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                       [            0,  1,             0],
                       [-np.sin(theta), 0, np.cos(theta)]])

        M  = np.column_stack([x_raw, y_raw, ze_raw])
        Mp = (Ry @ M.T).T
        x  = Mp[:, 0]
        y  = Mp[:, 1]
        ze = Mp[:, 2]

        # ── Histograms ───────────────────────────────────────────────────
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        lvx = xmax - xmin
        lvy = ymax - ymin

        dxx, dyy = 9, 9
        tbx = int(lvx / dxx)
        tby = int(lvy / dyy)

        msx, xs = matlab_hist(x, tbx)
        msy, ys = matlab_hist(y, tby)

        msx = msx / msx.max()
        msy = msy / msy.max()

        mux,  sigmax, muci,  sigmaci  = normfit(x)
        muy,  sigmay, mucy,  sigmacy  = normfit(y)

        msys = smooth(smooth(msy, 7), 7)
        msxs = smooth(smooth(msx, 7), 7)

        if tof_val == 9:
            msxs = smooth(msx, 7)
        if tof_val in (8, 7, 6):
            msxs = msx.copy()

        msxs = msxs.ravel()

        # ── Plot raw histogram (figure 100) ─────────────────────────────
        ax100 = axes100[im, it] if lmw > 1 else axes100[it]
        ax100.plot(xs, msx, 'g')

        msx0 = msx.copy()
        xs0  = xs.copy()

        # ── Peak detection (direct, before interp) ──────────────────────
        lxd   = len(msxs)
        icero = int(np.argmin(np.abs(xs)))

        fxleft  = msxs[:icero]
        fxright = msxs[icero + 1:]

        m1d = int(np.round(np.mean(np.where(fxleft  == fxleft.max())[0])))
        m2d = int(np.round(np.mean(np.where(fxright == fxright.max())[0]))) + icero + 1

        xm1d = xs[m1d]
        xm2d = xs[m2d]
        d    = int(np.round(abs(xm1d) + abs(xm2d)))
        print(f'xm1d={xm1d:.1f}  xm2d={xm2d:.1f}  d={d}')

        # ── Upsample (interp ×8) ─────────────────────────────────────────
        xs    = interp_upsample(xs,   8)
        msxs  = interp_upsample(msxs, 8)
        lxd   = len(msxs)
        icero = int(np.argmin(np.abs(xs)))

        fxleft  = msxs[:icero]
        fxright = msxs[icero + 1:]

        m1d = int(np.round(np.mean(np.where(fxleft  == fxleft.max())[0])))
        m2d = int(np.round(np.mean(np.where(fxright == fxright.max())[0]))) + icero + 1

        xm1d = xs[m1d]
        xm2d = xs[m2d]
        d    = int(np.round(abs(xm1d) + abs(xm2d)))
        d_dir[im, it] = d

        # ── Double-Gaussian fit ─────────────────────────────────────────
        p0 = [fxleft.max(), xm1d, d / 2,
              fxright.max(), xm2d, d / 2]

        try:
            popt_up, _ = curve_fit(double_gaussian, xs, msxs, p0=p0, maxfev=10000)
            a1, b1, c1, a2, b2, c2 = popt_up
        except RuntimeError:
            a1, b1, c1, a2, b2, c2 = p0

        # Second pass on original (xs0, msx0)
        try:
            popt0, _ = curve_fit(double_gaussian, xs0, msx0,
                                 p0=[a1, b1, c1, a2, b2, c2], maxfev=10000)
            a1, b1, c1, a2, b2, c2 = popt0
        except RuntimeError:
            pass

        parh[im][it, 0] = a1
        parh[im][it, 1] = b1
        parh[im][it, 2] = c1 / np.sqrt(2)
        parh[im][it, 3] = a2
        parh[im][it, 4] = b2
        parh[im][it, 5] = c2 / np.sqrt(2)
        parh[im][it, 6] = 0

        y_pred = double_gaussian(xs0, a1, b1, c1, a2, b2, c2)
        r2h[im, it] = r_squared(msx0, y_pred)
        cmx[im, it] = b1

        # ── Plot fitted curve ────────────────────────────────────────────
        aj = double_gaussian(xs, a1, b1, c1, a2, b2, c2)
        ax100.plot(xs, aj, 'c')
        ax100.set_xlim([-1.2 * d, 1.2 * d])

        ajleft  = aj[:icero]
        ajright = aj[icero + 1:]
        m1daj = int(np.round(np.mean(np.where(ajleft  == ajleft.max())[0])))
        m2daj = int(np.round(np.mean(np.where(ajright == ajright.max())[0]))) + icero + 1
        xm1daj = xs[m1daj]
        xm2daj = xs[m2daj]

        daj_max[im, it] = int(np.round(abs(xm1daj) + abs(xm2daj)))
        d_aj[im, it]    = abs(parh[im][it, 1]) + abs(parh[im][it, 4])

        # ── Vertical (y) position and time of fall ───────────────────────
        posy_idx = int(np.round(np.mean(np.where(msys == msys.max())[0])))
        posy = ys[posy_idx]
        tc   = np.sqrt(2 / 9.8 * posy)

        print(f'g2s={np.sqrt(g2):4.2f}  xm1d={xm1d:4.0f}  xm2d={xm2d:4.0f}  '
              f'posy={posy:4.0f}  tc={tc:4.1f}')
        print(f'd_dir={d:4.0f}  d_aj={d_aj[im,it]:4.0f}  '
              f'daj_max={daj_max[im,it]:4.0f}  r2={r2h[im,it]:f}')

        ax100.set_xlim([xmin, xmax])
        ax100.set_ylim([0, 1.1])
        ax100.set_title(f'g={np.sqrt(g2):2.1f} tof={legt[it]}')

        # ── Vertical Gaussian fit ────────────────────────────────────────
        try:
            popt_v, _ = curve_fit(single_gaussian, ys, msy,
                                  p0=[msy.max(), ys[np.argmax(msy)], 5],
                                  maxfev=10000)
            av1, bv1, cv1 = popt_v
        except RuntimeError:
            av1, bv1, cv1 = msy.max(), ys[np.argmax(msy)], 5

        parv[im][it, 0] = av1
        parv[im][it, 1] = bv1
        parv[im][it, 2] = cv1 / np.sqrt(2)
        parv[im][it, 3] = 0

        y_pred_v = single_gaussian(ys, av1, bv1, cv1)
        r2v[im, it] = r_squared(msy, y_pred_v)
        cmy[im, it] = bv1
        wy[im, it]  = parv[im][it, 2]

        ajv = single_gaussian(ys, av1, bv1, cv1)

        ax2 = axes2[im, it] if lmw > 1 else axes2[it]
        ax2.plot(ys, msy, 'g', ys, ajv, 'c')
        ax2.set_xlim([cmy[im, it] - 4 * wy[im, it],
                      cmy[im, it] + 4 * wy[im, it]])
        ax2.set_ylim([0, 1.1])
        ax2.set_title(f'g={np.sqrt(g2):2.1f}  r2={r2v[im,it]:1.3f}')

        print(f'wy={wy[im,it]:4.2f}  r2v={r2v[im,it]:1.4f}  cmy={cmy[im,it]:3.2f}')

    g2t[im] = np.sqrt(g2)

# ─────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────

tof_col       = np.array(tof).reshape(-1, 1)
tabla_teo_aj  = np.hstack([tof_col, np.fliplr(d_aj.T)])
tabla_aj_max  = np.hstack([tof_col, np.fliplr(daj_max.T)])

print('\ng2t:')
print(g2t)
print('\ntabla_teo_aj (flipud):')
print(np.flipud(tabla_teo_aj))
print('\ntabla_teo_aj_max (flipud):')
print(np.flipud(tabla_aj_max))
print('\nwy (rot90×2):')
print(np.rot90(wy, 2))

# ─────────────────────────────────────────────
# Save (optional)
# ─────────────────────────────────────────────

now      = datetime.now()
f0       = [now.year, now.month, now.day, now.hour, now.minute, now.second]
fsave    = ''.join(f'{v:02d}' for v in f0[1:6])    # MMDDHHMMSS equivalent
namesave = os.path.join(dirtemp1, f'bs_max_a{a}_{fsave}.npz')

if salvar == 'y':
    np.savez(namesave,
             d_dir=d_dir, d_aj=d_aj, daj_max=daj_max,
             r2h=r2h, r2v=r2v, cmx=cmx, cmy=cmy, wy=wy,
             g2t=g2t, tof=tof, mw=mw,
             tabla_teo_aj=tabla_teo_aj, tabla_aj_max=tabla_aj_max)
    print(f'Saved to {namesave}')

plt.tight_layout()
plt.show()