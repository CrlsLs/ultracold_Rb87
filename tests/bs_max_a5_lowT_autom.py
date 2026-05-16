# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:23:36 2026

@author: clher
"""

# %% lib
# clear all
# %close all
from pathlib import Path
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from datetime import datetime


salvar = 'n';
# %load('paramVac_a10.mat'); par0foocell=paramVac; clear paramVac_a10


a = 5;  # %tof = 18:-6:6; tof=fliplr(tof);
# if a==5
mw = [43, 36, 30]
tof = [18, 15, 13, 11, 10, 8, 7]
# end
# %mw = [36 25 15];
# %mw  = [36];
# %tof = [18 13 10 7]; %tof=fliplr(tof);
# %tof=18;

dira = str(a);
# %tof = [18 13  10 7 0]; ltof=length(tof);
lmw = len(mw); ltof = len(tof);

# % gp=[9.73 9.1 8.7 7.9]; lmw=length(gp);
# % for il=1:lmw
# %      legp{il}=[num2str(gp(il)) '\Gamma']; %leyendas power
# % end

legp = [str(pw) for pw in mw]          # power labels
legt = [f"{t}ms" for t in tof]
# end
# %% dirs
# %S_25mWtl1p5m3t8ms_T0p_fc0.DAT
# %S_30mWtl1p5m3t10ms_T0p_fc6um.DAT
# %S_36mWtl1p5m3t17ms_T0p_fc6um
# %\a5kz09989\LOWTMP\10000\43mW\S_43mWtl1p5m3t18ms_T0p_fc0.DAT
# %\a5kz09989\LOWTMP\10000\43mW\S_43mWtl1p5m3t15ms_fc0_tg1.DAT',1)
# %\a5kz09989\LOWTMP\0p33\36\S_36l1p5m3t18ms_fc0_tg1.DAT',1)
# %\a5kz09989\LOWTMP\0p33\36\S_36l1p5m3t18ms_fc0_tg1.DAT
# % dir1 = 'F:\Users\carlosluis\Documents\proy t\lab ICN SLM\programas\beam_splitter\a';
dir1 = r'E:\from CL7 HDD\proy t\lab ICN SLM\programas\beam_splitter\a';
dir2 = dira;  # %este es el valor de a
dirnp = '0p33';
dir3 = r'kz09989\2012enero\LOWTMP' + '\\' + dirnp + '\\'
# %dir3 = 'kz09989\cambioCI\170'
dir4 = legp;  # %mW
dir5 = '\S_';
dir6 = dir4;  # %mW
dir7 = 'l1p5m3t';
dir8 = legt;  # %tof
# %dir9 = '_T0p_fc0.DAT';
dir9 = '_fc0_tg1.DAT';

# %% Loop
# %dirtemp1=[dir1 dir2 dir3];
# %dirtemp1='F:\Users\carlosluis\Documents\proy t\lab ICN SLM\programas\beam_splitter\a5kz09989\2012enero\LOWTEMP\0p33\';
# %dirtemp1='F:\Users\carlosluis\Documents\proy t\lab ICN SLM\programas\beam_splitter\a5kz09989\2012enero\LOWTmCI\p33\';

dirtemp1 = Path(
    r'E:\from CL7 HDD\proy t\lab ICN SLM\programas\beam_splitter\a5kz09989\newCI_roc')

for im in range(lmw):
    for it in range(ltof):
        file = dir1 + dir2 + dir3 + dir4[im] + \
            dir5 + dir6[im] + dir7 + dir8[it] + dir9

        # %%
        # del vars nombres temp1
        # %dirtemp = [dir1 dir2 dir3 num2str(mw(im))];
        search_pattern_fol = os.path.join(dirtemp1, f"*{mw[im]}*")
        fol1_list = glob.glob(search_pattern_fol)
        dirtemp1.is_dir()
        if len(fol1_list) > 1:
            print("VARIOS FOLDERS")
        fol1 = os.path.basename(fol1_list[0]) if fol1_list else ""
        # %%
        if numel(fol1)~ = 1
            disp('VARIOS FOLDERS ')
        end
        clear fol1;
        fol1 = fol1s.name;

        fol1s = glob.glob(pattern_fol)
        if len(fol1s) != 1:
            print(f'WARNING: Expected 1 folder, found {
                  len(fol1s)} for mw={mw_val}')
            fol1 = fol1s[0]

        # %%
        vars = dir([dirtemp1 fol1 '\S_*' num2str(mw(im)) '*t' num2str(tof(it)) '*.dat']);
        nombres = {vars.name};
        if numel(nombres)~ = 1
            disp('VARIOS FILES '); nombres
            pause
        end
        clear nombres;
        nombre = vars.name;
        file = [dirtemp1 fol1 '\' nombre];
#%%
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



        # %%
        donda1=dir([dirtemp1 fol1 '\*onda*.dat']);
        donda=donda1.name;
        dirg2=[dirtemp1 fol1 '\' donda];
        # %%
        del data g22
        data=load(file);
        g22=importdata(dirg2); g2=g22.data(2);

        f=0.78;
        y=data(:,1)*f+140; x=data(:,2)*f; ze=data(:,3)*f; 
        # %%
        theta=10*pi/180;
        Rx=[[1       0          0    ];...
            [0 cos(theta) -sin(theta)];...
            [0 sin(theta)  cos(theta)]];
    
        Ry=[[cos(theta)  0 sin(theta) ];...
            [     0      1      0      ];...
            [-sin(theta) 0 cos(theta)]];
    
        Rz=[[cos(theta) -sin(theta) 0 ];...
            [sin(theta)  cos(theta) 0 ];...
            [    0             1    0 ]];
        M=[x y ze];
    # %%
        Mp =(Ry*M')';
        x=Mp(:,1); y=Mp(:,2); ze=Mp(:,3);
# %%

        xmin=min(x);  xmax=max(x); ymin=min(y);  ymax=max(y);
        l=length(x); lvx=xmax-xmin;   ly=length(y); lvy=ymax-ymin;
        
        dxx=9; dyy=9;
        tbx=lvx/dxx; tby=lvy/dyy;
        [msx,xs] = hist(x,tbx);  [msy,ys] = hist(y,tby);
        msx=msx/max(msx);  msy=msy/max(msy);
        [mux,sigmax,muci,sigmaci] = normfit(x); [muy,sigmay,mucy,sigmacy] = normfit(y);
        msys=smooth(smooth(msy,7));
        
        %msxs=smooth(smooth(msx,1));
        msxs=smooth(smooth(msx,7));
        %msxs=(smooth(msx,7));
    if (tof(it)==9)
        msxs=(smooth(msx,7)); flagtof=9;
    end        
    if (tof(it)==8 || tof(it)==7  || tof(it)==6)
       % msxs=(smooth(msx));
       msxs=msx';
        flagtof=1;
    end
        %msxs=msx';
    msxs=msxs';
       
    figure (100)
    subplot(lmw,ltof,it+(im-1)*ltof); 
    %plot(xs,(smooth(msx)))
    plot(xs,msx, 'g')
    hold on
    %plot(xs,msx, 'r')
    %%
    msx0=msx; xs0=xs;
    lxd=length(msxs);
    icero=find(abs(xs)==min(abs(xs)));
    fxleft=msxs(1:icero); fxright=msxs(icero+1:lxd);
    m1d=round(mean(find(fxleft==max(fxleft)))); m2d=round(mean(find(fxright==max(fxright))))+icero; %m2d=round(mean(find(fx(icero+1:lxd)==max(fx(icero+1:lxd)))))+icero;
    xm1d=xs(m1d); xm2d=xs(m2d);
    d=round(abs(xm1d)+abs(xm2d));
    disp(xm1d); disp(xm2d); disp(d);
        
    xs=interp(xs,8);
    msxs=interp(msxs,8);
    lxd=length(msxs);       
    icero=find(abs(xs)==min(abs(xs)));
    fxleft=msxs(1:icero); fxright=msxs(icero+1:lxd);
    m1d=round(mean(find(fxleft==max(fxleft)))); m2d=round(mean(find(fxright==max(fxright))))+icero; %m2d=round(mean(find(fx(icero+1:lxd)==max(fx(icero+1:lxd)))))+icero;
    xm1d=xs(m1d); xm2d=xs(m2d);
    d=round(abs(xm1d)+abs(xm2d)); 
    d_dir(im,it)=d;
    %%
   
    opt=fitoptions('gauss2');
    options.StartPoint = [max(fxleft) xm1d d/2 ...
                          max(fxright) xm2d d/2];
    options.Lower = [   0        xm1d-30  0      0        xm1d-30  0 ]; 
    options.Upper = [max(fxleft) xm1d+30 2*d  max(fxright) xm1d+30 2*d];
    [gfith gof output]=fit(xs',msxs','Gauss2');
    options.StartPoint = [gfith.a1 gfith.b1 gfith.c1 ...
                          gfith.a2 gfith.b2 gfith.c2];
    gof.rsquare                  
    [gfith gof output]=fit(xs0',msx0','Gauss2');
        
    parh{im}(it,1)=gfith.a1; parh{im}(it,2)=gfith.b1; parh{im}(it,3)=gfith.c1/sqrt(2); parh{im}(it,7)=0;
    parh{im}(it,4)=gfith.a2; parh{im}(it,5)=gfith.b2; parh{im}(it,6)=gfith.c2/sqrt(2);
    r2h(im,it)=gof.rsquare;  cmx(im,it)= parh{im}(it,2);
    
%%
    %[para, fval, flag, output]=fminsearch(@(par)Gauss2sum_teo(par,xs,msxs),par0cell{im}(it,:));
    aj=Double_gaussian(parh{im}(it,:),xs);
    hold on;
    plot(xs,aj, 'c');
    %plot(xs,msxs, 'g');
        
    ajleft=aj(1:icero); ajright=aj(icero+1:lxd);
    m1daj=round(mean(find(ajleft==max(ajleft)))); m2daj=round(mean(find(ajright==max(ajright))))+icero; %m2d=round(mean(find(fx(icero+1:lxd)==max(fx(icero+1:lxd)))))+icero;
    xm1daj=xs(m1daj); xm2daj=xs(m2daj);
    daj_max(im,it)=round(abs(xm1daj)+abs(xm2daj));
    d_aj(im,it)=abs(parh{im}(it,2))+abs(abs(parh{im}(it,5)));
    xlim([-1.2*d 1.2*d])
        
    posy=ys(round(mean(find(msys==max(msys))))); tc=sqrt(2/9.8*posy);

    fprintf('g2s=%4.2f\t xm1d=%4.0f \t xm2d=%4.0f \t posy=%4.0f \t tc=%4.1f \n d_dir=%4.0f \t d_aj=%4.0f \t daj_max=%4.0f \t r2=%f \n',...
    sqrt(g2),xm1d,xm2d, posy, tc,d,d_aj(im,it),daj_max(im,it),r2h(im,it));
    axis([xmin xmax 0 1.1]);
    name=['g=' num2str(sqrt(g2), '%2.1f') ' ' 'tof=' legt{it}];
    title(name)
    %%
    [gfitv gof]=fit(ys',msy','Gauss1');
    parv{im}(it,1)=gfitv.a1; parv{im}(it,2)=gfitv.b1; parv{im}(it,3)=gfitv.c1/sqrt(2); parv{im}(it,4)=0;
    r2v(im,it)=gof.rsquare;  cmy(im,it)= parv{im}(it,2);   wy(im,it)=parv{im}(it,3);  
    ajv=Gaussian(parv{im}(it,:),ys);
    figure(2)
    subplot(lmw,ltof,it+(im-1)*ltof); 
    hold on
    plot(ys,msy, 'g',ys,ajv,'c');
    
    fprintf('wy=%4.2f\t r2v=%1.4f \t cmy=%3.2f \n',wy(im,it), r2v(im,it), cmy(im,it));
    axis([cmy(im,it)-4*wy(im,it) cmy(im,it)+4*wy(im,it) 0 1.1]);
    %name=['g=' num2str(sqrt(g2),'%2.1f')  '\Gamma' 'tf=' legt{it} 'r2=' num2str(r2v(im,it),'%1.3f')];
    name=['g=' num2str(sqrt(g2),'%2.1f')  ' '  'r2=' num2str(r2v(im,it),'%1.3f')];
    title(name)
    end
    g2t(im)=sqrt(g2);
end
%%
tabla_teo_aj=[tof' fliplr(d_aj')];
tabla_teo_aj_max=[tof' fliplr(daj_max')];
%%
disp(g2t)
disp(flipud(tabla_teo_aj))
disp(flipud(tabla_teo_aj_max))
disp(rot90(wy,2))
%%
aa=fix(rem(now,1)*1e5);
fsave='1234567890';
f0=fix(clock);
for i=1:5
  if (f0(i+1)<10); 
     fsave(2*i-1)='0'; fsave(2*i)=num2str(f0(i+1));
  else temp=num2str(f0(i+1)); fsave(2*i-1)=temp(1); fsave(2*i)=temp(2);
  end
end
%%
clear data
namesave = [dirtemp1 'bs_max_a' num2str(a) '_' fsave '.mat'];
namef = [dirtemp1 'bs_max_a' num2str(a) '_' fsave ];
%%
%cd a5kz09989\LOWTMP\0p33
%%
%save(namesave);
if salvar=='y';
    save(namesave);
end


#%% From Claude
"""
Beam Splitter Analysis
Translated from MATLAB to Python.

Dependencies: numpy, scipy, matplotlib
Install with: pip install numpy scipy matplotlib
"""

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
dirtemp1 = r'E:\from CL7 HDD\proy t\lab ICN SLM\programas\beam_splitter\a5kz09989\newCI_roc' + os.sep

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

#%% From Gemini
import os
import glob
from datetime import datetime
import numpy as np
import scipy.interpolate as interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ==========================================
# Helper Functions (Replacing MATLAB Toolboxes)
# ==========================================

def smooth(x, window_len=7):
    """Replicates MATLAB's smooth() moving average function."""
    if window_len < 2:
        return x
    # Pad symmetrically to match MATLAB's endpoint behavior
    box = np.ones(window_len) / window_len
    padded = np.pad(x, window_len // 2, mode='edge')
    return np.convolve(padded, box, mode='valid')

def interp(x, factor=8):
    """Replicates MATLAB's interp() low-pass interpolation function."""
    x_old = np.arange(len(x))
    x_new = np.linspace(0, len(x) - 1, len(x) * factor)
    f = interpolate.interp1d(x_old, x, kind='cubic')
    return f(x_new)

def Double_gaussian(x, *params):
    """Custom model function for two Gaussians using standard widths."""
    # MATLAB's c1 parameter is standard width * sqrt(2)
    # inside exp(-((x-b1)/c1)^2). We map it straight from the params array.
    a1, b1, c1, a2, b2, c2, _ = params
    g1 = a1 * np.exp(-((x - b1) / (c1 * np.sqrt(2)))**2)
    g2 = a2 * np.exp(-((x - b2) / (c2 * np.sqrt(2)))**2)
    return g1 + g2

def Gaussian(y, *params):
    """Custom model function for single Gaussian."""
    a1, b1, c1, _ = params
    return a1 * np.exp(-((y - b1) / (c1 * np.sqrt(2)))**2)

def calculate_r2(y_true, y_pred):
    """Calculates Rsquare goodness of fit."""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# ==========================================
# Main Code Initialization
# ==========================================
salvar = 'n'

a = 5
if a == 5:
    mw = [43, 36, 30]
    tof = [18, 15, 13, 11, 10, 8, 7]

dira = str(a)
lmw = len(mw)
ltof = len(tof)

legp = [str(val) for val in mw]
legt = [f"{val}ms" for val in tof]

# System-agnostic path handling
dir1 = r"E:\from CL7 HDD\proy t\lab ICN SLM\programas\beam_splitter\a"
dir2 = dira
dirnp = '0p33'
dir3 = os.path.join(r"kz09989\2012enero\LOWTMP", dirnp, "")

dir4 = legp
dir5 = r"\S_"
dir6 = dir4
dir7 = 'l1p5m3t'
dir8 = legt
dir9 = '_fc0_tg1.DAT'

dirtemp1 = r"E:\from CL7 HDD\proy t\lab ICN SLM\programas\beam_splitter\a5kz09989\newCI_roc"

# Preallocating NumPy matrices
d_dir = np.zeros((lmw, ltof))
d_aj = np.zeros((lmw, ltof))
daj_max = np.zeros((lmw, ltof))
r2h = np.zeros((lmw, ltof))
cmx = np.zeros((lmw, ltof))
r2v = np.zeros((lmw, ltof))
cmy = np.zeros((lmw, ltof))
wy = np.zeros((lmw, ltof))
g2t = np.zeros(lmw)

parh = [np.zeros((ltof, 7)) for _ in range(lmw)]
parv = [np.zeros((ltof, 4)) for _ in range(lmw)]

plt.figure(100, figsize=(15, 10))
plt.figure(2, figsize=(15, 10))

# ==========================================
# File Processing and Analysis Loop
# ==========================================
for im in range(lmw):
    for it in range(ltof):
        
        # 1. Directory Scanning
        search_pattern_fol = os.path.join(dirtemp1, f"*{mw[im]}*")
        fol1_list = glob.glob(search_pattern_fol)
        
        if len(fol1_list) != 1:
            print("VARIOS FOLDERS")
        
        fol1 = os.path.basename(fol1_list[0]) if fol1_list else ""
        
        search_pattern_file = os.path.join(dirtemp1, fol1, f"S_*{mw[im]}*t{tof[it]}*.dat")
        vars_files = glob.glob(search_pattern_file)
        
        if len(vars_files) != 1:
            print("VARIOS FILES ", vars_files)
            input("Press Enter to continue...")  # Mimics MATLAB's pause
            
        nombre = os.path.basename(vars_files[0])
        file_path = os.path.join(dirtemp1, fol1, nombre)
        
        search_pattern_onda = os.path.join(dirtemp1, fol1, "*onda*.dat")
        donda1_files = glob.glob(search_pattern_onda)
        donda = os.path.basename(donda1_files[0])
        dirg2 = os.path.join(dirtemp1, fol1, donda)
        
        # 2. Loading Data
        data = np.loadtxt(file_path)

        with open(dirg2, 'r') as f:
            lines = f.readlines()
            
        # Create a dictionary to mimic the MATLAB struct behavior
        g22_dict = {}
        for line in lines:
            if '=' in line:
                # Split by '=' and strip whitespace
                parts = line.split('=')
                key = parts[0].strip()
                try:
                    # Extract the first numerical value found after the '='
                    value = float(parts[1].strip().split()[0])
                    g22_dict[key] = value
                except (ValueError, IndexError):
                    continue

        # MATLAB: g2 = g22.data(2); 
        # In your file, g2 is explicitly labeled. We can access it directly:
        g2 = g22_dict.get('g2', 0.0)

        # If you strictly want the "second number in the file" to match MATLAB index 2:
        # numeric_values = [v for v in g22_dict.values()]
        # g2 = numeric_values[1]
        
        # # Emulating importdata structural assignment
        # g22_data = np.loadtxt(dirg2)
        # g2 = g22_data[1]  # Index 1 is the 2nd element (MATLAB's index 2)
        
        f = 0.78
        y = data[:, 0] * f + 140
        x = data[:, 1] * f
        ze = data[:, 2] * f
        
        # 3. Coordinate Rotations
        theta = 10 * np.pi / 180
        Ry = np.array([
            [np.cos(theta),  0, np.sin(theta)],
            [0,              1, 0            ],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        M = np.column_stack((x, y, ze))
        Mp = (Ry @ M.T).T
        x, y, ze = Mp[:, 0], Mp[:, 1], Mp[:, 2]
        
        # 4. Histograms and Smoothing
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        lvx, lvy = xmax - xmin, ymax - ymin
        
        tbx = int(np.round(lvx / 9))
        tby = int(np.round(lvy / 9))
        
        msx, bin_edges_x = np.histogram(x, bins=tbx)
        xs = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
        
        msy, bin_edges_y = np.histogram(y, bins=tby)
        ys = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
        
        msx = msx / np.max(msx)
        msy = msy / np.max(msy)
        
        # normfit equivalent: mean and standard deviation
        mux, sigmax = np.mean(x), np.std(x, ddof=1)
        muy, sigmay = np.mean(y), np.std(y, ddof=1)
        
        msys = smooth(smooth(msy, 7), 7)
        msxs = smooth(smooth(msx, 7), 7)
        
        if tof[it] == 9:
            msxs = smooth(msx, 7)
            flagtof = 9
        if tof[it] in [8, 7, 6]:
            msxs = msx.copy()
            flagtof = 1
            
        # Plotting Setup
        plt.figure(100)
        plt.subplot(lmw, ltof, it + im * ltof + 1)
        plt.plot(xs, msx, 'g')
        
        msx0, xs0 = msx.copy(), xs.copy()
        
        # Finding local maxima on split arrays
        icero = np.argmin(np.abs(xs))
        fxleft = msxs[:icero+1]
        fxright = msxs[icero+1:]
        
        m1d = int(np.round(np.mean(np.where(fxleft == np.max(fxleft))[0])))
        m2d = int(np.round(np.mean(np.where(fxright == np.max(fxright))[0]))) + icero + 1
        
        xm1d, xm2d = xs[m1d], xs[m2d]
        d = int(np.round(np.abs(xm1d) + np.abs(xm2d)))
        print(f"xm1d: {xm1d}, xm2d: {xm2d}, d: {d}")
        
        # 5. Interpolation
        xs = interp(xs, 8)
        msxs = interp(msxs, 8)
        
        icero = np.argmin(np.abs(xs))
        fxleft = msxs[:icero+1]
        fxright = msxs[icero+1:]
        
        m1d = int(np.round(np.mean(np.where(fxleft == np.max(fxleft))[0])))
        m2d = int(np.round(np.mean(np.where(fxright == np.max(fxright))[0]))) + icero + 1
        
        xm1d, xm2d = xs[m1d], xs[m2d]
        d = int(np.round(np.abs(xm1d) + np.abs(xm2d)))
        d_dir[im, it] = d
        
        # 6. Curve Fitting (Gauss2)
        # Bounds constraints setup
        lower_bounds = [0, xm1d - 30, 0, 0, xm1d - 30, 0, -np.inf]
        upper_bounds = [np.max(fxleft), xm1d + 30, 2 * d, np.max(fxright), xm1d + 30, 2 * d, np.inf]
        p0 = [np.max(fxleft), xm1d, d / 2, np.max(fxright), xm2d, d / 2, 0]
        
        try:
            popt_h, _ = curve_fit(Double_gaussian, xs0, msx0, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
            aj = Double_gaussian(xs, *popt_h)
            
            # Record fit parameters 
            parh[im][it, 0:6] = popt_h[0:6]
            parh[im][it, 2] /= np.sqrt(2) # Converting scale variables to match target assignment
            parh[im][it, 5] /= np.sqrt(2)
            parh[im][it, 6] = 0
            
            r2h[im, it] = calculate_r2(msx0, Double_gaussian(xs0, *popt_h))
            cmx[im, it] = parh[im][it, 1]
            
            plt.figure(100)
            plt.plot(xs, aj, 'c')
            
            ajleft = aj[:icero+1]
            ajright = aj[icero+1:]
            m1daj = int(np.round(np.mean(np.where(ajleft == np.max(ajleft))[0])))
            m2daj = int(np.round(np.mean(np.where(ajright == np.max(ajright))[0]))) + icero + 1
            xm1daj, xm2daj = xs[m1daj], xs[m2daj]
            
            daj_max[im, it] = np.round(np.abs(xm1daj) + np.abs(xm2daj))
            d_aj[im, it] = np.abs(parh[im][it, 1]) + np.abs(parh[im][it, 4])
            
        except RuntimeError:
            print(f"Fit failed for Horizontal workspace im={im}, it={it}")
            
        plt.xlim([-1.2 * d, 1.2 * d])
        
        posy = ys[int(np.round(np.mean(np.where(msys == np.max(msys))[0])))]
        tc = np.sqrt(2 / 9.8 * posy)
        
        print(f"g2s={np.sqrt(g2):4.2f}\t xm1d={xm1d:4.0f} \t xm2d={xm2d:4.0f} \t posy={posy:4.0f} \t tc={tc:4.1f}")
        print(f" d_dir={d:4.0f} \t d_aj={d_aj[im, it]:4.0f} \t daj_max={daj_max[im, it]:4.0f} \t r2={r2h[im, it]:f}")
        plt.ylim([0, 1.1])
        plt.title(f"g={np.sqrt(g2):2.1f} tof={legt[it]}")
        
        # 7. Single Gaussian Vertical Fitting
        p0_v = [np.max(msy), ys[np.argmax(msy)], np.std(y), 0]
        try:
            popt_v, _ = curve_fit(Gaussian, ys, msy, p0=p0_v, maxfev=10000)
            ajv = Gaussian(ys, *popt_v)
            
            parv[im][it, 0:3] = popt_v[0:3]
            parv[im][it, 2] /= np.sqrt(2)
            parv[im][it, 3] = 0
            
            r2v[im, it] = calculate_r2(msy, ajv)
            cmy[im, it] = parv[im][it, 1]
            wy[im, it] = parv[im][it, 2]
            
            plt.figure(2)
            plt.subplot(lmw, ltof, it + im * ltof + 1)
            plt.plot(ys, msy, 'g', ys, ajv, 'c')
            
        except RuntimeError:
            print(f"Fit failed for Vertical workspace im={im}, it={it}")
            
        print(f"wy={wy[im, it]:4.2f}\t r2v={r2v[im, it]:1.4f} \t cmy={cmy[im, it]:3.2f}")
        plt.xlim([cmy[im, it] - 4 * wy[im, it], cmy[im, it] + 4 * wy[im, it]])
        plt.ylim([0, 1.1])
        plt.title(f"g={np.sqrt(g2):2.1f} r2={r2v[im, it]:1.3f}")
        
    g2t[im] = np.sqrt(g2)

# ==========================================
# Post-Processing & Tables Display
# ==========================================
# Simulating MATLAB's matrix structural modifications
tabla_teo_aj = np.column_stack((tof, np.fliplr(d_aj.T)))
tabla_teo_aj_max = np.column_stack((tof, np.fliplr(daj_max.T)))

print("\n--- Summary Matrices ---")
print("g2t:\n", g2t)
print("flipud(tabla_teo_aj):\n", np.flipud(tabla_teo_aj))
print("flipud(tabla_teo_aj_max):\n", np.flipud(tabla_teo_aj_max))
print("rot90(wy, 2):\n", np.rot90(wy, 2))

# ==========================================
# Saving Logic Block
# ==========================================
now = datetime.now()
fsave = now.strftime("%m%d%H%M%S") # Simulates padding loop dynamically 

namesave = os.path.join(dirtemp1, f"bs_max_a{a}_{fsave}.npz")

if salvar == 'y':
    # Native zipped collection replacement for MATLAB's workspaces .mat data
    np.savez(namesave, d_dir=d_dir, d_aj=d_aj, daj_max=daj_max, r2h=r2h, cmx=cmx, 
             r2v=r2v, cmy=cmy, wy=wy, g2t=g2t, parh=parh, parv=parv)
    print(f"Workspace variables saved successfully to {namesave}")

plt.show()

#%%&
# --- Replace the old g2 loading block with this ---
with open(dirg2, 'r') as f:
    lines = f.readlines()
    
# Create a dictionary to mimic the MATLAB struct behavior
g22_dict = {}
for line in lines:
    if '=' in line:
        # Split by '=' and strip whitespace
        parts = line.split('=')
        key = parts[0].strip()
        try:
            # Extract the first numerical value found after the '='
            value = float(parts[1].strip().split()[0])
            g22_dict[key] = value
        except (ValueError, IndexError):
            continue

# MATLAB: g2 = g22.data(2); 
# In your file, g2 is explicitly labeled. We can access it directly:
g2 = g22_dict.get('g2', 0.0)

# If you strictly want the "second number in the file" to match MATLAB index 2:
# numeric_values = [v for v in g22_dict.values()]
# g2 = numeric_values[1] 

print(f"Loaded g2 value: {g2}")
