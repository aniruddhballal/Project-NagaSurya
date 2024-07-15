import numpy as np
import sunpy.map
import matplotlib.pyplot as plt
from scipy.special import sph_harm

def calcalm(d,lm):
    lat, long = d.shape
    alm = np.zeros((lm+1, 2*lm+1), dtype = np.complex128)
    p=np.linspace(0,np.pi,long)
    t=np.linspace(0,np.pi,lat)
    pgrid, tgrid = np.meshgrid(p,t)
    for l in range (lm+1):
        for m  in range(-l,l+1):
            ylm = sph_harm(m,l,pgrid,tgrid)
            val = d*ylm.conj()*np.sin(tgrid)
            alm[l,m+lm] = np.sum(val)*(2*np.pi/long)*(np.pi/lat)
    return alm

fitspath = "C:/Users/aniru/OneDrive/Desktop/we goin solar/hmi.Synoptic_Mr_small.2110.fits"
smap = sunpy.map.Map(fitspath)

f = plt.figure(figsize=(12,5))
a = plt.subplot(projection=smap)
i = smap.plot(axes=a)
a.coords[0].set_axislabel("CR Long [deg]")
a.coords[1].set_axislabel("Lat [deg]")
a.coords.grid(color = 'black', alpha = 0.6, linestyle = 'dotted', linewidth = 0.5)
cb = plt.colorbar(i, fraction = 0.019, pad = 0.1)
cb.set_label(f"Radial mag field [{smap.unit}]")
a.set_ylim(bottom = 0)
a.set_title(f"{smap.meta['content']}, \nCR rotation {smap.meta['CAR_ROT']}")
plt.show()


d = smap.data
d = np.nan_to_num(d, nan = 0.0)

lm = 5
alm = calcalm(d,lm)

txtpath = "trialrename.txt"

with open(txtpath, "w") as file:
    for l in range(lm + 1):
        for m in range(-l,l+1):
            alm2 = alm[l,m+lm]
            if alm2.imag>0:
                file.write(f"{l}\t{m}\t{alm2.real}+{alm2.imag}i\n")
            elif alm2.imag<0:
                file.write(f"{l}\t{m}\t{alm2.real}{alm2.imag}i\n")
            else:
                file.write(f"{l}\t{m}\t{alm2.real}\n")
