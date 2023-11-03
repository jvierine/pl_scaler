import numpy as n
import glob
import matplotlib.pyplot as plt
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter

import h5py
import sys
import jcoord
import scipy.interpolate as sint

mh_lat=42.61956156064089
mh_lon=-71.4913065201972

def az_el_height_to_range(az,el,hgt):
    ranges=n.linspace(100e3,2000e3,num=600)
    hgts=[]
    for r in ranges:
        llh=jcoord.az_el_r2geodetic(mh_lat,mh_lon,0,az,el,r)
        hgt0=llh[2]
        hgts.append(hgt0)
    hfun=sint.interp1d(hgts,ranges,kind="cubic")
    print("az %1.2f el %1.2f h %1.2f"%(az,el,hgt))    
    radar_range=hfun(hgt)
    print("az %1.2f el %1.2f h %1.2f r %1.2f"%(az,el,hgt,radar_range))
    
    return(radar_range)

txtfl = glob.glob("/media/j/f194163e-0385-47e9-8892-182fe5b10ae5/pl/plasma_line/pl_txt_files/pl*.txt")

fl = glob.glob("/media/j/f194163e-0385-47e9-8892-182fe5b10ae5/pl/plasma_line/integrated_plasma_line_metadata_hires/*/pl*.h5")

t0s=[]
pfs=[]
rgs=[]
ants=[]

if False:
    for f in fl:
        h=h5py.File(f,"r")
    #    print(h.keys())
        for k in h.keys():
            print(h[k].keys())
        h.close()

for txtf in txtfl:
    print(txtf)
    lines=open(txtf,"r").readlines()
    
    for l in lines:
        ll=l.strip().split(" ")
        
        if len(ll) == 6:
            t0=float(ll[0])
            pf=8.98*n.sqrt(float(ll[1]))
            hgt=float(ll[2])
            az=float(ll[3])
            el=float(ll[4])            
            ant=ll[5]
            t0s.append(t0)
            pfs.append(pf)

            if hgt == 0.0:
                hgt=n.nan
            print(ant)
            if ant == "ZENITH" or ant == "zenith":
                ants.append(0)
                rgs.append(hgt)
            else:
                ants.append(1)
                rgs.append(az_el_height_to_range(az,el,hgt*1e3)/1e3)

t0s=n.array(t0s)
pfs=n.array(pfs)
rgs=n.array(rgs)
ants=n.array(ants)

ho=h5py.File("pl_scaling.h5","w")
ho["t0"]=t0s
ho["pfs"]=pfs
ho["rgs"]=rgs
ho["ant"]=ants
ho.close()

zenith_idx=n.where(ants == 0)[0]
misa_idx=n.where(ants == 1)[0]

plt.plot(t0s[zenith_idx],pfs[zenith_idx]/1e6,".")
plt.plot(t0s[misa_idx],pfs[misa_idx]/1e6,".")
plt.show()

plt.plot(t0s[zenith_idx],rgs[zenith_idx],".")
plt.plot(t0s[misa_idx],rgs[misa_idx],".")
plt.show()
