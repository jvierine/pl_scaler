import numpy as n
import glob
import matplotlib.pyplot as plt
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter

import h5py
import sys
import jcoord
import scipy.interpolate as sint
import stuffr


h=h5py.File("pl_scaling.h5","r")
ant=h["ant"][()]
pf=h["pfs"][()]
rg=h["rgs"][()]
t0=n.array(h["t0"][()],dtype=n.int64)
#print(t0)
h.close()

mh_lat=42.61956156064089
mh_lon=-71.4913065201972

fl = glob.glob("/media/j/f194163e-0385-47e9-8892-182fe5b10ae5/pl/plasma_line/integrated_plasma_line_metadata_hires/*/pl*.h5")

for f in fl:
    h=h5py.File(f,"r")
#    print(h.keys())
    for k in h.keys():
    #    print(h[k].keys())
        t0_this=h[k]["t1"][()]
        ranges=h[k]["ranges"][()]
        freqs=h[k]["freqs"][()]
   #     print(len(freqs))
        neg_idx=n.where(freqs <= 0)[0]
        pos_idx=n.where(freqs >= 0)[0]        
        spec=h[k]["spec"][()]
  #      print(spec.shape)
#        print(len(neg_idx))
 #       print(len(pos_idx))

        if spec.shape[1] > 100:
            spec[:,0:30]=n.nan

        specn=spec[neg_idx[::-1],:]
        specp=spec[pos_idx,:]        
        
        freq_vars_n=n.nanvar(specn,axis=1)
        freq_vars_p=n.nanvar(specp,axis=1)
#        print(len(freq_vars_n))
 #       print(len(freq_vars_p))        
        freq_dec=20

        spec2=n.zeros([len(pos_idx),spec.shape[1]])
        specd=n.zeros([int(len(pos_idx)/freq_dec),spec.shape[1]])        
        w=n.zeros([len(pos_idx),spec.shape[1]])
        
        for i in range(specn.shape[0]):
            if freq_vars_n[i] > 2*freq_vars_p[i]:
                spec2[i,:]+=specp[i,:]
            elif freq_vars_p[i] > 2*freq_vars_n[i]:
                spec2[i,:]+=specn[i,:]
            else:
                spec2[i,:]+=1/(1/freq_vars_n[i] + 1/freq_vars_p[i])*(specn[i,:]/freq_vars_n[i] + specp[i,:]/freq_vars_p[i])

        freq_vars=n.nanvar(spec2,axis=1)
        for i in range(specd.shape[0]):
            spec2[i,:]=spec2[i,:]-n.nanmedian(spec2[i,:])
            for fi in range(freq_dec):
                specd[i,:]+=spec2[i*freq_dec+fi,:]/freq_vars[i*freq_dec+fi]
            specd[i,:]=specd[i,:]/n.sum(1/freq_vars[(i*freq_dec):(i*freq_dec+freq_dec)])

        freqsd=stuffr.decimate(freqs[pos_idx],dec=freq_dec)
        tdiff=n.abs(t0 - t0_this)
        scaling_idx=n.argmin(tdiff)

        #       print(scaling_idx)
        #        print(tdiff[scaling_idx])
        if specd.shape[1]>100:
            plt.pcolormesh(freqsd,ranges,specd.T,vmin=0,vmax=0.05,cmap="plasma")
            if tdiff[scaling_idx] < 120.0:
                print("found dt %1.2f pf %1.2f rg %1.2f"%(tdiff[scaling_idx],pf[scaling_idx]/1e6,rg[scaling_idx]))

                plt.axhline(rg[scaling_idx],color="white",alpha=0.2)
                plt.axvline(pf[scaling_idx]/1e6,color="white",alpha=0.2)            
            plt.title(specd.shape)
            plt.colorbar()
            plt.savefig("%s.png"%(t0_this))
            plt.close("all")


#        print(t0)
    h.close()

