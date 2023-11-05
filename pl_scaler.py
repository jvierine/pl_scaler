import numpy as n
import glob
import matplotlib.pyplot as plt
from digital_rf import DigitalRFReader, DigitalMetadataReader, DigitalMetadataWriter

import h5py
import sys
import jcoord
import scipy.interpolate as sint
import stuffr
import os

os.system("export OMP_NUM_THREADS=1")
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI
comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

def normalize_pl(ranges, freqs, spec):

    neg_idx=n.where(freqs <= 0)[0]
    pos_idx=n.where(freqs >= 0)[0]        

    speco=n.copy(spec)

    if spec.shape[1] > 100:
        spec[:,0:30]=n.nan

    specn=spec[neg_idx[::-1],:]
    specp=spec[pos_idx,:]        

    freq_vars_n=n.nanvar(specn,axis=1)
    freq_vars_p=n.nanvar(specp,axis=1)

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

    if specd.shape[1]>100:
        return(ranges[30:len(ranges)], freqsd, specd[:,30:len(ranges)].T)
    else:
        return([],[],[])

def scale_pl(spec,t0,scaler="juha"):
    global x,y
    x=n.nan
    y=n.nan
    fig, ax = plt.subplots(constrained_layout=True,figsize=(16*1.5,9*1.5))

    def redraw():
        ax.clear()
        ax.pcolormesh(spec,vmin=0,vmax=0.05,cmap="plasma")
        ax.set_title("1) set hmf & fof2 9) save 0) skip")
        
        ax.axvline(x,color="red")
        ax.axhline(y,color="red")        
        fig.canvas.draw()
    redraw()
        
    def press(event):
        global x,y
        
        print("press %f %f"%(x,y))
        sys.stdout.flush()
        if event.key == '1':
            x, y = event.xdata, event.ydata
            redraw()
            
        if event.key == '9':
            ofname="%d.h5"%(t0)
            print("saving %s"%(ofname))
            hout=h5py.File(ofname,"w")
            hout["x"]=x
            hout["y"]=y
            hout["spec"]=spec
            hout["t0"]=t0
            hout["scaler"]=scaler
            hout.close()
            plt.savefig("%d.png"%(t0))
            plt.close()
        if event.key == '0':            
            plt.close()
            
    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()


def label_files():
    fl = glob.glob("/media/j/f194163e-0385-47e9-8892-182fe5b10ae5/pl/plasma_line/integrated_plasma_line_metadata_hires/*/pl*.h5")
    n.random.shuffle(fl)
#    fl.sort()
    for f in fl:
        h=h5py.File(f,"r")
        print(f)
        for k in h.keys():

            t0_this=h[k]["t1"][()]
            ranges=h[k]["ranges"][()]
            freqs=h[k]["freqs"][()]
            spec=h[k]["spec"][()]
            
            ranges_d, freqs_d, spec_d=normalize_pl(ranges, freqs, spec)
            if len(ranges_d) == 258 and len(freqs_d) == 300:
                print("high res plasma-line")
                if os.path.exists("%d.h5"%(t0_this)):
                    print("already scaled, ignoring")
                else:
                    scale_pl(spec_d,t0_this)
            else:
                print("low res plasma-line. ignoring")
#                print(spec_d.shape)

        h.close()


if __name__ == "__main__":
    label_files()
