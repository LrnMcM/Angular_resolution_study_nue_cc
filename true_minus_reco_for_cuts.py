import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path="LArRecoND_1_999_Partially_Cheated.root"
tree=uproot.open(file_path)["LArRecoND"]

branches=[
    "mcPx","mcPy","mcPz","mcPDG","mcNuE",
    "dirX","dirY","dirZ",
    "trkfitStartDirX","trkfitStartDirY","trkfitStartDirZ",
    "trkfitEndDirX","trkfitEndDirY","trkfitEndDirZ",
    "shwrfitDirX","shwrfitDirY","shwrfitDirZ"
]

data=tree.arrays(branches,library="ak")

event_id=ak.flatten(ak.broadcast_arrays(data["mcPx"],ak.local_index(data["mcPx"],axis=0))[1])

DF=pd.DataFrame({
    "event_id":ak.to_numpy(event_id),
    "mcPx":ak.to_numpy(ak.flatten(data["mcPx"])),
    "mcPy":ak.to_numpy(ak.flatten(data["mcPy"])),
    "mcPz":ak.to_numpy(ak.flatten(data["mcPz"])),
    "mcPDG":ak.to_numpy(ak.flatten(data["mcPDG"])),
    "trk_xhat":ak.to_numpy(ak.flatten(data["trkfitStartDirX"])),
    "trk_yhat":ak.to_numpy(ak.flatten(data["trkfitStartDirY"])),
    "trk_zhat":ak.to_numpy(ak.flatten(data["trkfitStartDirZ"])),
    "shw_xhat":ak.to_numpy(ak.flatten(data["shwrfitDirX"])),
    "shw_yhat":ak.to_numpy(ak.flatten(data["shwrfitDirY"])),
    "shw_zhat":ak.to_numpy(ak.flatten(data["shwrfitDirZ"]))
})

p=np.sqrt(DF["mcPx"]**2+DF["mcPy"]**2+DF["mcPz"]**2)
DF["tru_xhat"]=DF["mcPx"]/p
DF["tru_yhat"]=DF["mcPy"]/p
DF["tru_zhat"]=DF["mcPz"]/p

DF_electrons=DF[np.abs(DF["mcPDG"])==11]

electron_diff=DF_electrons["shw_xhat"]-DF_electrons["tru_xhat"]
electron_flipped=DF_electrons["tru_xhat"]+DF_electrons["shw_xhat"]

electron_cut=(np.abs(electron_flipped)<=0.15)&(np.abs(electron_diff)>=0.1)
electron_column_cut=(np.abs(electron_diff)>0.1)&(DF_electrons["tru_xhat"].between(-0.25,0.25))

DF_electron_cut=DF_electrons[electron_cut]
DF_electron_column_cut=DF_electrons[electron_column_cut]

plt.figure(figsize=(8,6))
plt.scatter(DF_electrons["tru_xhat"],DF_electrons["shw_xhat"],s=8,alpha=0.25,label="All electrons")
plt.scatter(DF_electron_cut["tru_xhat"],DF_electron_cut["shw_xhat"],s=12,alpha=0.8,label="Electrons passing flipped cut")
plt.xlabel(r"True $\hat{p}_x$")
plt.ylabel(r"Shower reconstructed $\hat{p}_x$")
plt.legend()
plt.tight_layout()
plt.savefig("electrons_passing_flipped_cut.png",dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(DF_electrons["tru_xhat"],DF_electrons["shw_xhat"],s=8,alpha=0.25,label="All electrons")
plt.scatter(DF_electron_column_cut["tru_xhat"],DF_electron_column_cut["shw_xhat"],s=12,alpha=0.8,label="Electrons passing column cut")
plt.axvline(-0.25,linestyle="--",linewidth=1)
plt.axvline(0.25,linestyle="--",linewidth=1)
plt.xlabel(r"True $\hat{p}_x$")
plt.ylabel(r"Shower reconstructed $\hat{p}_x$")
plt.legend()
plt.tight_layout()
plt.savefig("electrons_passing_column_cut.png",dpi=300)
plt.close()

electron_excluded=(np.abs(DF_electrons["tru_xhat"]-DF_electrons["shw_xhat"])<0.1)&(DF_electrons["tru_xhat"].between(-0.25,0.25))
DF_electron_excluded=DF_electrons[electron_excluded]

plt.figure(figsize=(8,6))
plt.scatter(DF_electrons["tru_xhat"],DF_electrons["shw_xhat"],s=8,alpha=0.25,label="All electrons")
plt.scatter(DF_electron_excluded["tru_xhat"],DF_electron_excluded["shw_xhat"],s=12,alpha=0.8,label="Region to exclude")
plt.axvline(-0.25,linestyle="--",linewidth=1)
plt.axvline(0.25,linestyle="--",linewidth=1)
plt.axline((0,-0.1),slope=1,linestyle="--",linewidth=1,label=r"$p_{x,true}-p_{x,reco}=0.1$")
plt.axline((0,0.1),slope=1,linestyle="--",linewidth=1,label=r"$p_{x,true}-p_{x,reco}=0.1$")
plt.xlabel(r"True $\hat{p}_x$")
plt.ylabel(r"Shower reconstructed $\hat{p}_x$")
plt.legend()
plt.tight_layout()
plt.savefig("electrons_region_to_exclude.png",dpi=300)
plt.close()

DF_electrons_remaining=DF_electrons[~electron_excluded]

px_diff_all=DF_electrons_remaining["tru_xhat"]-DF_electrons_remaining["shw_xhat"]
px_diff_diagonal=(DF_electrons["tru_xhat"]-DF_electrons["shw_xhat"])[electron_cut]
px_diff_column=(DF_electrons["tru_xhat"]-DF_electrons["shw_xhat"])[electron_column_cut]

bins=np.linspace(-1,1,101)

plt.figure(figsize=(8,6))
plt.hist(px_diff_all,bins=bins,histtype="step",linewidth=2,label="All excluding selected region")
plt.hist(px_diff_diagonal,bins=bins,histtype="step",linewidth=2,label="Diagonal cut")
plt.hist(px_diff_column,bins=bins,histtype="step",linewidth=2,label="Column cut")
plt.axvline(0,linestyle="--",linewidth=1)
plt.xlabel(r"True $\hat{p}_x$ - Reconstructed $\hat{p}_x$")
plt.ylabel("Number of electrons")
plt.legend()
plt.tight_layout()
plt.savefig("electron_px_true_minus_reco_cuts.png",dpi=300)
plt.close()
