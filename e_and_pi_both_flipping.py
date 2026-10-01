import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file=uproot.open("LArRecoND_1_999_Partially_Cheated.root")
tree=file["LArRecoND"]
branches=["mcPx","mcPy","mcPz","mcPDG","mcNuE","dirX","dirY","dirZ","trkfitStartDirX","trkfitStartDirY","trkfitStartDirZ","trkfitEndDirX","trkfitEndDirY","trkfitEndDirZ","shwrfitDirX","shwrfitDirY","shwrfitDirZ"]
data=tree.arrays(branches,library="ak")

e_color="skyblue"
pi_color="hotpink"

event_id = ak.flatten(ak.broadcast_arrays(data["mcPx"],ak.local_index(data["mcPx"], axis=0))[1])

DF = pd.DataFrame({
    "event_id": ak.to_numpy(event_id),
    "mcPx": ak.to_numpy(ak.flatten(data["mcPx"])),
    "mcPy": ak.to_numpy(ak.flatten(data["mcPy"])),
    "mcPz": ak.to_numpy(ak.flatten(data["mcPz"])),
    "mcPDG": ak.to_numpy(ak.flatten(data["mcPDG"])),
    "mcNuE": ak.to_numpy(ak.flatten(data["mcNuE"])),
    "int_xhat": ak.to_numpy(ak.flatten(data["dirX"])),
    "int_yhat": ak.to_numpy(ak.flatten(data["dirY"])),
    "int_zhat": ak.to_numpy(ak.flatten(data["dirZ"])),
    "trk_xhat": ak.to_numpy(ak.flatten(data["trkfitStartDirX"])),
    "trk_yhat": ak.to_numpy(ak.flatten(data["trkfitStartDirY"])),
    "trk_zhat": ak.to_numpy(ak.flatten(data["trkfitStartDirZ"])),
    "trke_xhat": ak.to_numpy(ak.flatten(data["trkfitEndDirX"])),
    "trke_yhat": ak.to_numpy(ak.flatten(data["trkefitEndDirY"])) if False else ak.to_numpy(ak.flatten(data["trkfitEndDirY"])),
    "trke_zhat": ak.to_numpy(ak.flatten(data["trkfitEndDirZ"])),
    "shw_xhat": ak.to_numpy(ak.flatten(data["shwrfitDirX"])),
    "shw_yhat": ak.to_numpy(ak.flatten(data["shwrfitDirY"])),
    "shw_zhat": ak.to_numpy(ak.flatten(data["shwrfitDirZ"]))
})

p=np.sqrt(DF["mcPx"]**2+DF["mcPy"]**2+DF["mcPz"]**2)
DF["tru_xhat"]=DF["mcPx"]/p
DF["tru_yhat"]=DF["mcPy"]/p
DF["tru_zhat"]=DF["mcPz"]/p

DF_electrons = DF[np.abs(DF["mcPDG"]) == 11]
DF_pions = DF[np.abs(DF["mcPDG"]) == 211]

electron_diff = (DF_electrons["shw_xhat"] - DF_electrons["tru_xhat"])
electron_only_diagonal = (DF_electrons["tru_xhat"] + DF_electrons["shw_xhat"])
electron_cut = ((np.abs(electron_only_diagonal) <= 0.15)&(np.abs(electron_diff) >= 0.1))

DF_electron_cut = DF_electrons[electron_cut]
selected_event_ids = DF_electron_cut["event_id"].unique()
DF_pions_same_events = DF_pions[DF_pions["event_id"].isin(selected_event_ids)]

plt.figure(figsize=(8,6))
plt.scatter(DF_electrons["tru_xhat"],DF_electrons["shw_xhat"],s=20,color= e_color, alpha=0.5)
x=np.linspace(-1,1,100)
plt.plot(x,x,"k--",label=r"$p_x^{reco}=p_x^{true}$")
plt.xlabel(r"Truth $\hat{p}_x$")
plt.ylabel(r"Shower $\hat{p}_x$")
plt.title(r"Electron shower $\hat{p}_x$ vs Truth $\hat{p}_x$")
plt.legend()
plt.tight_layout()
plt.savefig("all_e_shower_px.png")
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(DF_electron_cut["tru_xhat"],DF_electron_cut["shw_xhat"],s=20,color= e_color,alpha=0.5)
x=np.linspace(-1,1,100)
plt.plot(x,x,"k--",label=r"$p_x^{reco}=p_x^{true}$")
plt.plot(x,x+0.1,"r--",label=r"$p_x^{reco}-p_x^{true}=0.1$")
plt.plot(x,x-0.1,"b--",label=r"$p_x^{reco}-p_x^{true}=-0.1$")
plt.xlabel(r"Truth $\hat{p}_x$")
plt.ylabel(r"Shower $\hat{p}_x$")
plt.title(r"Electron shower $\hat{p}_x$ vs Truth $\hat{p}_x$ after cut")
plt.legend()
plt.tight_layout()
plt.savefig("off_diagonal_electrons.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(DF_electron_cut["tru_yhat"],DF_electron_cut["shw_yhat"],color= e_color,s=20,alpha=0.5)
y = np.linspace(-1, 1, 100)
plt.plot(y, y, "k--",label=r"$p_y^{reco}=p_y^{true}$")
plt.plot(y, y + 0.1, "r--",label=r"$p_y^{reco}-p_y^{true}=0.1$")
plt.plot(y, y - 0.1, "b--",label=r"$p_y^{reco}-p_y^{true}=-0.1$")
plt.xlabel(r"Truth $\hat{p}_y$")
plt.ylabel(r"Shower $\hat{p}_y$")
plt.title(r"Electron shower $\hat{p}_y$ for electrons passing both cuts")
plt.legend()
plt.tight_layout()
plt.savefig("selected_electrons_yhat.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(DF_pions_same_events["tru_xhat"],DF_pions_same_events["trk_xhat"],color= pi_color,s=20,alpha=0.5)
x = np.linspace(-1, 1, 100)
plt.plot(x, x, "k--",label=r"$p_x^{reco}=p_x^{true}$")
plt.xlabel(r"Truth $\hat{p}_x$")
plt.ylabel(r"Track $\hat{p}_x$")
plt.title(r"Pions in events containing electrons passing both cuts")
plt.legend()
plt.tight_layout()
plt.savefig("pions_in_selected_electron_events.png")
plt.close()


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(DF_electron_cut["tru_xhat"], DF_electron_cut["shw_xhat"],color= e_color, s=20, alpha=0.5)
x = np.linspace(-1, 1, 100)
axes[0].plot(x, x, "k--", label=r"$p_x^{reco}=p_x^{true}$")
axes[0].plot(x, x + 0.1, "r--", label=r"$p_x^{reco}-p_x^{true}=0.1$")
axes[0].plot(x, x - 0.1, "b--", label=r"$p_x^{reco}-p_x^{true}=-0.1$")
axes[0].set_xlabel(r"Truth $\hat{p}_x$")
axes[0].set_ylabel(r"Shower $\hat{p}_x$")
axes[0].set_title(r"Selected electrons: $\hat{p}_x$")
axes[0].legend()

axes[1].scatter(DF_electron_cut["tru_yhat"], DF_electron_cut["shw_yhat"], s=20, color= e_color, alpha=0.5)
y = np.linspace(-1, 1, 100)
#axes[1].plot(y, y, "k--", label=r"$p_y^{reco}=p_y^{true}$")
#axes[1].plot(y, y + 0.1, "r--", label=r"$p_y^{reco}-p_y^{true}=0.1$")
#axes[1].plot(y, y - 0.1, "b--", label=r"$p_y^{reco}-p_y^{true}=-0.1$")
axes[1].set_xlabel(r"Truth $\hat{p}_y$")
axes[1].set_ylabel(r"Shower $\hat{p}_y$")
axes[1].set_title(r"Selected electrons: $\hat{p}_y$")
#axes[1].legend()

axes[2].scatter(DF_pions_same_events["tru_xhat"], DF_pions_same_events["trk_xhat"], color= pi_color, s=20, alpha=0.5)
x = np.linspace(-1, 1, 100)
#axes[2].plot(x, x, "k--", label=r"$p_x^{reco}=p_x^{true}$")
axes[2].set_xlabel(r"Truth $\hat{p}_x$")
axes[2].set_ylabel(r"Track $\hat{p}_x$")
axes[2].set_title(r"Pions in selected electron events")
#axes[2].legend()

for ax in axes:
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("combined_electron_pion_plots.png", dpi=300, bbox_inches="tight")
plt.close()
