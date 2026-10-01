import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = uproot.open("LArRecoND_1_999_Partially_Cheated.root")
tree = file["LArRecoND"]

branches = [
    "mcPx","mcPy","mcPz","mcPDG","mcNuE",
    "dirX","dirY","dirZ",
    "trkfitStartDirX","trkfitStartDirY","trkfitStartDirZ",
    "trkfitEndDirX","trkfitEndDirY","trkfitEndDirZ",
    "shwrfitDirX","shwrfitDirY","shwrfitDirZ","nUHits"
]

data = tree.arrays(branches,library="ak")

event_index = ak.local_index(data["mcPDG"],axis=0)
event_index = ak.broadcast_arrays(event_index,data["mcPDG"])[0]
nUHits_event = ak.to_numpy(ak.sum(data["nUHits"],axis=1))

DF = pd.DataFrame({
    "event":ak.to_numpy(ak.flatten(event_index)),
    "mcPx":ak.to_numpy(ak.flatten(data["mcPx"])),
    "mcPy":ak.to_numpy(ak.flatten(data["mcPy"])),
    "mcPz":ak.to_numpy(ak.flatten(data["mcPz"])),
    "mcPDG":ak.to_numpy(ak.flatten(data["mcPDG"])),
    "mcNuE":ak.to_numpy(ak.flatten(data["mcNuE"])),
    "int_xhat":ak.to_numpy(ak.flatten(data["dirX"])),
    "int_yhat":ak.to_numpy(ak.flatten(data["dirY"])),
    "int_zhat":ak.to_numpy(ak.flatten(data["dirZ"])),
    "trk_xhat":ak.to_numpy(ak.flatten(data["trkfitStartDirX"])),
    "trk_yhat":ak.to_numpy(ak.flatten(data["trkfitStartDirY"])),
    "trk_zhat":ak.to_numpy(ak.flatten(data["trkfitStartDirZ"])),
    "trke_xhat":ak.to_numpy(ak.flatten(data["trkfitEndDirX"])),
    "trke_yhat":ak.to_numpy(ak.flatten(data["trkfitEndDirY"])),
    "trke_zhat":ak.to_numpy(ak.flatten(data["trkfitEndDirZ"])),
    "shw_xhat":ak.to_numpy(ak.flatten(data["shwrfitDirX"])),
    "shw_yhat":ak.to_numpy(ak.flatten(data["shwrfitDirY"])),
    "shw_zhat":ak.to_numpy(ak.flatten(data["shwrfitDirZ"]))
})

p = np.sqrt(DF["mcPx"]**2 + DF["mcPy"]**2 + DF["mcPz"]**2)
valid_p = p > 0

DF["tru_xhat"] = np.nan
DF["tru_yhat"] = np.nan
DF["tru_zhat"] = np.nan

DF.loc[valid_p,"tru_xhat"] = DF.loc[valid_p,"mcPx"]/p[valid_p]
DF.loc[valid_p,"tru_yhat"] = DF.loc[valid_p,"mcPy"]/p[valid_p]
DF.loc[valid_p,"tru_zhat"] = DF.loc[valid_p,"mcPz"]/p[valid_p]

DF_electrons = DF[np.abs(DF["mcPDG"]) == 11].copy()
DF_pions = DF[np.abs(DF["mcPDG"]) == 211].copy()

electron_diff = DF_electrons["shw_xhat"] - DF_electrons["tru_xhat"]
pion_diff = DF_pions["trk_xhat"] - DF_pions["tru_xhat"]

electron_diagonal_cut = np.abs(electron_diff) > 0.1
electron_column_cut = (
    (np.abs(electron_diff) > 0.1) &
    (DF_electrons["tru_xhat"].between(-0.25,0.25))
)

pion_diagonal_cut = np.abs(pion_diff) > 0.1
pion_column_cut = (
    (np.abs(pion_diff) > 0.1) &
    (DF_pions["tru_xhat"].between(-0.25,0.25))
)

DF_electron_diagonal = DF_electrons[electron_diagonal_cut]
DF_electron_column = DF_electrons[electron_column_cut]

DF_pion_diagonal = DF_pions[pion_diagonal_cut]
DF_pion_column = DF_pions[pion_column_cut]

all_electron_events = np.unique(DF_electrons["event"].to_numpy())
electron_diagonal_events = np.unique(DF_electron_diagonal["event"].to_numpy())
electron_column_events = np.unique(DF_electron_column["event"].to_numpy())

all_pion_events = np.unique(DF_pions["event"].to_numpy())
pion_diagonal_events = np.unique(DF_pion_diagonal["event"].to_numpy())
pion_column_events = np.unique(DF_pion_column["event"].to_numpy())

nUHits_all_electrons = nUHits_event[all_electron_events]
electron_diagonal_hits = nUHits_event[electron_diagonal_events]
electron_column_hits = nUHits_event[electron_column_events]

nUHits_all_pions = nUHits_event[all_pion_events]
pion_diagonal_hits = nUHits_event[pion_diagonal_events]
pion_column_hits = nUHits_event[pion_column_events]

print("Electrons:")
print("All electron events:",len(all_electron_events))
print("Diagonal cut events:",len(electron_diagonal_events))
print("Column cut events:",len(electron_column_events))

print("\nPions:")
print("All pion events:",len(all_pion_events))
print("Diagonal cut events:",len(pion_diagonal_events))
print("Column cut events:",len(pion_column_events))

plt.figure(figsize=(8,6))
plt.hist(nUHits_all_electrons,bins=50,histtype="step",linewidth=2,label="All electrons")
plt.hist(electron_diagonal_hits,bins=50,histtype="step",linewidth=2,label="Electrons: diagonal cut")
plt.hist(electron_column_hits,bins=50,histtype="step",linewidth=2,label="Electrons: column cut")
plt.xlabel("Number of U-plane hits per event")
plt.ylabel("Events")
plt.title("U-plane hits per event for electrons")
plt.legend()
plt.tight_layout()
plt.savefig("electron_U_hits_per_event_comparison.png",dpi=150)
plt.close()

plt.figure(figsize=(8,6))
plt.hist(nUHits_all_pions,bins=50,histtype="step",linewidth=2,label="All pions")
plt.hist(pion_diagonal_hits,bins=50,histtype="step",linewidth=2,label="Pions: diagonal cut")
plt.hist(pion_column_hits,bins=50,histtype="step",linewidth=2,label="Pions: column cut")
plt.xlabel("Number of U-plane hits per event")
plt.ylabel("Events")
plt.title("U-plane hits per event for pions")
plt.legend()
plt.tight_layout()
plt.savefig("pion_U_hits_per_event_comparison.png",dpi=150)
plt.close()

print("\nPlots saved successfully.")
