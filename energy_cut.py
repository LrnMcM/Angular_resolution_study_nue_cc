import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

def normalised_kinematics(data, px, py, pz, nupx, nupy, nupz, pdgid):
    sel = abs(data["mcPDG"]) == pdgid
    px = px[sel]
    py = py[sel]
    pz = pz[sel]
    energy = data["mcEnergy"][sel]
    p = np.sqrt(px**2 + py**2 + pz**2)
    px = px / p
    py = py / p
    pz = pz / p
    pt = np.sqrt(px**2 + py**2)
    theta = np.arctan2(pt, pz)
    phi = np.arctan2(py, px)
    nupx = nupx[sel]
    nupy = nupy[sel]
    nupz = nupz[sel]
    nup = np.sqrt(nupx**2 + nupy**2 + nupz**2)
    nupx = nupx / nup
    nupy = nupy / nup
    nupz = nupz / nup
    nupt = np.sqrt(nupx**2 + nupy**2)
    nutheta = np.arctan2(nupt, nupz)
    nuphi = np.arctan2(nupy, nupx)
    thetarel = theta - nutheta
    phirel = phi - nuphi
    theta_np = ak.to_numpy(ak.flatten(thetarel, axis=None))
    phi_np = ak.to_numpy(ak.flatten(phirel, axis=None))
    energy_np = ak.to_numpy(ak.flatten(energy, axis=None))
    phi_np = (phi_np + np.pi) % (2 * np.pi) - np.pi

    return theta_np, phi_np, energy_np



def rootTreeToDataFrame():
    #insert file name here
    fname_f = "NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"
    #fname_f = "LArRecoND_1_999_Partially_Cheated.root"


    #load file
    file_f = uproot.open(fname_f)

    #get tree from file
    t_f = file_f["LArRecoND"]

    #pull out the branches you need: momenta in x,y,z and PDG code here
    branches = ["mcNuPDG", "mcNuPx", "mcNuPy", "mcEnergy", "mcNuPz", "mcPx", "mcPy", "mcPz", "mcPDG", 'shwrfitDirX', 'shwrfitDirY', 'shwrfitDirZ', 'dirX', 'dirY', 'dirZ', 'trkfitStartDirX', 'trkfitStartDirY', 'trkfitStartDirZ', 'trkfitEndDirX', 'trkfitEndDirY', 'trkfitEndDirZ']

    direction_sets = {
        "MC": ("mcPx", "mcPy", "mcPz","mcNuPx", "mcNuPy", "mcNuPz"), 
        "Shower fit": ("shwrfitDirX", "shwrfitDirY", "shwrfitDirZ","mcNuPx", "mcNuPy", "mcNuPz"),
        "Interface": ("dirX", "dirY", "dirZ","mcNuPx", "mcNuPy", "mcNuPz"),
        "Track start": ("trkfitStartDirX","trkfitStartDirY","trkfitStartDirZ","mcNuPx", "mcNuPy", "mcNuPz"),
        "Track end": ("trkfitEndDirX","trkfitEndDirY","trkfitEndDirZ","mcNuPx", "mcNuPy", "mcNuPz")
    }

    #read data into arrays
    data = t_f.arrays(branches, library="ak")
    nue_color="gold"
    e_color="skyblue"
    e_mc_color="cornflowerblue"
    pi_color="hotpink"
    pi_mc_color="violet"

    # TBrowser shows some mcNuPDG=0 entries: chuck them out.
    data = data[(np.abs(data["mcNuPDG"]) == 12)]

    mc_e_theta, mc_e_phi, mc_e_energy = normalised_kinematics(data,data["mcPx"],data["mcPy"],data["mcPz"],data["mcNuPx"],data["mcNuPy"],data["mcNuPz"],11)
    mc_pi_theta, mc_pi_phi, mc_pi_energy = normalised_kinematics(data, data["mcPx"], data["mcPy"], data["mcPz"], data["mcNuPx"], data["mcNuPy"], data["mcNuPz"], 211)

    shower_e_theta, shower_e_phi,_ = normalised_kinematics(data, data["shwrfitDirX"], data["shwrfitDirY"], data["shwrfitDirZ"], data["mcNuPx"], data["mcNuPy"], data["mcNuPz"],11)
    track_pi_theta, track_pi_phi ,_= normalised_kinematics(data, data["trkfitStartDirX"], data["trkfitStartDirY"], data["trkfitStartDirZ"], data["mcNuPx"], data["mcNuPy"], data["mcNuPz"],211)

    angle_mask = (shower_e_theta >= 1.0) & (shower_e_theta <= 1.5)
    selected_energy = mc_e_energy[angle_mask]
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")

    ax.hist(selected_energy,bins=75,histtype="stepfilled",alpha=0.3,edgecolor=e_mc_color,facecolor=e_mc_color,linewidth=2,label=rf"MC electron ($1 < \Delta \theta < 1.5$ rad, $n={len(selected_energy)}$)")
    ax.hist(selected_energy,bins=75,histtype="step",linewidth=2,edgecolor=e_mc_color)
    ax.set_xlabel(r"Electron $E$ (GeV)", fontsize=16)
    ax.set_ylabel("Number of events", fontsize=16)
    ax.set_title(r"MC Electron Energy for $1 < \Delta \theta < 1.5$ rad",fontsize=18)
    ax.legend(fontsize=14)
    figname = "mc_electron_energy_theta_1_to_1p5.png"
    print(f"Saving {figname}...")
    fig.savefig(figname, dpi=300)
    plt.close()


rootTreeToDataFrame()
