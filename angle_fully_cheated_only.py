import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt


def rootTreeToDataFrame():
    #insert file name here
    fname_f = "NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"

    #load file
    file_f = uproot.open(fname_f)

    #get tree from file
    t_f = file_f["LArRecoND"]

    #pull out the branches you need: momenta in x,y,z and PDG code here
    branches = ["mcPx", "mcPy", "mcPz", "mcPDG", 'shwrfitDirX', 'shwrfitDirY', 'shwrfitDirZ', 'dirX', 'dirY', 'dirZ', 'trkfitStartDirX', 'trkfitStartDirY', 'trkfitStartDirZ', 'trkfitEndDirX', 'trkfitEndDirY', 'trkfitEndDirZ']

    direction_sets = {
        "MC": ("mcPx", "mcPy", "mcPz"), 
        "Shower fit": ("shwrfitDirX", "shwrfitDirY", "shwrfitDirZ"),
        "Track fit": ("dirX", "dirY", "dirZ"),
        "Track start": ("trkfitStartDirX","trkfitStartDirY","trkfitStartDirZ"),"Track end": ("trkfitEndDirX","trkfitEndDirY","trkfitEndDirZ")
    }

    #read data into arrays
    data = t_f.arrays(branches, library="ak")
    e_color="skyblue"
    pi_color="hotpink"

    for name, (x_branch, y_branch, z_branch) in direction_sets.items():

        #for ease of use here, asign arrays as variable names

        x = data[x_branch]
        y = data[y_branch]
        z = data[z_branch]
        pdg = data["mcPDG"]

        #isolate electrons/positrons (with absolute value) and pions(+/-)
        e_only = abs(pdg) == 11
        pi_only = abs(pdg) == 211

        #get out the momenta in x,y,z for electrons and pions 
        e_px = x[e_only]
        e_py = y[e_only]
        e_pz = z[e_only]

        pi_px = x[pi_only]
        pi_py = y[pi_only]
        pi_pz = z[pi_only]

        #get total momentum 
        ep = np.sqrt(e_px**2 + e_py**2 + e_pz**2)
        pi_p = np.sqrt(pi_px**2 + pi_py**2 + pi_pz**2)
        e_r = np.sqrt(e_px * e_px + e_py * e_py)
        pi_r = np.sqrt(pi_px * pi_px + pi_py * pi_py)

        #normalise momentum
        e_xhat = e_px / ep
        e_yhat = e_py / ep
        e_zhat = e_pz / ep
        e_rhat = e_r / ep

        pi_xhat = pi_px / pi_p
        pi_yhat = pi_py / pi_p
        pi_zhat = pi_pz / pi_p
        pi_rhat = pi_r / pi_p

        e_theta = np.arctan2(e_rhat, e_zhat)
        e_phi   = np.arctan2(e_yhat, e_xhat)
        pi_theta = np.arctan2(pi_rhat, pi_zhat)
        pi_phi   = np.arctan2(pi_yhat, pi_xhat)

        e_theta_numpy = ak.to_numpy(ak.flatten(e_theta, axis=None))
        e_phi_numpy = ak.to_numpy(ak.flatten(e_phi, axis=None))
        pi_theta_numpy = ak.to_numpy(ak.flatten(pi_theta, axis=None))
        pi_phi_numpy = ak.to_numpy(ak.flatten(pi_phi, axis=None))    

        fig, axs = plt.subplots(figsize=(20, 8), nrows=1, ncols=2, layout='constrained')
        ax1= axs[0]
        n_e_theta = len(e_theta_numpy)
        n_pi_theta = len(pi_theta_numpy)
        mean_e_theta = np.mean(e_theta_numpy)
        mean_pi_theta = np.mean(pi_theta_numpy)
        ax1.hist(e_theta_numpy, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 0.3,  facecolor=e_color, label=f"Electron (n={n_e_theta})")
        ax1.hist(pi_theta_numpy, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha= 0.3,facecolor=pi_color, label=f"Pion (n={n_pi_theta})")
        ax1.hist(e_theta_numpy, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 1,  facecolor='none')
        ax1.hist(pi_theta_numpy, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha=1,facecolor='none')
        ax1.axvline(mean_e_theta, color=e_color, linestyle="--", linewidth = 2, label= f"Electron mean = {mean_e_theta:.3f} (rad)")
        ax1.axvline(mean_pi_theta, color=pi_color, linestyle="--", linewidth = 2, label= f"Pion mean = {mean_pi_theta:.3f} (rad)")
        ax1.set_xlabel(r"$\theta$ (rad)", fontsize=16)
        ax1.set_ylabel("Density", fontsize=16)
        ax1.set_title(r"$\theta$", fontsize=16)
        ax1.legend(fontsize=16)
    
        ax2 = axs[1]
        n_e_phi = len(e_phi_numpy)
        n_pi_phi = len(pi_phi_numpy)
        mean_e_phi = np.mean(e_phi_numpy)
        mean_pi_phi = np.mean(pi_phi_numpy)
        ax2.hist(e_phi_numpy, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 0.3,  facecolor=e_color, label=f"Electron (n={n_e_phi})")
        ax2.hist(pi_phi_numpy, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha= 0.3,facecolor=pi_color, label=f"Pion (n={n_pi_phi})")
        ax2.hist(e_phi_numpy, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 1,  facecolor='none')
        ax2.hist(pi_phi_numpy, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha=1,facecolor='none')
        ax2.axvline(mean_e_phi, color=e_color, linestyle="--", linewidth = 2, label= f"Electron mean = {mean_e_phi:.3f} (rad)")
        ax2.axvline(mean_pi_phi, color=pi_color, linestyle="--", linewidth = 2, label= f"Pion mean = {mean_pi_theta:.3f} (rad)")
        ax2.set_xlabel(r"$\phi$ (rad)", fontsize=16)
        ax2.set_ylabel("Density", fontsize=16)
        ax2.set_title(r"$\phi$", fontsize=16)
        ax2.legend(fontsize=16)

        safe_name = name.lower().replace(" ", "_")
        figname = f"fc_angles_{safe_name}.png"
        fig.suptitle(f"{name} angle plots for e and $\\pi$", fontsize=16)
        print(f"Saving {figname}...")
        fig.savefig(figname)
        plt.close()

    

rootTreeToDataFrame()
