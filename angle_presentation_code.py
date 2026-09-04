import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

def normalised_kinematics(data,px,py,pz,nupx,nupy,nupz,pdgid):
    sel = abs(data["mcPDG"]) == pdgid
    # particle kinematics (e or pi)
    px = px[sel]
    py = py[sel]
    pz = pz[sel]

    p = np.sqrt( px**2 + py**2 + pz**2)
    px = px/p
    py = py/p
    pz = pz/p
    pt = np.sqrt( px**2 + py**2 )

    theta = np.arctan2(pt, pz)
    phi   = np.arctan2(py, px)

    # neutrino kinematics
    nupx = nupx[sel]
    nupy = nupy[sel]
    nupz = nupz[sel]

    nup  = np.sqrt( nupx**2 + nupy**2 + nupz**2)
    nupx = nupx/nup
    nupy = nupy/nup
    nupz = nupz/nup
    nupt = np.sqrt( nupx**2 + nupy**2 )

    nutheta = np.arctan2(nupt, nupz)
    nuphi   = np.arctan2(nupy, nupx)

    # relative angles between particle and neutrino
    thetarel = theta - nutheta
    phirel = phi - nuphi
    #thetarel = [a-b for a,b in zip(theta, nutheta) ]
    #phirel = [a-b for a,b in zip(phi, nuphi) ]
        
    # to numpy
    theta_np = ak.to_numpy(ak.flatten(thetarel, axis=None))
    phi_np = ak.to_numpy(ak.flatten(phirel, axis=None))

    # fix the rel phi range to be within 0,2pi 
    phi_np = (phi_np + np.pi) % (2 * np.pi) - np.pi

    return theta_np, phi_np



def rootTreeToDataFrame():
    #insert file name here
    fname_f = "NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"
    #fname_f = "LArRecoND_1_999_Partially_Cheated.root"


    #load file
    file_f = uproot.open(fname_f)

    #get tree from file
    t_f = file_f["LArRecoND"]

    #pull out the branches you need: momenta in x,y,z and PDG code here
    branches = ["mcNuPDG", "mcNuPx", "mcNuPy", "mcNuPz", "mcPx", "mcPy", "mcPz", "mcPDG", 'shwrfitDirX', 'shwrfitDirY', 'shwrfitDirZ', 'dirX', 'dirY', 'dirZ', 'trkfitStartDirX', 'trkfitStartDirY', 'trkfitStartDirZ', 'trkfitEndDirX', 'trkfitEndDirY', 'trkfitEndDirZ']

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

    mc_e_theta, mc_e_phi = normalised_kinematics(data, data["mcPx"], data["mcPy"], data["mcPz"], data["mcNuPx"], data["mcNuPy"], data["mcNuPz"], 11)
    mc_pi_theta, mc_pi_phi = normalised_kinematics(data, data["mcPx"], data["mcPy"], data["mcPz"], data["mcNuPx"], data["mcNuPy"], data["mcNuPz"], 211)

    shower_e_theta, shower_e_phi = normalised_kinematics(data, data["shwrfitDirX"], data["shwrfitDirY"], data["shwrfitDirZ"], data["mcNuPx"], data["mcNuPy"], data["mcNuPz"],11)
    track_pi_theta, track_pi_phi = normalised_kinematics(data, data["trkfitStartDirX"], data["trkfitStartDirY"], data["trkfitStartDirZ"], data["mcNuPx"], data["mcNuPy"], data["mcNuPz"],211)

    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    ax.hist(mc_e_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_mc_color, alpha = 0.3,  facecolor=e_mc_color, label=rf"MC e ($n={len(mc_e_theta)}$)")
    ax.hist(mc_pi_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=pi_mc_color, alpha = 0.3,  facecolor=pi_mc_color, label=rf"MC $\pi$ ($n={len(mc_pi_theta)}$)")
    ax.hist(shower_e_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 0.3,  facecolor=e_color, label=rf"Shower e ($n={len(shower_e_theta)}$)")
    ax.hist(track_pi_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=pi_color, alpha = 0.3,  facecolor=pi_color,label=rf"Track-start $\pi$ ($n={len(track_pi_theta)}$)")

    ax.hist(mc_e_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_mc_color, alpha = 1,  facecolor='none')
    ax.hist(mc_pi_theta,bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=pi_mc_color, alpha = 1,  facecolor='none')
    ax.hist(shower_e_theta,bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 1,  facecolor='none')
    ax.hist(track_pi_theta,bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=pi_color, alpha = 1,  facecolor='none')
    ax.set_xlabel(r"$\Delta\theta$ (rad)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.legend(fontsize=14)
    fig.suptitle( r"$\Delta\theta$ for MC truth and Reco Directions (e Shower, $\pi$ TrackStart) for Fully Cheated", fontsize=16)
    figname = "delta_theta_MC_vs_reco.png"
    print(f"Saving {figname}...")
    fig.savefig(figname, dpi=300)
    plt.close()


    fig, axs = plt.subplots(figsize=(20, 8), nrows=1, ncols=2, layout="constrained")
    ax1 = axs[0]
    ax1.hist(mc_e_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_mc_color, alpha = 0.3,  facecolor=e_mc_color, label=rf"MC e ($n={len(mc_e_theta)}$)")
    ax1.hist(shower_e_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 0.3,  facecolor=e_color, label=rf"Shower e ($n={len(shower_e_theta)}$)")
    ax1.hist(mc_e_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor="black", alpha = 1,  facecolor='none')
    ax1.hist(shower_e_theta,bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 1,  facecolor='none')
    ax1.set_xlabel(r"$\Delta\theta$ (rad)", fontsize=16)
    ax1.set_ylabel("Density", fontsize=16)
    ax1.set_title(r"$\theta$ for MC and Electron (Shower)", fontsize=16)
    ax1.legend(fontsize=14)
    ax2 = axs[1]
    ax2.hist(mc_pi_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=pi_mc_color, alpha = 0.3,  facecolor=pi_mc_color, label=rf"MC $\pi$ ($n={len(mc_pi_theta)}$)")
    ax2.hist(track_pi_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=pi_color, alpha = 0.3,  facecolor=pi_color,label=f"Track-start $\pi$ ($n={len(track_pi_theta)}$)")
    ax2.hist(mc_pi_theta,bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor="black", alpha = 1,  facecolor='none')
    ax2.hist(track_pi_theta,bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=pi_color, alpha = 1,  facecolor='none')
    ax2.set_xlabel(r"$\Delta\theta$ (rad)", fontsize=16)
    ax2.set_ylabel("Density", fontsize=16)
    ax2.set_title(r"$\theta$ for MC and Pion (Track)", fontsize=16)
    ax2.legend(fontsize=14)
    fig.suptitle(r"$\Delta\theta$ for MC truth and Reco Directions for Fully Cheated",fontsize=18)
    figname = "delta_theta_electron_pion_MC_vs_reco.png"
    print(f"Saving {figname}...")
    fig.savefig(figname, dpi=300)
    plt.close()


    for name, (x_branch, y_branch, z_branch, nux_branch, nuy_branch, nuz_branch) in direction_sets.items():

        #for ease of use here, asign arrays as variable names
        nux = data[nux_branch]
        nuy = data[nuy_branch]
        nuz = data[nuz_branch]

        x = data[x_branch]
        y = data[y_branch]
        z = data[z_branch]

        e_theta, e_phi = normalised_kinematics(data, x, y, z, nux, nuy, nuz, 11)
        pi_theta, pi_phi = normalised_kinematics(data, x, y, z, nux, nuy, nuz, 211)     

        # THETA PLOTS
        fig, axs = plt.subplots(figsize=(20, 8), nrows=1, ncols=2, layout='constrained')
        ax1= axs[0]

        n_e_theta = len(e_theta)
        n_pi_theta = len(pi_theta)

        mean_e_theta = np.mean(e_theta)
        mean_pi_theta = np.mean(pi_theta)

        # filled hist
        ax1.hist(e_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 0.3,  facecolor=e_color, label=r'$e, \nu_e$'+f" (n={n_e_theta})")
        ax1.hist(pi_theta, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha= 0.3,facecolor=pi_color, label=r'$\pi, \nu_e$'+f" (n={n_pi_theta})")
        # outline hist
        ax1.hist(e_theta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 1,  facecolor='none')
        ax1.hist(pi_theta, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha=1,facecolor='none')
        #ax1.axvline(mean_e_theta, color=e_color, linestyle="--", linewidth = 2, label= f"Electron mean = {mean_e_theta:.3f} (rad)")
        #ax1.axvline(mean_pi_theta, color=pi_color, linestyle="--", linewidth = 2, label= f"Pion mean = {mean_pi_theta:.3f} (rad)")
        ax1.set_xlabel(r"$\Delta \theta$ (rad)", fontsize=16)
        ax1.set_ylabel("Density", fontsize=16)
        #ax1.set_title(r"$\theta$", fontsize=16)
        ax1.legend(fontsize=16)

        # PHI PLOTS
        ax2 = axs[1]
        n_e_phi = len(e_phi)
        n_pi_phi = len(pi_phi)

        mean_e_phi = np.mean(e_phi)
        mean_pi_phi = np.mean(pi_phi)

        ax2.hist(e_phi, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 0.3,  facecolor=e_color, label=r'$e, \nu_e$'+f" (n={n_e_phi})") 
        ax2.hist(pi_phi, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha= 0.3,facecolor=pi_color, label=r'$\pi, \nu_e$'+f" (n={n_pi_phi})")
        ax2.hist(e_phi, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 1,  facecolor='none')
        ax2.hist(pi_phi, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha=1,facecolor='none')
        #ax2.axvline(mean_e_phi, color=e_color, linestyle="--", linewidth = 2, label= f"Electron mean = {mean_e_phi:.3f} (rad)")
        #ax2.axvline(mean_pi_phi, color=pi_color, linestyle="--", linewidth = 2, label= f"Pion mean = {mean_pi_phi:.3f} (rad)")
        ax2.set_xlabel(r"$\Delta \phi$ (rad)", fontsize=16)
        ax2.set_ylabel("Density", fontsize=16)
        #ax2.set_title(r"$\phi$", fontsize=16)
        ax2.legend(fontsize=16)

        safe_name = name.lower().replace(" ", "_")
        figname = f"fc_angles_{safe_name}.png"
        fig.suptitle(f"{name} angle plots for e and $\\pi$", fontsize=16)
        print(f"Saving {figname}...")
        fig.savefig(figname)
        plt.close()

rootTreeToDataFrame()
