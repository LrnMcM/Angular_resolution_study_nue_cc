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
    branches = ["mcPx", "mcPy", "mcPz", "mcPDG", "mcNuPx", "mcNuPy", "mcNuPz"]

    #read data into arrays
    data = t_f.arrays(branches, library="ak")
    e_color="skyblue"
    pi_color="hotpink"

    pdg = data["mcPDG"]
    mc_e_only = abs(pdg) == 11

    daughter_mcpx = data["mcPx"]
    daughter_mcpy = data["mcPy"]
    daughter_mcpz = data["mcPz"]

    e_mcpx = daughter_mcpx[mc_e_only]
    e_mcpy = daughter_mcpy[mc_e_only]
    e_mcpz = daughter_mcpz[mc_e_only]

    e_mcp = np.sqrt(e_mcpx**2 + e_mcpy**2 + e_mcpz**2)
    e_mcr = np.sqrt(e_mcpx * e_mcpx + e_mcpy * e_mcpy)

    e_mcxhat = e_mcpx / e_mcp
    e_mcyhat = e_mcpy / e_mcp
    e_mczhat = e_mcpz / e_mcp
    e_mcrhat = e_mcr / e_mcp

    mc_e_theta = np.arctan2(e_mcrhat, e_mczhat)
    mc_e_phi   = np.arctan2(e_mcyhat, e_mcxhat)

    sel_nu_mcpx = data["mcNuPx"]
    sel_nu_mcpy = data["mcNuPy"]
    sel_nu_mcpz = data["mcNuPz"]

    nu_mcpx = sel_nu_mcpx[mc_e_only]
    nu_mcpy = sel_nu_mcpy[mc_e_only]
    nu_mcpz = sel_nu_mcpz[mc_e_only]

    nu_mcp = np.sqrt(nu_mcpx**2 + nu_mcpy**2 + nu_mcpz**2)
    nu_mcr = np.sqrt(nu_mcpx * nu_mcpx + nu_mcpy * nu_mcpy)

    nu_mcxhat = nu_mcpx / nu_mcp
    nu_mcyhat = nu_mcpy / nu_mcp
    nu_mczhat = nu_mcpz / nu_mcp
    nu_mcrhat = nu_mcr / nu_mcp

    mc_nu_theta = np.arctan2(nu_mcrhat, nu_mczhat)
    mc_nu_phi   = np.arctan2(nu_mcyhat, nu_mcxhat)

    dtheta = mc_nu_theta - mc_e_theta
    dphi = mc_nu_phi - mc_e_phi

    mc_nu_theta = ak.to_numpy(ak.flatten(mc_nu_theta))
    mc_e_theta = ak.to_numpy(ak.flatten(mc_e_theta))

    mc_nu_phi = ak.to_numpy(ak.flatten(mc_nu_phi))
    mc_e_phi = ak.to_numpy(ak.flatten(mc_e_phi))

    dtheta = ak.to_numpy(ak.flatten(dtheta))
    dphi = ak.to_numpy(ak.flatten(dphi))


    plt.figure(figsize=(7, 6))
    plt.scatter(mc_nu_theta, mc_e_theta, s=5, alpha=0.5)
    plt.xlabel(r'Neutrino $\theta_\nu$ [rad]')
    plt.ylabel(r'Electron $\theta_e$ [rad]')
    plt.title(r'\theta for electron vs neutrino')
    plt.tight_layout()
    figname= "nu_e_theta_scatter.png"
    plt.savefig(figname)
    print(f"Saving {figname}...")
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(mc_nu_phi, mc_e_phi, s=5, alpha=0.5)
    plt.xlabel(r'Neutrino $\phi_\nu$ [rad]')
    plt.ylabel(r'Electron $\phi_e$ [rad]')
    plt.title(r'\phi for electron vs neutrino')
    plt.tight_layout()
    figname= "nu_e_phi_scatter.png"
    plt.savefig(figname)
    print(f"Saving {figname}...")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')
    ax.hist(dtheta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 0.3,  facecolor=e_color, label=r"d$\theta$")
    ax.hist(dphi, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha= 0.3,facecolor=pi_color, label=r"d$\phi$")
    ax.hist(dtheta, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=e_color, alpha = 1,  facecolor='none')
    ax.hist(dphi, bins=50, histtype="stepfilled",linewidth=2, density=True, edgecolor= pi_color, alpha=1,facecolor='none')
    ax.set_xlabel(r"Change in angle (rad)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title(f"Difference in angles between neutrino and electron", fontsize=16)
    ax.legend(fontsize=16)
    figname = "d_angles_nu_e.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()

rootTreeToDataFrame()
