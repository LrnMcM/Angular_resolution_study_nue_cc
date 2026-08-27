import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt


def rootTreeToDataFrame():
    #insert file name here
    #fname = "NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"
    fname = "LArRecoND_1_999_Partially_Cheated.root"

    #load file
    file = uproot.open(fname)
    #get tree from file
    t = file["LArRecoND"]

    #pull out the branches you need: momenta in x,y,z and PDG code here
    branches = ["mcPx", "mcPy", "mcPz", "mcPDG", 'shwrfitDirX', 'shwrfitDirY', 'shwrfitDirZ', 'dirX', 'dirY', 'dirZ', 'trkfitStartDirX', 'trkfitStartDirY', 'trkfitStartDirZ', 'trkfitEndDirX', 'trkfitEndDirY', 'trkfitEndDirZ']

    #read data into arrays
    data = t.arrays(branches, library="ak")

    #for ease of use here, asign arrays as variable names
    mcpx = data["mcPx"]
    mcpy = data["mcPy"]
    mcpz = data["mcPz"]
    pdg = data["mcPDG"]

    #isolate electrons/positrons (with absolute value) and pions(+/-)
    mc_e_only = abs(pdg) == 11
    mc_pi_only = abs(pdg) == 211

    #get out the momenta in x,y,z for electrons and pions 
    e_mcpx = mcpx[mc_e_only]
    e_mcpy = mcpy[mc_e_only]
    e_mcpz = mcpz[mc_e_only]

    pi_mcpx = mcpx[mc_pi_only]
    pi_mcpy = mcpy[mc_pi_only]
    pi_mcpz = mcpz[mc_pi_only]

    #get total momentum 
    e_mcp = np.sqrt(e_mcpx**2 + e_mcpy**2 + e_mcpz**2)
    pi_mcp = np.sqrt(pi_mcpx**2 + pi_mcpy**2 + pi_mcpz**2)

    #normalise momentum
    e_mcxhat = e_mcpx / e_mcp
    e_mcyhat = e_mcpy / e_mcp
    e_mczhat = e_mcpz / e_mcp

    pi_mcxhat = pi_mcpx / pi_mcp
    pi_mcyhat = pi_mcpy / pi_mcp
    pi_mczhat = pi_mcpz / pi_mcp

    mc_e_theta = np.arctan2(e_mcxhat, e_mczhat)
    mc_e_phi   = np.arctan2(e_mcyhat, e_mczhat)
    mc_e_rho   = np.arctan2(e_mcyhat, e_mcxhat)
    mc_pi_theta = np.arctan2(pi_mcxhat, pi_mczhat)
    mc_pi_phi   = np.arctan2(pi_mcyhat, pi_mczhat)
    mc_pi_rho   = np.arctan2(pi_mcyhat, pi_mcxhat)


    mc_e_theta_numpy = ak.to_numpy(ak.flatten(mc_e_theta, axis=None))
    mc_e_phi_numpy = ak.to_numpy(ak.flatten(mc_e_phi, axis=None))
    mc_e_rho_numpy   = ak.to_numpy(ak.flatten(mc_e_rho, axis=None))
    mc_pi_theta_numpy = ak.to_numpy(ak.flatten(mc_pi_theta, axis=None))
    mc_pi_phi_numpy = ak.to_numpy(ak.flatten(mc_pi_phi, axis=None))
    mc_pi_rho_numpy = ak.to_numpy(ak.flatten(mc_pi_rho, axis=None))


    bins = np.linspace(-np.pi, np.pi, 51)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(mc_e_theta_numpy, bins=bins, histtype="step", linewidth=2, label="Electron")
    ax.hist(mc_pi_theta_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"Truth $\theta$ for e and $\pi$")
    ax.legend()
    figname = "mc_theta_e_vs_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of electrons: {len(mc_e_theta_numpy)}")
    print(f"Number of pions: {len(mc_pi_theta_numpy)}")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(mc_e_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(mc_pi_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"Truth $\phi$ for e and $\pi$")
    ax.legend()
    figname = "mc_phi_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(mc_e_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(mc_pi_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\rho$ (rad)")
    ax.set_ylabel("Particles")
    ax.set_title(r"Truth $\rho$ for e and $\pi$")
    ax.legend()
    figname = "mc_rho_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()











    shwrpx = data["shwrfitDirX"]
    shwrpy = data["shwrfitDirY"]
    shwrpz = data["shwrfitDirZ"]
    pdg = data["mcPDG"]

    #isolate electrons/positrons (with absolute value) and pions(+/-)
    shwr_e_only = abs(pdg) == 11
    shwr_pi_only = abs(pdg) == 211

    #get out the momenta in x,y,z for electrons and pions 
    e_shwrpx = shwrpx[shwr_e_only]
    e_shwrpy = shwrpy[shwr_e_only]
    e_shwrpz = shwrpz[shwr_e_only]

    pi_shwrpx = shwrpx[shwr_pi_only]
    pi_shwrpy = shwrpy[shwr_pi_only]
    pi_shwrpz = shwrpz[shwr_pi_only]

    #get total momentum 
    e_shwrp = np.sqrt(e_shwrpx**2 + e_shwrpy**2 + e_shwrpz**2)
    pi_shwrp = np.sqrt(pi_shwrpx**2 + pi_shwrpy**2 + pi_shwrpz**2)

    #normalise momentum
    e_shwrxhat = e_shwrpx / e_shwrp
    e_shwryhat = e_shwrpy / e_shwrp
    e_shwrzhat = e_shwrpz / e_shwrp

    pi_shwrxhat = pi_shwrpx / pi_shwrp
    pi_shwryhat = pi_shwrpy / pi_shwrp
    pi_shwrzhat = pi_shwrpz / pi_shwrp

    shwr_e_theta = np.arctan2(e_shwrxhat, e_shwrzhat)
    shwr_e_phi   = np.arctan2(e_shwryhat, e_shwrzhat)
    shwr_e_rho   = np.arctan2(e_shwryhat, e_shwrxhat)
    shwr_pi_theta = np.arctan2(pi_shwrxhat, pi_shwrzhat)
    shwr_pi_phi   = np.arctan2(pi_shwryhat, pi_shwrzhat)
    shwr_pi_rho   = np.arctan2(pi_shwryhat, pi_shwrxhat)


    shwr_e_theta_numpy = ak.to_numpy(ak.flatten(shwr_e_theta, axis=None))
    shwr_e_phi_numpy = ak.to_numpy(ak.flatten(shwr_e_phi, axis=None))
    shwr_e_rho_numpy   = ak.to_numpy(ak.flatten(shwr_e_rho, axis=None))
    shwr_pi_theta_numpy = ak.to_numpy(ak.flatten(shwr_pi_theta, axis=None))
    shwr_pi_phi_numpy = ak.to_numpy(ak.flatten(shwr_pi_phi, axis=None))
    shwr_pi_rho_numpy = ak.to_numpy(ak.flatten(shwr_pi_rho, axis=None))


    bins = np.linspace(-np.pi, np.pi, 51)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(shwr_e_theta_numpy, bins=bins, histtype="step", linewidth=2, label="Electron")
    ax.hist(shwr_pi_theta_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"Shower $\theta$ for e and $\pi$")
    ax.legend()
    figname = "shwr_theta_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(shwr_e_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(shwr_pi_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"Shower $\phi$ for e and $\pi$")
    ax.legend()
    figname = "shwr_phi_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(shwr_e_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(shwr_pi_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\rho$ (rad)")
    ax.set_ylabel("Particles")
    ax.set_title(r"Shower $\rho$ for e and $\pi$")
    ax.legend()
    figname = "shwr_rho_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()











    intrfpx = data["dirX"]
    intrfpy = data["dirY"]
    intrfpz = data["dirZ"]
    pdg = data["mcPDG"]

    #isolate electrons/positrons (with absolute value) and pions(+/-)
    intrf_e_only = abs(pdg) == 11
    intrf_pi_only = abs(pdg) == 211

    #get out the momenta in x,y,z for electrons and pions 
    e_intrfpx = intrfpx[intrf_e_only]
    e_intrfpy = intrfpy[intrf_e_only]
    e_intrfpz = intrfpz[intrf_e_only]

    pi_intrfpx = intrfpx[intrf_pi_only]
    pi_intrfpy = intrfpy[intrf_pi_only]
    pi_intrfpz = intrfpz[intrf_pi_only]

    #get total momentum 
    e_intrfp = np.sqrt(e_intrfpx**2 + e_intrfpy**2 + e_intrfpz**2)
    pi_intrfp = np.sqrt(pi_intrfpx**2 + pi_intrfpy**2 + pi_intrfpz**2)

    #normalise momentum
    e_intrfxhat = e_intrfpx / e_intrfp
    e_intrfyhat = e_intrfpy / e_intrfp
    e_intrfzhat = e_intrfpz / e_intrfp

    pi_intrfxhat = pi_intrfpx / pi_intrfp
    pi_intrfyhat = pi_intrfpy / pi_intrfp
    pi_intrfzhat = pi_intrfpz / pi_intrfp

    intrf_e_theta = np.arctan2(e_intrfxhat, e_intrfzhat)
    intrf_e_phi   = np.arctan2(e_intrfyhat, e_intrfzhat)
    intrf_e_rho   = np.arctan2(e_intrfyhat, e_intrfxhat)
    intrf_pi_theta = np.arctan2(pi_intrfxhat, pi_intrfzhat)
    intrf_pi_phi   = np.arctan2(pi_intrfyhat, pi_intrfzhat)
    intrf_pi_rho   = np.arctan2(pi_intrfyhat, pi_intrfxhat)


    intrf_e_theta_numpy = ak.to_numpy(ak.flatten(intrf_e_theta, axis=None))
    intrf_e_phi_numpy = ak.to_numpy(ak.flatten(intrf_e_phi, axis=None))
    intrf_e_rho_numpy   = ak.to_numpy(ak.flatten(intrf_e_rho, axis=None))
    intrf_pi_theta_numpy = ak.to_numpy(ak.flatten(intrf_pi_theta, axis=None))
    intrf_pi_phi_numpy = ak.to_numpy(ak.flatten(intrf_pi_phi, axis=None))
    intrf_pi_rho_numpy = ak.to_numpy(ak.flatten(intrf_pi_rho, axis=None))


    bins = np.linspace(-np.pi, np.pi, 51)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(intrf_e_theta_numpy, bins=bins, histtype="step", linewidth=2, label="Electron")
    ax.hist(intrf_pi_theta_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"Shower $\theta$ for e and $\pi$")
    ax.legend()
    figname = "intrf_theta_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(intrf_e_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(intrf_pi_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"Shower $\phi$ for e and $\pi$")
    ax.legend()
    figname = "intrf_phi_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(intrf_e_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(intrf_pi_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\rho$ (rad)")
    ax.set_ylabel("Particles")
    ax.set_title(r"Shower $\rho$ for e and $\pi$")
    ax.legend()
    figname = "intrf_rho_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()















    trkendpx = data["trkfitStartDirX"]
    trkendpy = data["trkfitStartDirY"]
    trkendpz = data["trkfitStartDirZ"]
    pdg = data["mcPDG"]

    #isolate electrons/positrons (with absolute value) and pions(+/-)
    trkend_e_only = abs(pdg) == 11
    trkend_pi_only = abs(pdg) == 211

    #get out the momenta in x,y,z for electrons and pions 
    e_trkendpx = trkendpx[trkend_e_only]
    e_trkendpy = trkendpy[trkend_e_only]
    e_trkendpz = trkendpz[trkend_e_only]

    pi_trkendpx = trkendpx[trkend_pi_only]
    pi_trkendpy = trkendpy[trkend_pi_only]
    pi_trkendpz = trkendpz[trkend_pi_only]

    #get total momentum 
    e_trkendp = np.sqrt(e_trkendpx**2 + e_trkendpy**2 + e_trkendpz**2)
    pi_trkendp = np.sqrt(pi_trkendpx**2 + pi_trkendpy**2 + pi_trkendpz**2)

    #normalise momentum
    e_trkendxhat = e_trkendpx / e_trkendp
    e_trkendyhat = e_trkendpy / e_trkendp
    e_trkendzhat = e_trkendpz / e_trkendp

    pi_trkendxhat = pi_trkendpx / pi_trkendp
    pi_trkendyhat = pi_trkendpy / pi_trkendp
    pi_trkendzhat = pi_trkendpz / pi_trkendp

    trkend_e_theta = np.arctan2(e_trkendxhat, e_trkendzhat)
    trkend_e_phi   = np.arctan2(e_trkendyhat, e_trkendzhat)
    trkend_e_rho   = np.arctan2(e_trkendyhat, e_trkendxhat)
    trkend_pi_theta = np.arctan2(pi_trkendxhat, pi_trkendzhat)
    trkend_pi_phi   = np.arctan2(pi_trkendyhat, pi_trkendzhat)
    trkend_pi_rho   = np.arctan2(pi_trkendyhat, pi_trkendxhat)


    trkend_e_theta_numpy = ak.to_numpy(ak.flatten(trkend_e_theta, axis=None))
    trkend_e_phi_numpy = ak.to_numpy(ak.flatten(trkend_e_phi, axis=None))
    trkend_e_rho_numpy   = ak.to_numpy(ak.flatten(trkend_e_rho, axis=None))
    trkend_pi_theta_numpy = ak.to_numpy(ak.flatten(trkend_pi_theta, axis=None))
    trkend_pi_phi_numpy = ak.to_numpy(ak.flatten(trkend_pi_phi, axis=None))
    trkend_pi_rho_numpy = ak.to_numpy(ak.flatten(trkend_pi_rho, axis=None))


    bins = np.linspace(-np.pi, np.pi, 51)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkend_e_theta_numpy, bins=bins, histtype="step", linewidth=2, label="Electron")
    ax.hist(trkend_pi_theta_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"TrackStartDir $\theta$ for e and $\pi$")
    ax.legend()
    figname = "trkend_theta_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkend_e_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(trkend_pi_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"TrackStartDir $\phi$ for e and $\pi$")
    ax.legend()
    figname = "trkend_phi_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkend_e_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(trkend_pi_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\rho$ (rad)")
    ax.set_ylabel("Particles")
    ax.set_title(r"TrackStartDir $\rho$ for e and $\pi$")
    ax.legend()
    figname = "trkend_rho_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()












    trkstrtpx = data["trkfitEndDirX"]
    trkstrtpy = data["trkfitEndDirY"]
    trkstrtpz = data["trkfitEndDirZ"]
    pdg = data["mcPDG"]

    #isolate electrons/positrons (with absolute value) and pions(+/-)
    trkstrt_e_only = abs(pdg) == 11
    trkstrt_pi_only = abs(pdg) == 211

    #get out the momenta in x,y,z for electrons and pions 
    e_trkstrtpx = trkstrtpx[trkstrt_e_only]
    e_trkstrtpy = trkstrtpy[trkstrt_e_only]
    e_trkstrtpz = trkstrtpz[trkstrt_e_only]

    pi_trkstrtpx = trkstrtpx[trkstrt_pi_only]
    pi_trkstrtpy = trkstrtpy[trkstrt_pi_only]
    pi_trkstrtpz = trkstrtpz[trkstrt_pi_only]

    #get total momentum 
    e_trkstrtp = np.sqrt(e_trkstrtpx**2 + e_trkstrtpy**2 + e_trkstrtpz**2)
    pi_trkstrtp = np.sqrt(pi_trkstrtpx**2 + pi_trkstrtpy**2 + pi_trkstrtpz**2)

    #normalise momentum
    e_trkstrtxhat = e_trkstrtpx / e_trkstrtp
    e_trkstrtyhat = e_trkstrtpy / e_trkstrtp
    e_trkstrtzhat = e_trkstrtpz / e_trkstrtp

    pi_trkstrtxhat = pi_trkstrtpx / pi_trkstrtp
    pi_trkstrtyhat = pi_trkstrtpy / pi_trkstrtp
    pi_trkstrtzhat = pi_trkstrtpz / pi_trkstrtp

    trkstrt_e_theta = np.arctan2(e_trkstrtxhat, e_trkstrtzhat)
    trkstrt_e_phi   = np.arctan2(e_trkstrtyhat, e_trkstrtzhat)
    trkstrt_e_rho   = np.arctan2(e_trkstrtyhat, e_trkstrtxhat)
    trkstrt_pi_theta = np.arctan2(pi_trkstrtxhat, pi_trkstrtzhat)
    trkstrt_pi_phi   = np.arctan2(pi_trkstrtyhat, pi_trkstrtzhat)
    trkstrt_pi_rho   = np.arctan2(pi_trkstrtyhat, pi_trkstrtxhat)


    trkstrt_e_theta_numpy = ak.to_numpy(ak.flatten(trkstrt_e_theta, axis=None))
    trkstrt_e_phi_numpy = ak.to_numpy(ak.flatten(trkstrt_e_phi, axis=None))
    trkstrt_e_rho_numpy   = ak.to_numpy(ak.flatten(trkstrt_e_rho, axis=None))
    trkstrt_pi_theta_numpy = ak.to_numpy(ak.flatten(trkstrt_pi_theta, axis=None))
    trkstrt_pi_phi_numpy = ak.to_numpy(ak.flatten(trkstrt_pi_phi, axis=None))
    trkstrt_pi_rho_numpy = ak.to_numpy(ak.flatten(trkstrt_pi_rho, axis=None))


    bins = np.linspace(-np.pi, np.pi, 51)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkstrt_e_theta_numpy, bins=bins, histtype="step", linewidth=2, label="Electron")
    ax.hist(trkstrt_pi_theta_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"TrackEndDir $\theta$ for e and $\pi$")
    ax.legend()
    figname = "trkstrt_theta_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkstrt_e_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(trkstrt_pi_phi_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel("Count")
    ax.set_title(r"TrackEndDir $\phi$ for e and $\pi$")
    ax.legend()
    figname = "trkstrt_phi_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkstrt_e_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Electron")
    ax.hist(trkstrt_pi_rho_numpy, bins=bins, histtype="step",linewidth=2, label="Pion")
    ax.set_xlabel(r"$\rho$ (rad)")
    ax.set_ylabel("Particles")
    ax.set_title(r"TrackEndDir $\rho$ for e and $\pi$")
    ax.legend()
    figname = "trkstrt_rho_e_vs_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


rootTreeToDataFrame()
