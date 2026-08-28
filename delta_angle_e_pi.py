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

    #combine electron x,y,z momentum and pion x,y,z momentum into one array each with awkward.zip
    mc_e = ak.zip({"x": e_mcxhat, "y": e_mcyhat, "z": e_mczhat})
    mc_pi = ak.zip({"x": pi_mcxhat, "y": pi_mcyhat, "z": pi_mczhat})

    #take cartesian product of two combined arrays to "pair up" electrons and pions within the same event
    #ensures that number of electrons and pions are the same and that each event that has them has one electron and one pion (theoretically)
    mc_pairs = ak.cartesian([mc_e, mc_pi],axis=1, nested=True)

    #pull out the electron and pions components of the pairs
    mc_e_pairs = mc_pairs["0"]
    mc_pi_pairs = mc_pairs["1"]

    """
    #opening angle between momenta 
    cos_theta = (mc_e_pairs["x"] * mc_pi_pairs["x"] + mc_e_pairs["y"] * mc_pi_pairs["y"] + mc_e_pairs["z"] * mc_pi_pairs["z"])
    #force cos theta to lie between -1.0 and 1.0 (just in case)
    cos_theta_clipped = ak.where(cos_theta > 1.0, 1.0, ak.where(cos_theta < -1.0, -1.0, cos_theta))
    #get the angle using arccos 
    theta_e_pi = np.arccos(cos_theta_clipped)
    """

    #get theta: the angle between x and z for the pion and electron
    mc_theta_pion = np.arctan2(mc_pi_pairs["x"], mc_pi_pairs["z"])
    #theta_pi_clip = ak.where(mc_theta_pion > np.pi, np.pi, ak.where(mc_theta_pion < -np.pi, -np.pi, mc_theta_pion))
    mc_theta_electron = np.arctan2(mc_e_pairs["x"], mc_e_pairs["z"])
    #theta_e_clip = ak.where(mc_theta_electron > np.pi, np.pi, ak.where(mc_theta_electron < -np.pi, -np.pi, mc_theta_electron))

    #get the difference in angle 
    #mc_dtheta = theta_pi_clip - theta_e_clip
    mc_dtheta = mc_theta_pion - mc_theta_electron

    #get rid of nesting in the array and make it numpy array for plotting 
    mc_dtheta = ak.flatten(mc_dtheta, axis=None)
    mc_dtheta = ak.to_numpy(mc_dtheta)

    #plot mc_dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(mc_dtheta, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\theta_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Truth d$\theta$ between e and $\pi$")
    figname = "mc_dtheta_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(mc_dtheta)}")
    plt.savefig(figname)
    plt.close()

    #get phi: the angle between y and z for the pion and electron
    mc_phi_pion = np.arctan2(mc_pi_pairs["y"], mc_pi_pairs["z"])
    #phi_pi_clip = ak.where(mc_phi_pion > np.pi, np.pi, ak.where(mc_phi_pion < -np.pi, -np.pi, mc_phi_pion))
    mc_phi_electron = np.arctan2(mc_e_pairs["y"], mc_e_pairs["z"])
    #phi_e_clip = ak.where(mc_phi_electron > np.pi, np.pi, ak.where(mc_phi_electron < -np.pi, -np.pi, mc_phi_electron))

    #get the difference in angle 
    #mc_dphi = phi_pi_clip - phi_e_clip
    mc_dphi = mc_phi_pion - mc_phi_electron


    #get rid of nesting in the array and make it numpy array for plotting 
    mc_dphi = ak.flatten(mc_dphi, axis=None)
    mc_dphi = ak.to_numpy(mc_dphi)

    #plot mc_dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(mc_dphi, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\phi_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Truth d$\phi$ between e and $\pi$")
    figname = "mc_dphi_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(mc_dphi)}")
    plt.savefig(figname)
    plt.close()
    

    #get rho: the angle between x and y for the pion and electron
    mc_rho_pion = np.arctan2(mc_pi_pairs["y"], mc_pi_pairs["x"])
    #rho_pi_clip = ak.where(mc_rho_pion > np.pi, np.pi, ak.where(mc_rho_pion < -np.pi, -np.pi, mc_rho_pion))
    mc_rho_electron = np.arctan2(mc_e_pairs["y"], mc_e_pairs["x"])
    #rho_e_clip = ak.where(mc_mc_rho_electron > np.pi, np.pi, ak.where(mc_rho_electron < -np.pi, -np.pi, mc_rho_electron))

    #get the difference in angle 
    #mc_drho = rho_pi_clip - rho_e_clip
    mc_drho = mc_rho_pion - mc_rho_electron

    #get rid of nesting in the array and make it numpy array for plotting 
    mc_drho = ak.flatten(mc_drho, axis=None)
    mc_drho = ak.to_numpy(mc_drho)

    #plot mc_dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(mc_drho, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\rho_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Truth d$\rho$ between e and $\pi$")
    figname = "mc_drho_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(mc_drho)}")
    plt.savefig(figname)
    plt.close()


    #plot all on the same set of axes
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(mc_dtheta, bins=50, histtype="step", linewidth=2, label=r"$d\theta$")
    ax.hist(mc_dphi, bins=50, histtype="step", linewidth=2, label=r"$d\phi$")
    ax.hist(mc_drho, bins=50, histtype="step", linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Truth angle differences between e and $\pi$")
    ax.legend()
    figname = "mc_dall_angles_e_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()
    















    #do the same for shower
    #for ease of use here, asign arrays as variable names
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

    #combine electron x,y,z momentum and pion x,y,z momentum into one array each with awkward.zip
    shwr_e = ak.zip({"x": e_shwrxhat, "y": e_shwryhat, "z": e_shwrzhat})
    shwr_pi = ak.zip({"x": pi_shwrxhat, "y": pi_shwryhat, "z": pi_shwrzhat})

    #take cartesian product of two combined arrays to "pair up" electrons and pions within the same event
    #ensures that number of electrons and pions are the same and that each event that has them has one electron and one pion (theoretically)
    shwr_pairs = ak.cartesian([shwr_e, shwr_pi],axis=1, nested=True)

    #pull out the electron and pions components of the pairs
    shwr_e_pairs = shwr_pairs["0"]
    shwr_pi_pairs = shwr_pairs["1"]

    """
    #opening angle between momenta 
    cos_theta = (shwr_e_pairs["x"] * shwr_pi_pairs["x"] + shwr_e_pairs["y"] * shwr_pi_pairs["y"] + shwr_e_pairs["z"] * shwr_pi_pairs["z"])
    #force cos theta to lie between -1.0 and 1.0 (just in case)
    cos_theta_clipped = ak.where(cos_theta > 1.0, 1.0, ak.where(cos_theta < -1.0, -1.0, cos_theta))
    #get the angle using arccos 
    theta_e_pi = np.arccos(cos_theta_clipped)
    """

    #get theta: the angle between x and z for the pion and electron
    shwr_theta_pion = np.arctan2(shwr_pi_pairs["x"], shwr_pi_pairs["z"])
    #theta_pi_clip = ak.where(shwr_theta_pion > np.pi, np.pi, ak.where(shwr_theta_pion < -np.pi, -np.pi, shwr_theta_pion))
    shwr_theta_electron = np.arctan2(shwr_e_pairs["x"], shwr_e_pairs["z"])
    #theta_e_clip = ak.where(shwr_theta_electron > np.pi, np.pi, ak.where(shwr_theta_electron < -np.pi, -np.pi, shwr_theta_electron))

    #get the difference in angle 
    #shwr_dtheta = theta_pi_clip - theta_e_clip
    shwr_dtheta = shwr_theta_pion - shwr_theta_electron

    #get rid of nesting in the array and make it numpy array for plotting 
    shwr_dtheta = ak.flatten(shwr_dtheta, axis=None)
    shwr_dtheta = ak.to_numpy(shwr_dtheta)

    #plot shwr_dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(shwr_dtheta, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\theta_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Shower d$\theta$ between e and $\pi$")
    figname = "shwr_dtheta_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(shwr_dtheta)}")
    plt.savefig(figname)
    plt.close()

    #get phi: the angle between y and z for the pion and electron
    shwr_phi_pion = np.arctan2(shwr_pi_pairs["y"], shwr_pi_pairs["z"])
    #phi_pi_clip = ak.where(shwr_phi_pion > np.pi, np.pi, ak.where(shwr_phi_pion < -np.pi, -np.pi, shwr_phi_pion))
    shwr_phi_electron = np.arctan2(shwr_e_pairs["y"], shwr_e_pairs["z"])
    #phi_e_clip = ak.where(shwr_phi_electron > np.pi, np.pi, ak.where(shwr_phi_electron < -np.pi, -np.pi, shwr_phi_electron))

    #get the difference in angle 
    #shwr_dphi = phi_pi_clip - phi_e_clip
    shwr_dphi = shwr_phi_pion - shwr_phi_electron


    #get rid of nesting in the array and make it numpy array for plotting 
    shwr_dphi = ak.flatten(shwr_dphi, axis=None)
    shwr_dphi = ak.to_numpy(shwr_dphi)

    #plot shwr_dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(shwr_dphi, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\phi_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Shower d$\phi$ between e and $\pi$")
    figname = "shwr_dphi_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(shwr_dphi)}")
    plt.savefig(figname)
    plt.close()
    

    #get rho: the angle between x and y for the pion and electron
    shwr_rho_pion = np.arctan2(shwr_pi_pairs["y"], shwr_pi_pairs["x"])
    #rho_pi_clip = ak.where(shwr_rho_pion > np.pi, np.pi, ak.where(shwr_rho_pion < -np.pi, -np.pi, shwr_rho_pion))
    shwr_rho_electron = np.arctan2(shwr_e_pairs["y"], shwr_e_pairs["x"])
    #rho_e_clip = ak.where(shwr_shwr_rho_electron > np.pi, np.pi, ak.where(shwr_rho_electron < -np.pi, -np.pi, shwr_rho_electron))

    #get the difference in angle 
    #shwr_drho = rho_pi_clip - rho_e_clip
    shwr_drho = shwr_rho_pion - shwr_rho_electron

    #get rid of nesting in the array and make it numpy array for plotting 
    shwr_drho = ak.flatten(shwr_drho, axis=None)
    shwr_drho = ak.to_numpy(shwr_drho)

    #plot shwr_dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(shwr_drho, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\rho_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Shower d$\rho$ between e and $\pi$")
    figname = "shwr_drho_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(shwr_drho)}")
    plt.savefig(figname)
    plt.close()


    #plot all on the same set of axes
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(shwr_dtheta, bins=50, histtype="step", linewidth=2, label=r"$d\theta$")
    ax.hist(shwr_dphi, bins=50, histtype="step", linewidth=2, label=r"$d\phi$")
    ax.hist(shwr_drho, bins=50, histtype="step", linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Shower angle differences between e and $\pi$")
    ax.legend()
    figname = "shwr_dall_angles_e_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()
















    #do the same for interface
    intpx = data["dirX"]
    intpy = data["dirY"]
    intpz = data["dirZ"]
    pdg = data["mcPDG"]

    int_e_only = abs(pdg) == 11
    int_pi_only = abs(pdg) == 211

    e_intpx = intpx[int_e_only]
    e_intpy = intpy[int_e_only]
    e_intpz = intpz[int_e_only]

    pi_intpx = intpx[int_pi_only]
    pi_intpy = intpy[int_pi_only]
    pi_intpz = intpz[int_pi_only]

    e_intp = np.sqrt(e_intpx**2 + e_intpy**2 + e_intpz**2)
    pi_intp = np.sqrt(pi_intpx**2 + pi_intpy**2 + pi_intpz**2)

    e_intxhat = e_intpx / e_intp
    e_intyhat = e_intpy / e_intp
    e_intzhat = e_intpz / e_intp

    pi_intxhat = pi_intpx / pi_intp
    pi_intyhat = pi_intpy / pi_intp
    pi_intzhat = pi_intpz / pi_intp

    int_e = ak.zip({"x": e_intxhat, "y": e_intyhat, "z": e_intzhat})
    int_pi = ak.zip({"x": pi_intxhat, "y": pi_intyhat, "z": pi_intzhat})
    int_pairs = ak.cartesian([int_e, int_pi],axis=1, nested=True)
    int_e_pairs = int_pairs["0"]
    int_pi_pairs = int_pairs["1"]

    int_theta_pion = np.arctan2(int_pi_pairs["x"], int_pi_pairs["z"])
    #theta_pi_clip = ak.where(int_theta_pion > np.pi, np.pi, ak.where(int_theta_pion < -np.pi, -np.pi, int_theta_pion))
    int_theta_electron = np.arctan2(int_e_pairs["x"], int_e_pairs["z"])
    #theta_e_clip = ak.where(int_theta_electron > np.pi, np.pi, ak.where(int_theta_electron < -np.pi, -np.pi, int_theta_electron))

    #int_dtheta = theta_pi_clip - theta_e_clip
    int_dtheta = int_theta_pion - int_theta_electron
    int_dtheta = ak.flatten(int_dtheta, axis=None)
    int_dtheta = ak.to_numpy(int_dtheta)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(int_dtheta, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\theta_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Interface d$\theta$ between e and $\pi$")
    figname = "int_dtheta_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(int_dtheta)}")
    plt.savefig(figname)
    plt.close()

    int_phi_pion = np.arctan2(int_pi_pairs["y"], int_pi_pairs["z"])
    #phi_pi_clip = ak.where(int_phi_pion > np.pi, np.pi, ak.where(int_phi_pion < -np.pi, -np.pi, int_phi_pion))
    int_phi_electron = np.arctan2(int_e_pairs["y"], int_e_pairs["z"])
    #phi_e_clip = ak.where(int_phi_electron > np.pi, np.pi, ak.where(int_phi_electron < -np.pi, -np.pi, int_phi_electron))

    #int_dphi = phi_pi_clip - phi_e_clip
    int_dphi = int_phi_pion - int_phi_electron
    int_dphi = ak.flatten(int_dphi, axis=None)
    int_dphi = ak.to_numpy(int_dphi)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(int_dphi, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\phi_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Interface d$\phi$ between e and $\pi$")
    figname = "int_dphi_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(int_dphi)}")
    plt.savefig(figname)
    plt.close()
    

    int_rho_pion = np.arctan2(int_pi_pairs["y"], int_pi_pairs["x"])
    #rho_pi_clip = ak.where(int_rho_pion > np.pi, np.pi, ak.where(int_rho_pion < -np.pi, -np.pi, int_rho_pion))
    int_rho_electron = np.arctan2(int_e_pairs["y"], int_e_pairs["x"])
    #rho_e_clip = ak.where(int_int_rho_electron > np.pi, np.pi, ak.where(int_rho_electron < -np.pi, -np.pi, int_rho_electron))

    #int_drho = rho_pi_clip - rho_e_clip
    int_drho = int_rho_pion - int_rho_electron
    int_drho = ak.flatten(int_drho, axis=None)
    int_drho = ak.to_numpy(int_drho)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(int_drho, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\rho_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Interface d$\rho$ between e and $\pi$")
    figname = "int_drho_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(int_drho)}")
    plt.savefig(figname)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(int_dtheta, bins=50, histtype="step", linewidth=2, label=r"$d\theta$")
    ax.hist(int_dphi, bins=50, histtype="step", linewidth=2, label=r"$d\phi$")
    ax.hist(int_drho, bins=50, histtype="step", linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Interface angle differences between e and $\pi$")
    ax.legend()
    figname = "int_dall_angles_e_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()















    #do the same for trackstart
    trkstrtpx = data["trkfitStartDirX"]
    trkstrtpy = data["trkfitStartDirY"]
    trkstrtpz = data["trkfitStartDirZ"]
    pdg = data["mcPDG"]

    trkstrt_e_only = abs(pdg) == 11
    trkstrt_pi_only = abs(pdg) == 211

    e_trkstrtpx = trkstrtpx[trkstrt_e_only]
    e_trkstrtpy = trkstrtpy[trkstrt_e_only]
    e_trkstrtpz = trkstrtpz[trkstrt_e_only]

    pi_trkstrtpx = trkstrtpx[trkstrt_pi_only]
    pi_trkstrtpy = trkstrtpy[trkstrt_pi_only]
    pi_trkstrtpz = trkstrtpz[trkstrt_pi_only]

    e_trkstrtp = np.sqrt(e_trkstrtpx**2 + e_trkstrtpy**2 + e_trkstrtpz**2)
    pi_trkstrtp = np.sqrt(pi_trkstrtpx**2 + pi_trkstrtpy**2 + pi_trkstrtpz**2)

    e_trkstrtxhat = e_trkstrtpx / e_trkstrtp
    e_trkstrtyhat = e_trkstrtpy / e_trkstrtp
    e_trkstrtzhat = e_trkstrtpz / e_trkstrtp

    pi_trkstrtxhat = pi_trkstrtpx / pi_trkstrtp
    pi_trkstrtyhat = pi_trkstrtpy / pi_trkstrtp
    pi_trkstrtzhat = pi_trkstrtpz / pi_trkstrtp

    trkstrt_e = ak.zip({"x": e_trkstrtxhat, "y": e_trkstrtyhat, "z": e_trkstrtzhat})
    trkstrt_pi = ak.zip({"x": pi_trkstrtxhat, "y": pi_trkstrtyhat, "z": pi_trkstrtzhat})
    trkstrt_pairs = ak.cartesian([trkstrt_e, trkstrt_pi],axis=1, nested=True)
    trkstrt_e_pairs = trkstrt_pairs["0"]
    trkstrt_pi_pairs = trkstrt_pairs["1"]

    trkstrt_theta_pion = np.arctan2(trkstrt_pi_pairs["x"], trkstrt_pi_pairs["z"])
    #theta_pi_clip = ak.where(trkstrt_theta_pion > np.pi, np.pi, ak.where(trkstrt_theta_pion < -np.pi, -np.pi, trkstrt_theta_pion))
    trkstrt_theta_electron = np.arctan2(trkstrt_e_pairs["x"], trkstrt_e_pairs["z"])
    #theta_e_clip = ak.where(trkstrt_theta_electron > np.pi, np.pi, ak.where(trkstrt_theta_electron < -np.pi, -np.pi, trkstrt_theta_electron))

    #trkstrt_dtheta = theta_pi_clip - theta_e_clip
    trkstrt_dtheta = trkstrt_theta_pion - trkstrt_theta_electron
    trkstrt_dtheta = ak.flatten(trkstrt_dtheta, axis=None)
    trkstrt_dtheta = ak.to_numpy(trkstrt_dtheta)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkstrt_dtheta, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\theta_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"TrackStartDir d$\theta$ between e and $\pi$")
    figname = "trkstrt_dtheta_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(trkstrt_dtheta)}")
    plt.savefig(figname)
    plt.close()

    trkstrt_phi_pion = np.arctan2(trkstrt_pi_pairs["y"], trkstrt_pi_pairs["z"])
    #phi_pi_clip = ak.where(trkstrt_phi_pion > np.pi, np.pi, ak.where(trkstrt_phi_pion < -np.pi, -np.pi, trkstrt_phi_pion))
    trkstrt_phi_electron = np.arctan2(trkstrt_e_pairs["y"], trkstrt_e_pairs["z"])
    #phi_e_clip = ak.where(trkstrt_phi_electron > np.pi, np.pi, ak.where(trkstrt_phi_electron < -np.pi, -np.pi, trkstrt_phi_electron))

    #trkstrt_dphi = phi_pi_clip - phi_e_clip
    trkstrt_dphi = trkstrt_phi_pion - trkstrt_phi_electron
    trkstrt_dphi = ak.flatten(trkstrt_dphi, axis=None)
    trkstrt_dphi = ak.to_numpy(trkstrt_dphi)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkstrt_dphi, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\phi_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"TrackStartDir d$\phi$ between e and $\pi$")
    figname = "trkstrt_dphi_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(int_dphi)}")
    plt.savefig(figname)
    plt.close()
    

    trkstrt_rho_pion = np.arctan2(trkstrt_pi_pairs["y"], trkstrt_pi_pairs["x"])
    #rho_pi_clip = ak.where(trkstrt_rho_pion > np.pi, np.pi, ak.where(trkstrt_rho_pion < -np.pi, -np.pi, trkstrt_rho_pion))
    trkstrt_rho_electron = np.arctan2(trkstrt_e_pairs["y"], trkstrt_e_pairs["x"])
    #rho_e_clip = ak.where(trkstrt_trkstrt_rho_electron > np.pi, np.pi, ak.where(trkstrt_rho_electron < -np.pi, -np.pi, trkstrt_rho_electron))

    #trkstrt_drho = rho_pi_clip - rho_e_clip
    trkstrt_drho = trkstrt_rho_pion - trkstrt_rho_electron
    trkstrt_drho = ak.flatten(trkstrt_drho, axis=None)
    trkstrt_drho = ak.to_numpy(trkstrt_drho)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkstrt_drho, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\rho_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"TrackStartDir d$\rho$ between e and $\pi$")
    figname = "trkstrt_drho_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(trkstrt_drho)}")
    plt.savefig(figname)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkstrt_dtheta, bins=50, histtype="step", linewidth=2, label=r"$d\theta$")
    ax.hist(trkstrt_dphi, bins=50, histtype="step", linewidth=2, label=r"$d\phi$")
    ax.hist(trkstrt_drho, bins=50, histtype="step", linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"TrackStartDir angle differences between e and $\pi$")
    ax.legend()
    figname = "trkstrt_dall_angles_e_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()














    #do the same for trackend
    trkendpx = data["trkfitEndDirX"]
    trkendpy = data["trkfitEndDirY"]
    trkendpz = data["trkfitEndDirZ"]
    pdg = data["mcPDG"]

    trkend_e_only = abs(pdg) == 11
    trkend_pi_only = abs(pdg) == 211

    e_trkendpx = trkendpx[trkend_e_only]
    e_trkendpy = trkendpy[trkend_e_only]
    e_trkendpz = trkendpz[trkend_e_only]

    pi_trkendpx = trkendpx[trkend_pi_only]
    pi_trkendpy = trkendpy[trkend_pi_only]
    pi_trkendpz = trkendpz[trkend_pi_only]

    e_trkendp = np.sqrt(e_trkendpx**2 + e_trkendpy**2 + e_trkendpz**2)
    pi_trkendp = np.sqrt(pi_trkendpx**2 + pi_trkendpy**2 + pi_trkendpz**2)

    e_trkendxhat = e_trkendpx / e_trkendp
    e_trkendyhat = e_trkendpy / e_trkendp
    e_trkendzhat = e_trkendpz / e_trkendp

    pi_trkendxhat = pi_trkendpx / pi_trkendp
    pi_trkendyhat = pi_trkendpy / pi_trkendp
    pi_trkendzhat = pi_trkendpz / pi_trkendp

    trkend_e = ak.zip({"x": e_trkendxhat, "y": e_trkendyhat, "z": e_trkendzhat})
    trkend_pi = ak.zip({"x": pi_trkendxhat, "y": pi_trkendyhat, "z": pi_trkendzhat})
    trkend_pairs = ak.cartesian([trkend_e, trkend_pi],axis=1, nested=True)
    trkend_e_pairs = trkend_pairs["0"]
    trkend_pi_pairs = trkend_pairs["1"]

    trkend_theta_pion = np.arctan2(trkend_pi_pairs["x"], trkend_pi_pairs["z"])
    #theta_pi_clip = ak.where(trkend_theta_pion > np.pi, np.pi, ak.where(trkend_theta_pion < -np.pi, -np.pi, trkend_theta_pion))
    trkend_theta_electron = np.arctan2(trkend_e_pairs["x"], trkend_e_pairs["z"])
    #theta_e_clip = ak.where(trkend_theta_electron > np.pi, np.pi, ak.where(trkend_theta_electron < -np.pi, -np.pi, trkend_theta_electron))

    #trkend_dtheta = theta_pi_clip - theta_e_clip
    trkend_dtheta = trkend_theta_pion - trkend_theta_electron
    trkend_dtheta = ak.flatten(trkend_dtheta, axis=None)
    trkend_dtheta = ak.to_numpy(trkend_dtheta)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkend_dtheta, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\theta_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"TrackEndDir d$\theta$ between e and $\pi$")
    figname = "trkend_dtheta_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(trkend_dtheta)}")
    plt.savefig(figname)
    plt.close()

    trkend_phi_pion = np.arctan2(trkend_pi_pairs["y"], trkend_pi_pairs["z"])
    #phi_pi_clip = ak.where(trkend_phi_pion > np.pi, np.pi, ak.where(trkend_phi_pion < -np.pi, -np.pi, trkend_phi_pion))
    trkend_phi_electron = np.arctan2(trkend_e_pairs["y"], trkend_e_pairs["z"])
    #phi_e_clip = ak.where(trkend_phi_electron > np.pi, np.pi, ak.where(trkend_phi_electron < -np.pi, -np.pi, trkend_phi_electron))

    #trkend_dphi = phi_pi_clip - phi_e_clip
    trkend_dphi = trkend_phi_pion - trkend_phi_electron
    trkend_dphi = ak.flatten(trkend_dphi, axis=None)
    trkend_dphi = ak.to_numpy(trkend_dphi)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkend_dphi, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\phi_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"TrackEndDir d$\phi$ between e and $\pi$")
    figname = "trkend_dphi_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(int_dphi)}")
    plt.savefig(figname)
    plt.close()
    

    trkend_rho_pion = np.arctan2(trkend_pi_pairs["y"], trkend_pi_pairs["x"])
    #rho_pi_clip = ak.where(trkend_rho_pion > np.pi, np.pi, ak.where(trkend_rho_pion < -np.pi, -np.pi, trkend_rho_pion))
    trkend_rho_electron = np.arctan2(trkend_e_pairs["y"], trkend_e_pairs["x"])
    #rho_e_clip = ak.where(trkend_trkend_rho_electron > np.pi, np.pi, ak.where(trkend_rho_electron < -np.pi, -np.pi, trkend_rho_electron))

    #trkend_drho = rho_pi_clip - rho_e_clip
    trkend_drho = trkend_rho_pion - trkend_rho_electron
    trkend_drho = ak.flatten(trkend_drho, axis=None)
    trkend_drho = ak.to_numpy(trkend_drho)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkend_drho, bins=50, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\rho_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"TrackEndtDir d$\rho$ between e and $\pi$")
    figname = "trkend_drho_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(trkend_drho)}")
    plt.savefig(figname)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(trkend_dtheta, bins=50, histtype="step", linewidth=2, label=r"$d\theta$")
    ax.hist(trkend_dphi, bins=50, histtype="step", linewidth=2, label=r"$d\phi$")
    ax.hist(trkend_drho, bins=50, histtype="step", linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"TrackEndDir angle differences between e and $\pi$")
    ax.legend()
    figname = "trkend_dall_angles_e_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()



    #for comparaison's sake
    fig, axes = plt.subplots(3, 2,figsize=(16, 18),layout="constrained")

    ax = axes[0, 0]
    ax.hist(mc_dtheta, bins=50, histtype="step",linewidth=2, label=r"$d\theta$")
    ax.hist(mc_dphi, bins=50, histtype="step",linewidth=2, label=r"$d\phi$")
    ax.hist(mc_drho, bins=50, histtype="step",linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title("Truth angle differences")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(shwr_dtheta, bins=50, histtype="step",linewidth=2, label=r"$d\theta$")
    ax.hist(shwr_dphi, bins=50, histtype="step",linewidth=2, label=r"$d\phi$")
    ax.hist(shwr_drho, bins=50, histtype="step",linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title("Shower angle differences")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(int_dtheta, bins=50, histtype="step",linewidth=2, label=r"$d\theta$")
    ax.hist(int_dphi, bins=50, histtype="step",linewidth=2, label=r"$d\phi$")
    ax.hist(int_drho, bins=50, histtype="step",linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title("Interface angle differences")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(trkstrt_dtheta, bins=50, histtype="step",linewidth=2, label=r"$d\theta$")
    ax.hist(trkstrt_dphi, bins=50, histtype="step",linewidth=2, label=r"$d\phi$")
    ax.hist(trkstrt_drho, bins=50, histtype="step",linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title("TrackStartDir angle differences")
    ax.legend()

    ax = axes[2, 0]
    ax.hist(trkend_dtheta, bins=50, histtype="step",linewidth=2, label=r"$d\theta$")
    ax.hist(trkend_dphi, bins=50, histtype="step",linewidth=2, label=r"$d\phi$")
    ax.hist(trkend_drho, bins=50, histtype="step",linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title("TrackEndDir angle differences")
    ax.legend()

    axes[2, 1].axis("off")
    figname = "all_dall_angles_in_one_e_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname, dpi=300)
    plt.close()


rootTreeToDataFrame()
