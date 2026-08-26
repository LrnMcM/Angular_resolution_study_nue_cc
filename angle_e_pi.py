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
    branches = ["mcPx", "mcPy", "mcPz", "mcPDG", 'shwrfitDirX', 'shwrfitDirY', 'shwrfitDirZ']

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
    figname = "mc_theta_e_pi.png"
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
    figname = "mc_phi_e_pi.png"
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
    figname = "mc_rho_e_pi.png"
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
    figname = "mc_all_angles_e_pi.png"
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
    figname = "shwr_theta_e_pi.png"
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
    figname = "shwr_phi_e_pi.png"
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
    figname = "shwr_rho_e_pi.png"
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
    figname = "shwr_all_angles_e_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


rootTreeToDataFrame()
