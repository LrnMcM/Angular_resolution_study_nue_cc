import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt


def rootTreeToDataFrame():
    #insert file name here
    fname = "NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"

    #load file
    file = uproot.open(fname)
    #get tree from file
    t = file["LArRecoND"]

    #pull out the branches you need: momenta in x,y,z and PDG code here
    branches = ["mcPx", "mcPy", "mcPz", "mcPDG"]

    #read data into arrays
    data = t.arrays(branches, library="ak")

    #for ease of use here, asign arrays as variable names
    px = data["mcPx"]
    py = data["mcPy"]
    pz = data["mcPz"]
    pdg = data["mcPDG"]

    #isolate electrons/positrons (with absolute value) and pions(+/-)
    e_only = abs(pdg) == 11
    pi_only = abs(pdg) == 211

    #get out the momenta in x,y,z for electrons and pions 
    e_px = px[e_only]
    e_py = py[e_only]
    e_pz = pz[e_only]

    pi_px = px[pi_only]
    pi_py = py[pi_only]
    pi_pz = pz[pi_only]

    #get total momentum 
    e_p = np.sqrt(e_px**2 + e_py**2 + e_pz**2)
    pi_p = np.sqrt(pi_px**2 + pi_py**2 + pi_pz**2)

    #normalise momentum
    e_xhat = e_px / e_p
    e_yhat = e_py / e_p
    e_zhat = e_pz / e_p

    pi_xhat = pi_px / pi_p
    pi_yhat = pi_py / pi_p
    pi_zhat = pi_pz / pi_p

    #combine electron x,y,z momentum and pion x,y,z momentum into one array each with awkward.zip
    e = ak.zip({"x": e_xhat, "y": e_yhat, "z": e_zhat})
    pi = ak.zip({"x": pi_xhat, "y": pi_yhat, "z": pi_zhat})

    #take cartesian product of two combined arrays to "pair up" electrons and pions within the same event
    #ensures that number of electrons and pions are the same and that each event that has them has one electron and one pion (theoretically)
    pairs = ak.cartesian([e, pi],axis=1, nested=True)

    #pull out the electron and pions components of the pairs
    e_pairs = pairs["0"]
    pi_pairs = pairs["1"]

    """
    #opening angle between momenta 
    cos_theta = (e_pairs["x"] * pi_pairs["x"] + e_pairs["y"] * pi_pairs["y"] + e_pairs["z"] * pi_pairs["z"])
    #force cos theta to lie between -1.0 and 1.0 (just in case)
    cos_theta_clipped = ak.where(cos_theta > 1.0, 1.0, ak.where(cos_theta < -1.0, -1.0, cos_theta))
    #get the angle using arccos 
    theta_e_pi = np.arccos(cos_theta_clipped)
    """

    #get theta: the angle between x and z for the pion and electron, do not need to clip as arctan2 does it automatically
    theta_pion = np.arctan2(pi_pairs["x"], pi_pairs["z"])
    #theta_pi_clip = ak.where(theta_pion > np.pi, np.pi, ak.where(theta_pion < -np.pi, -np.pi, theta_pion))
    theta_electron = np.arctan2(e_pairs["x"], e_pairs["z"])
    #theta_e_clip = ak.where(theta_electron > np.pi, np.pi, ak.where(theta_electron < -np.pi, -np.pi, theta_electron))

    #get the difference in angle 
    #dtheta = theta_pi_clip - theta_e_clip
    dtheta = theta_pion - theta_electron

    #get rid of nesting in the array and make it numpy array for plotting 
    dtheta = ak.flatten(dtheta, axis=None)
    dtheta = ak.to_numpy(dtheta)

    #plot dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(dtheta, bins=30, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\theta_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Truth d$\theta$ between e and $\pi$")
    figname = "theta_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(dtheta)}")
    plt.savefig(figname)
    plt.close()

    #get phi: the angle between y and z for the pion and electron, do not need to clip as arctan2 does it automatically
    phi_pion = np.arctan2(pi_pairs["y"], pi_pairs["z"])
    #phi_pi_clip = ak.where(phi_pion > np.pi, np.pi, ak.where(phi_pion < -np.pi, -np.pi, phi_pion))
    phi_electron = np.arctan2(e_pairs["y"], e_pairs["z"])
    #phi_e_clip = ak.where(phi_electron > np.pi, np.pi, ak.where(phi_electron < -np.pi, -np.pi, phi_electron))

    #get the difference in angle 
    #dphi = phi_pi_clip - phi_e_clip
    dphi = phi_pion - phi_electron


    #get rid of nesting in the array and make it numpy array for plotting 
    dphi = ak.flatten(dphi, axis=None)
    dphi = ak.to_numpy(dphi)

    #plot dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(dphi, bins=30, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\phi_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Truth d$\phi$ between e and $\pi$")
    figname = "phi_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(dphi)}")
    plt.savefig(figname)
    plt.close()
    

    #get rho: the angle between x and y for the pion and electron, do not need to clip as arctan2 does it automatically
    rho_pion = np.arctan2(pi_pairs["y"], pi_pairs["x"])
    #rho_pi_clip = ak.where(rho_pion > np.pi, np.pi, ak.where(rho_pion < -np.pi, -np.pi, rho_pion))
    rho_electron = np.arctan2(e_pairs["y"], e_pairs["x"])
    #rho_e_clip = ak.where(rho_electron > np.pi, np.pi, ak.where(rho_electron < -np.pi, -np.pi, rho_electron))

    #get the difference in angle 
    #drho = rho_pi_clip - rho_e_clip
    drho = rho_pion - rho_electron

    #get rid of nesting in the array and make it numpy array for plotting 
    drho = ak.flatten(drho, axis=None)
    drho = ak.to_numpy(drho)

    #plot dtheta 
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(drho, bins=30, histtype="step", linewidth=2)
    ax.set_xlabel(r"$\rho_{e\pi}$ (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Truth d$\rho$ between e and $\pi$")
    figname = "rho_e_pi.png"
    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(drho)}")
    plt.savefig(figname)
    plt.close()


    #plot all on the same set of axes
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(dtheta, bins=30, histtype="step", linewidth=2, label=r"$d\theta$")
    ax.hist(dphi, bins=30, histtype="step", linewidth=2, label=r"$d\phi$")
    ax.hist(drho, bins=30, histtype="step", linewidth=2, label=r"$d\rho$")
    ax.set_xlabel("Angle difference (rad)")
    ax.set_ylabel(r"e-$\pi$ pairs")
    ax.set_title(r"Truth angle differences between e and $\pi$")
    ax.legend()
    figname = "all_angles_e_pi.png"
    print(f"Saving {figname}...")
    plt.savefig(figname)
    plt.close()


    


rootTreeToDataFrame()    pi_px = px[pi_only]
    pi_py = py[pi_only]
    pi_pz = pz[pi_only]

    #get total momentum 
    e_p = np.sqrt(e_px**2 + e_py**2 + e_pz**2)
    pi_p = np.sqrt(pi_px**2 + pi_py**2 + pi_pz**2)

    #normalise momentum
    e_xhat = e_px / e_p
    e_yhat = e_py / e_p
    e_zhat = e_pz / e_p

    pi_xhat = pi_px / pi_p
    pi_yhat = pi_py / pi_p
    pi_zhat = pi_pz / pi_p

    #combine electron x,y,z momentum and pion x,y,z momentum into one array each with awkward.zip
    e = ak.zip({"x": e_xhat, "y": e_yhat, "z": e_zhat})
    pi = ak.zip({"x": pi_xhat, "y": pi_yhat, "z": pi_zhat})

    #take cartesian product of two combined arrays to "pair up" electrons and pions within the same event
    #ensures that number of electrons and pions are the same and that each event that has them has one electron and one pion (theoretically)
    pairs = ak.cartesian([e, pi],axis=1, nested=True)

    #pull out the electron and pions components of the pairs
    e_pairs = pairs["0"]
    pi_pairs = pairs["1"]

    #opening angle between momenta 
    """
    cos_theta = (e_pairs["x"] * pi_pairs["x"] + e_pairs["y"] * pi_pairs["y"] + e_pairs["z"] * pi_pairs["z"])
    """

    #force cos theta to lie between -1.0 and 1.0 (just in case)
    cos_theta_clipped = ak.where(cos_theta > 1.0, 1.0, ak.where(cos_theta < -1.0, -1.0, cos_theta))

    #get the angle using arccos 
    theta_e_pi = np.arccos(cos_theta_clipped)
    #get rid of nesting in the array and make it numpy array for plotting 
    theta_e_pi = ak.flatten(theta_e_pi, axis=None)
    theta_e_pi = ak.to_numpy(theta_e_pi)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.hist(theta_e_pi, bins=30, histtype="step", linewidth=2)

    ax.set_xlabel(r"$\theta_{e\pi}$ (rad)")
    ax.set_ylabel("e-pi pairs")
    ax.set_title("Truth e-pi Opening Angle")

    figname = "e_pi_angle.png"

    print(f"Saving {figname}...")
    print(f"Number of e-pi pairs: {len(theta_e_pi)}")

    plt.savefig(figname)
    plt.close()


rootTreeToDataFrame()
