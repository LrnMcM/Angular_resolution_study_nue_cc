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
