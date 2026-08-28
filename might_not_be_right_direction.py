import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt


def angular_diff(a, b):
    #get smallest angular difference
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def pair_up(data, x_branch, y_branch, z_branch):
    px = data[x_branch]
    py = data[y_branch]
    pz = data[z_branch]
    pdg = data["mcPDG"]

    # Select electrons and pions
    electron_selection = abs(pdg) == 11
    pion_selection = abs(pdg) == 211

    #select electron momenta
    e_px = px[electron_selection]
    e_py = py[electron_selection]
    e_pz = pz[electron_selection]

    #select pion momenta
    pi_px = px[pion_selection]
    pi_py = py[pion_selection]
    pi_pz = pz[pion_selection]

    #get momentum
    e_p = np.sqrt(e_px**2 + e_py**2 + e_pz**2)
    pi_p = np.sqrt(pi_px**2 + pi_py**2 + pi_pz**2)

    #combine into "electron" and "pion array"
    e = ak.zip({"x": e_px / e_p, "y": e_py / e_p, "z": e_pz / e_p})
    pi = ak.zip({"x": pi_px / pi_p, "y": pi_py / pi_p,"z": pi_pz / pi_p})
    #pairs up e and pi so they are 1:1
    pairs = ak.cartesian([e, pi],axis=1,nested=True)
    return pairs


def get_angle_differences(pairs):
    # Electron = pairs["0"]
    # Pion     = pairs["1"]

    # theta: x-z plane
    theta_e = np.arctan2(pairs["0"]["x"], pairs["0"]["z"])
    theta_pi = np.arctan2(pairs["1"]["x"], pairs["1"]["z"])

    # phi: y-z plane
    phi_e = np.arctan2(pairs["0"]["y"], pairs["0"]["z"])
    phi_pi = np.arctan2(pairs["1"]["y"], pairs["1"]["z"])

    # rho: x-y plane
    rho_e = np.arctan2(pairs["0"]["y"], pairs["0"]["x"])
    rho_pi = np.arctan2(pairs["1"]["y"], pairs["1"]["x"])

    #calls above function to get smallest angular difference
    dtheta = angular_diff(theta_pi, theta_e)
    dphi = angular_diff(phi_pi, phi_e)
    drho = angular_diff(rho_pi, rho_e)

    return dtheta, dphi, drho


def get_all_angles(data):
    #load in some data branches 
    angle_branches = {
        "MC": ("mcPx", "mcPy","mcPz"),
        "Shower": ("shwrfitDirX", "shwrfitDirY", "shwrfitDirZ"),
        "Interface": ("dirX", "dirY", "dirZ"),
        "TrackStart": ("trkfitStartDirX", "trkfitStartDirY", "trkfitStartDirZ"),
        "TrackEnd": ("trkfitEndDirX", "trkfitEndDirY","trkfitEndDirZ")}
    angles = {}

    for name, branches in angle_branches.items():
        #just sets up the "pair" array to gave the name and x,y,z
        pairs = pair_up(data, branches[0], branches[1], branches[2])
        #get the angle difference
        dtheta, dphi, drho = get_angle_differences(pairs)
        #labels them
        angles[name] = {"dtheta": dtheta, "dphi": dphi, "drho": drho}
    return angles

def cut_fctn(angles, cut=1.5):
    #constrain the angle difference between the hadron and the electron to be less than a cut value: with the goal to compare
    #truth and reco angular separation profules for small angles 
    #CHANGE CUT VALUE HERE
    apply_cut = ((abs(angles["MC"]["dtheta"]) < cut) & (abs(angles["MC"]["dphi"]) < cut) & (abs(angles["MC"]["drho"]) < cut))
    apply_cut = apply_cut & ((abs(angles["Shower"]["dtheta"]) < cut) & (abs(angles["Shower"]["dphi"]) < cut) & (abs(angles["Shower"]["drho"]) < cut))
    apply_cut = apply_cut & ((abs(angles["Interface"]["dtheta"]) < cut) & (abs(angles["Interface"]["dphi"]) < cut) & (abs(angles["Interface"]["drho"]) < cut))
    apply_cut = apply_cut & ((abs(angles["TrackStart"]["dtheta"]) < cut) & (abs(angles["TrackStart"]["dphi"]) < cut) & (abs(angles["TrackStart"]["drho"]) < cut))
    apply_cut = apply_cut & ((abs(angles["TrackEnd"]["dtheta"]) < cut) & (abs(angles["TrackEnd"]["dphi"]) < cut) & (abs(angles["TrackEnd"]["drho"]) < cut))
    #apply cut
    for name in angles:
        angles[name]["dtheta"] = angles[name]["dtheta"][apply_cut]
        angles[name]["dphi"] = angles[name]["dphi"][apply_cut]
        angles[name]["drho"] = angles[name]["drho"][apply_cut]
    return angles

def flatten_array(angles):
    #gets rid of nesting 
    for name in angles:
        angles[name]["dtheta"] = ak.to_numpy(ak.flatten(angles[name]["dtheta"], axis=None))
        angles[name]["dphi"] = ak.to_numpy(ak.flatten(angles[name]["dphi"], axis=None))
        angles[name]["drho"] = ak.to_numpy(ak.flatten(angles[name]["drho"], axis=None))
    return angles

def plot_mc_reco(truth, reco, angle_name, reco_name, cut=1.5):
    #sets up plotting for ease later
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    angle_symbols = {"theta": r"\theta","phi": r"\phi","rho": r"\rho",}
    ax.scatter(truth, reco, s=8, alpha=0.4)
    ax.plot([-cut, cut], [-cut, cut], linestyle="--", linewidth=2, label="reco=mc")
    ax.set_xlim(-cut, cut)
    ax.set_ylim(-cut, cut)
    ax.set_xlabel(rf"Truth $\Delta {angle_symbols[angle_name]}$ (rad)")
    ax.set_ylabel(rf"{reco_name} $\Delta {angle_symbols[angle_name]}$ (rad)")
    ax.set_title(rf"Truth vs {reco_name}: $\Delta {angle_symbols[angle_name]}$")
    ax.legend()
    figname = (f"truth_vs_{reco_name.lower()}_d{angle_name}.png")
    print(f"Saving {figname}...")
    plt.savefig(figname,dpi=300)
    plt.close()


def apply_all_function(data, cut=1.5):
    angles = get_all_angles(data)
    angles = cut_fctn(angles,cut=cut)
    angles = flatten_array(angles)
    print(f"\nNumber of e-pi pairs after common, angle cut = {len(angles['MC']['dtheta'])}")
    truth = angles["MC"]
    shower = angles["Shower"]
    interface = angles["Interface"]
    trackstart = angles["TrackStart"]
    trackend = angles["TrackEnd"]
    plot_mc_reco(truth["dtheta"], shower["dtheta"], "theta", "Shower", cut)
    plot_mc_reco(truth["dphi"], shower["dphi"], "phi", "Shower", cut)
    plot_mc_reco(truth["drho"], shower["drho"], "rho", "Shower", cut)
    plot_mc_reco(truth["dtheta"], interface["dtheta"], "theta", "Interface", cut)
    plot_mc_reco(truth["dphi"], interface["dphi"], "phi", "Interface", cut)
    plot_mc_reco(truth["drho"], interface["drho"], "rho", "Interface", cut)
    plot_mc_reco(truth["dtheta"], trackstart["dtheta"], "theta", "TrackStart", cut)
    plot_mc_reco(truth["dphi"], trackstart["dphi"], "phi", "TrackStart", cut)
    plot_mc_reco(truth["drho"], trackstart["drho"], "rho", "TrackStart",cut)
    plot_mc_reco(truth["dtheta"], trackend["dtheta"], "theta", "TrackEnd", cut)
    plot_mc_reco(truth["dphi"], trackend["dphi"], "phi", "TrackEnd", cut)
    plot_mc_reco(truth["drho"], trackend["drho"], "rho", "TrackEnd", cut)
    return angles

def rootTreeToDataFrame():
    fname = "NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"
    file = uproot.open(fname)
    t = file["LArRecoND"]
    branches = ["mcPx", "mcPy", "mcPz", "mcPDG", "shwrfitDirX", "shwrfitDirY", "shwrfitDirZ", "dirX", "dirY", "dirZ", "trkfitStartDirX", "trkfitStartDirY", "trkfitStartDirZ", "trkfitEndDirX", "trkfitEndDirY", "trkfitEndDirZ"]
    data = t.arrays(branches, library="ak")
    angles = apply_all_function(data, cut=1.5)
    return angles


rootTreeToDataFrame()
