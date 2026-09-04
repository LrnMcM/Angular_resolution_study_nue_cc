
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
mpl.rcParams['legend.fontsize']= 18
mpl.rcParams['axes.labelsize']= 16
mpl.rcParams['axes.titlesize']= 16
mpl.rcParams['xtick.labelsize']=16
mpl.rcParams['ytick.labelsize']=16

def rootTreeToDataFrame():

    fname="NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"
    #fname = "LArRecoND_1_999_Partially_Cheated.root"


    file = uproot.open(fname)

    treename = "LArRecoND" # our tree
    t  = file[treename]

    mcvars_mom = ['mcPx', 'mcPy', 'mcPz','mcPDG', 'mcNuE']

    mcDF = ak.to_dataframe(t.arrays(mcvars_mom, library="ak"))

    p = np.sqrt( mcDF["mcPx"]**2 + mcDF["mcPy"]**2 + mcDF["mcPz"]**2 )

    lime = "lime"
    red_color="orangered"
    blue_color="cornflowerblue"
    e_color="skyblue"
    e_mc_color="cornflowerblue"
    pi_color="hotpink"
    pi_mc_color="violet"


    # normalise truth momentum vars to get apples apples with others
    mcDF["tru_xhat"] = mcDF["mcPx"] / p
    mcDF["tru_yhat"] = mcDF["mcPy"] / p
    mcDF["tru_zhat"] = mcDF["mcPz"] / p
    
    hitvars = ['dirX', 'dirY', 'dirZ']
    trkvars = ['trkfitStartDirX', 'trkfitStartDirY', 'trkfitStartDirZ','trkfitEndDirX', 'trkfitEndDirY', 'trkfitEndDirZ']
    shwvars = ['shwrfitDirX', 'shwrfitDirY', 'shwrfitDirZ']
    
    recovars = hitvars + trkvars + shwvars

    recoDF = ak.to_dataframe(t.arrays(recovars, library="ak"))

    # rename the other vars too for consistency
    recoDF["int_xhat"] = recoDF['dirX']
    recoDF["int_yhat"] = recoDF['dirY']
    recoDF["int_zhat"] = recoDF['dirZ']

    recoDF["trk_xhat"] = recoDF['trkfitStartDirX']
    recoDF["trk_yhat"] = recoDF['trkfitStartDirY']
    recoDF["trk_zhat"] = recoDF['trkfitStartDirZ']

    recoDF["trke_xhat"] = recoDF['trkfitEndDirX']
    recoDF["trke_yhat"] = recoDF['trkfitEndDirY']
    recoDF["trke_zhat"] = recoDF['trkfitEndDirZ']

    recoDF["shw_xhat"] = recoDF['shwrfitDirX']
    recoDF["shw_yhat"] = recoDF['shwrfitDirY']
    recoDF["shw_zhat"] = recoDF['shwrfitDirZ']
    
    Energy_cut = 2.0

    DF = mcDF.join(recoDF)

    DF_electrons = DF[np.abs(DF.mcPDG)==11]  
    #DF = DF[DF.mcNuE > Energy_cut]


    enery_lower_quad_shwr = (DF_electrons["shw_xhat"] - DF_electrons["tru_xhat"]) >= 0.1
    enery_lower_quad_on_shwr = (DF_electrons["shw_xhat"] - DF_electrons["tru_xhat"]).between(-0.1, 0.1)
    energy_pass = DF_electrons.loc[enery_lower_quad_shwr, "mcNuE"]
    energy_pass_on = DF_electrons.loc[enery_lower_quad_on_shwr, "mcNuE"]

    print(f"Number of events passing cut: {len(energy_pass)}")


    plt.figure(figsize=(8,6))
    length_none= len(DF_electrons["mcNuE"])
    length_on= len(energy_pass_on)
    length_off= len(energy_pass)
    plt.hist(DF_electrons["mcNuE"], bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=lime, alpha = 0.3,  facecolor=lime, label=f"No Cut, n = {length_none}")
    plt.hist(energy_pass, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=red_color, alpha = 0.3,  facecolor=red_color, label=f"Off-Diagonal, n = {length_off}")
    plt.hist(energy_pass_on, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=blue_color, alpha = 0.3,  facecolor=blue_color, label=f"On-Diagonal, n = {length_on}")
    plt.hist(DF_electrons["mcNuE"], bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=lime, alpha = 1,  facecolor='none')
    plt.hist(energy_pass, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=red_color, alpha = 1,  facecolor='none')   
    plt.hist(energy_pass_on, bins=50, histtype="stepfilled", linewidth=2, density=True, edgecolor=blue_color, alpha = 1,  facecolor='none')
    plt.xlabel(r"$E_\nu$ (GeV)", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.legend(fontsize=16)
    plt.title(f"Energy distribution, selecting for e, fontsize=20")
    figname="Energy_combo_pions.png"
    plt.savefig(figname)
    print(f"Saving {figname}...")
    plt.clf()

   
    Energy_cut = DF_electrons[(DF_electrons["mcNuE"])>= 25.0]
    print(f"Number of events passing px cut: {len(Energy_cut)}")
    # Extract px values
    #px_true = Energy_cut["tru_xhat"]
    #px_shw = Energy_cut["shw_xhat"]
    px_true = DF_electrons["tru_xhat"]
    px_shw = DF_electrons["shw_xhat"]


    plt.figure(figsize=(8,6))
    plt.scatter(px_true, px_shw, s=20, alpha=0.4, color = e_color )
    xmin = min(px_true.min(), px_shw.min())
    xmax = max(px_true.max(), px_shw.max())
    plt.plot([xmin, xmax], [xmin, xmax], linestyle="--",color = "red", label=r"$\hat{p}_x^{\mathrm{shower}} = \hat{p}_x^{\mathrm{true}}$ ")
    n_electrons = len(DF_electrons)
    plt.scatter( [], [], label=f"Pions, n = {n_electrons}", color = e_color)
    plt.xlabel(r"MC $\hat{p}_x", fontsize=16)
    plt.ylabel(r"Shower $\hat{p}_x$", fontsize=16)
    plt.title(r"Shower VS Truth $\hat{p}_x$ for e, PC", fontsize=20)
    #plt.title(r"Shower x component vs True x component for Energy range 25+ GeV", fontsize=12)
    plt.legend()
    figname = "Shower_px_vs_True_px_cut.png"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saving {figname}...")
    plt.clf()


    electron_bins = [("0–5 GeV", 0, 5),("5–15 GeV", 5, 15),("15+ GeV", 15, np.inf)]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for ax, (label, emin, emax) in zip(axes, electron_bins):
        if np.isinf(emax):
            mask = DF_electrons["mcNuE"] >= emin
        else:
            mask = ((DF_electrons["mcNuE"] >= emin) & (DF_electrons["mcNuE"] < emax))
        df_bin = DF_electrons[mask]
        px_true = df_bin["tru_xhat"]
        px_shw = df_bin["shw_xhat"]
        n_electrons = len(df_bin)
        ax.scatter(px_true,px_shw,s=20,alpha=0.4,color=e_color,label=f"electron, n = {n_electrons}")
        if n_electrons > 0:
            xmin = min(px_true.min(), px_shw.min())
            xmax = max(px_true.max(), px_shw.max())
            ax.plot([xmin, xmax],[xmin, xmax],linestyle="--",color="red",label=r"$\hat{p}_x^{\mathrm{shower}} = \hat{p}_x^{\mathrm{true}}$")

        ax.set_xlabel(r"MC $\hat{p}_x$", fontsize=16)
        ax.set_ylabel(r"Shower $\hat{p}_x$", fontsize=16)
        ax.set_title(rf"$E_\nu$ = {label}, for e",fontsize=18)
        ax.legend(fontsize=12)
    plt.suptitle(r"Electron Shower $\hat{p}_x$ for different $E_\nu$ cuts, FC", fontsize=20)
    plt.tight_layout(pad=2.0, w_pad=2.5, rect=[0, 0, 1, 0.98])
    figname = "Electron_Shower_px_vs_True_px_EnergyBins.png"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saving {figname}...")
    plt.clf()


    DF_pions = DF[np.abs(DF["mcPDG"]) == 211]
    print(f"Number of pion events: {len(DF_pions)}")

    px_true = DF_pions["tru_xhat"]
    px_trk = DF_pions["trk_xhat"]

    plt.figure(figsize=(8,6))
    plt.scatter( px_true,px_trk,s=20,alpha=0.4, color= pi_color)
    xmin = min(px_true.min(), px_trk.min())
    xmax = max(px_true.max(), px_trk.max())
    plt.plot([xmin, xmax],[xmin, xmax],linestyle="--",color="red",label=r"$\hat{p}_x^{\mathrm{track\,start}} = \hat{p}_x^{\mathrm{true}}$")
    n_pions = len(DF_pions)
    plt.scatter( [], [], label=f"Pions, n = {n_pions}", color = pi_color)
    plt.xlabel(r"MC $\hat{p}_x$", fontsize=16)
    plt.ylabel(r"Track Start $\hat{p}_x$", fontsize=16)
    plt.title(r"Track Start vs Truth $\hat{p}_x$ for $\pi$, PC", fontsize=20)
    plt.legend()
    figname = "TrackStart_px_vs_True_px_pions.png"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saving {figname}...")
    plt.clf()

    pion_bins = [("0–10 GeV", 0, 10),("10–25 GeV", 10, 25),("25+ GeV", 25, np.inf)]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for ax, (label, pimin, pimax) in zip(axes, pion_bins):
        if np.isinf(pimax):
            mask = DF_pions["mcNuE"] >= pimin
        else:
            mask = ((DF_pions["mcNuE"] >= pimin) & (DF_pions["mcNuE"] < pimax))
        df_bin = DF_pions[mask]
        px_true = df_bin["tru_xhat"]
        px_shw = df_bin["trk_xhat"]
        n_pions = len(df_bin)
        ax.scatter(px_true,px_shw,s=20,alpha=0.4,color=pi_color,label=f"pions, n = {n_pions}")
        if n_pions > 0:
            xmin = min(px_true.min(), px_shw.min())
            xmax = max(px_true.max(), px_shw.max())
            ax.plot([xmin, xmax],[xmin, xmax],linestyle="--",color="red",label=r"$\hat{p}_x^{\mathrm{track}} = \hat{p}_x^{\mathrm{true}}$")

        ax.set_xlabel(r"MC $\hat{p}_x$", fontsize=16)
        ax.set_ylabel(r"Track $\hat{p}_x$", fontsize=16)
        ax.set_title(rf"$E_\nu$ = {label}, for $\pi$",fontsize=18)
        ax.legend(fontsize=12)
    plt.suptitle(r"Pion Track $\hat{p}_x$ for different $E_\nu$ cuts, FC", fontsize=20)
    plt.tight_layout(pad=2.0, w_pad=2.5, rect=[0, 0, 1, 0.98])
    figname = "Pion_Track_px_vs_True_px_EnergyBins.png"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saving {figname}...")
    plt.clf()


rootTreeToDataFrame()
