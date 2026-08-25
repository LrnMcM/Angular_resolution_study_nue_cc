
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#mpl.use('Agg')
#mpl.rcParams['agg.path.chunksize'] = 20000000
mpl.rcParams['legend.fontsize']= 18
mpl.rcParams['axes.labelsize']= 16
mpl.rcParams['axes.titlesize']= 16
mpl.rcParams['xtick.labelsize']=16
mpl.rcParams['ytick.labelsize']=16

def rootTreeToDataFrame():

    fname="NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"

    file = uproot.open(fname)

    treename = "LArRecoND" # our tree
    t  = file[treename]

    mcvars_mom = ['mcPx', 'mcPy', 'mcPz','mcPDG', 'mcNuE', 'clusterId', 'nUHits']

    mcDF = ak.to_dataframe(t.arrays(mcvars_mom, library="ak"))

    p = np.sqrt( mcDF["mcPx"]**2 + mcDF["mcPy"]**2 + mcDF["mcPz"]**2 )

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

    print(f"mcDF: {mcDF.shape}")
    print(f"recoDF: {recoDF.shape}")
    
    Energy_cut = 2.0

    DF = mcDF.join(recoDF)	

    print(f"DF: {DF}")

    #DF = DF[np.abs(DF.mcPDG)==211]  
    DF = DF[DF.mcNuE > Energy_cut]
    
    print(f"DF before filtering: {DF}")

    enery_lower_quad = (DF["shw_xhat"] - DF["tru_xhat"]) >= 0.1
    enery_lower_quad_on = (DF["shw_xhat"] - DF["tru_xhat"]).between(-0.1, 0.1)
    energy_pass = DF.loc[enery_lower_quad, "mcNuE"]
    energy_pass_on = DF.loc[enery_lower_quad_on, "mcNuE"]

    print(f"Number of events passing cut: {len(energy_pass)}")

    print(f"DF: {DF}")


    #enery_lower_quad = (DF["shw_xhat"] - DF["tru_xhat"]).between(-0.25, 0.25)

    #DF_cut = DF[(DF["shw_xhat"] - DF["tru_xhat"]) >= 0.1]
    #DF_cut = DF[(DF["shw_xhat"] - DF["tru_xhat"]).between(-0.1,0.1)]
    #DF_cut = DF[(DF["mcNuE"]).between(10.0,25.0)]
    #DF_cut = DF[(DF["mcNuE"])>= 25.0]

    #print(f"Number of events passing px cut: {len(DF_cut)}")


    DF_cut_clusterId_1_2 = DF[DF["clusterId"].between(1.0, 2.0)]
    DF_cut_clusterId_3_9 = DF[DF["clusterId"].between(3.0, 9.0)]
    DF_cut_clusterId_10 = DF[(DF["clusterId"])>= 10.0]


    #px_true = DF_cut["tru_xhat"]
    #px_shw = DF_cut["shw_xhat"]
    px_true = DF_cut_clusterId_1_2["tru_xhat"]
    px_shw = DF_cut_clusterId_1_2["shw_xhat"]
    #n_clusters = DF_cut_clusterId["clusterId"]
    n_u_hits_1_2 = DF_cut_clusterId_1_2["nUHits"]
    n_u_hits_3_9 = DF_cut_clusterId_3_9["nUHits"]
    n_u_hits_10 = DF_cut_clusterId_10["nUHits"]

    plt.figure(figsize=(8,6))
    plt.hist(n_u_hits_10, bins=20,alpha=0.5)
    plt.xlabel(r"$E_\nu$ (GeV)", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title("Energy distribution on diagonal + pion", fontsize=20)
    plt.savefig("clusterId_cut_by_number.png")
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.hist(n_u_hits_10, bins=20,alpha=0.5)
    plt.xlabel(r"nUHits", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title("Number of Hits on U plane with Cluster Cut: 10+ ", fontsize=20)
    plt.savefig("number_u_hits_cluster_cut.png")
    plt.clf()


    plt.figure(figsize=(8,6))
    plt.hist(n_u_hits_1_2, bins=20, histtype="step", linewidth=2, alpha=0.8, density=True, label="1 and 2")
    plt.hist(n_u_hits_3_9, bins=20, histtype="step", linewidth=2, alpha=0.8, density=True, label="3 to 9")
    plt.hist(n_u_hits_10, bins=20, histtype="step", linewidth=2, alpha=0.8,density=True, label="10+")
    plt.xlabel(r"nUHits", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title(" Number of hits on U plane for different ClusterIds", fontsize=20)
    plt.legend(fontsize=14)
    plt.savefig("Comparaison_numberUhits.png")
    plt.clf()


    plt.figure(figsize=(8,6))
    plt.scatter(px_true, px_shw, s=20, alpha=0.4)
    #xmin = min(px_true.min(), px_shw.min())
    #xmax = max(px_true.max(), px_shw.max())
    #plt.plot([xmin, xmax], [xmin, xmax], linestyle="--",color = "red", label=r"$\hat{p}_x^{\mathrm{shower}} = \hat{p}_x^{\mathrm{true}}$ ")
    #plt.plot([xmin, xmax], [xmin + 0.1, xmax + 0.1], linestyle=":",color = "red", label=r"$\hat{p}_x^{\mathrm{shower}} = \hat{p}_x^{\mathrm{true}} + 0.1$")
    #plt.plot([xmin, xmax], [xmin - 0.1, xmax - 0.1], linestyle=":",color = "red")
    plt.xlabel(r"True x component", fontsize=16)
    plt.ylabel(r"Shower x component", fontsize=16)
    plt.title(r"Shower x component vs True x component for Energy range 25+ GeV", fontsize=12)
    #plt.legend()
    plt.savefig("Shower_px_vs_True_cluster_id.png")
    plt.clf()


rootTreeToDataFrame()
