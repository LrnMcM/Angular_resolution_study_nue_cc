
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

    mcvars_mom = ['mcPx', 'mcPy', 'mcPz','mcPDG', 'mcNuE']

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
    DF = DF[np.abs(DF.mcPDG)==211]  
    #DF = DF[DF.mcNuE > Energy_cut]

    print(f"DF before filtering: {DF}")

    enery_lower_quad = (DF["shw_xhat"] - DF["tru_xhat"]) >= 0.1
    enery_lower_quad_on = (DF["shw_xhat"] - DF["tru_xhat"]).between(-0.1, 0.1)
    energy_pass = DF.loc[enery_lower_quad, "mcNuE"]
    energy_pass_on = DF.loc[enery_lower_quad_on, "mcNuE"]

    print(f"Number of events passing cut: {len(energy_pass)}")

    print(f"DF: {DF}")

    plt.figure(figsize=(8,6))
    plt.hist(energy_pass_on, bins=20,alpha=0.5)
    plt.xlabel(r"$E_\nu$ (GeV)", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title("Energy distribution on diagonal + pion", fontsize=20)
    plt.savefig("Energy_after_px_cut.png")
    plt.clf()

    #DF = mcDF.join(recoDF)

    print(f"DF before cut: {DF}")

    plt.figure(figsize=(8,6))
    plt.hist(DF["mcNuE"], bins=20,alpha=0.5)
    plt.xlabel(r"$E_\nu$ (GeV)", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title(r"Energy distribution no cuts + electron", fontsize=20)
    plt.savefig("Energy_before_px_cut.png")
    plt.clf()


    plt.figure(figsize=(8,6))

    plt.hist(energy_pass, bins=20, histtype="step", linewidth=2, alpha=0.8, density=True, label="Off-diag")
    plt.hist(energy_pass_on, bins=20, histtype="step", linewidth=2, alpha=0.8,density=True, label="On-diag")
    plt.hist(DF["mcNuE"], bins=20, histtype="step", linewidth=2, alpha=0.8, density=True, label="No cut")

    plt.xlabel(r"$E_\nu$ (GeV)", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title("Energy comparison electron cut applied", fontsize=20)
    plt.legend(fontsize=14)

    plt.savefig("Energy_comparison_px_cut.png")
    plt.clf()

    #enery_lower_quad = (DF["shw_xhat"] - DF["tru_xhat"]).between(-0.25, 0.25)

    #DF_cut = DF[(DF["shw_xhat"] - DF["tru_xhat"]) >= 0.1]
    #DF_cut = DF[(DF["shw_xhat"] - DF["tru_xhat"]).between(-0.1,0.1)]
    #DF_cut = DF[(DF["mcNuE"]).between(10.0,25.0)]
    DF_cut = DF[(DF["mcNuE"])>= 25.0]

    print(f"Number of events passing px cut: {len(DF_cut)}")


    # Extract px values
    px_true = DF_cut["tru_xhat"]
    px_shw = DF_cut["shw_xhat"]


    plt.figure(figsize=(8,6))
    plt.scatter(px_true, px_shw, s=20, alpha=0.4)
    xmin = min(px_true.min(), px_shw.min())
    xmax = max(px_true.max(), px_shw.max())
    plt.plot([xmin, xmax], [xmin, xmax], linestyle="--",color = "red", label=r"$\hat{p}_x^{\mathrm{shower}} = \hat{p}_x^{\mathrm{true}}$ ")
    #plt.plot([xmin, xmax], [xmin + 0.1, xmax + 0.1], linestyle=":",color = "red", label=r"$\hat{p}_x^{\mathrm{shower}} = \hat{p}_x^{\mathrm{true}} + 0.1$")
    #plt.plot([xmin, xmax], [xmin - 0.1, xmax - 0.1], linestyle=":",color = "red")
    plt.xlabel(r"True x component", fontsize=16)
    plt.ylabel(r"Shower x component", fontsize=16)
    plt.title(r"Shower x component vs True x component for Energy range 25+ GeV", fontsize=12)
    plt.legend()
    plt.savefig("Shower_px_vs_True_px_cut.png")
    plt.clf()


    '''
    DF_cut = DF[(DF["shw_xhat"] - DF["tru_xhat"]) >= 0.25]
    print(f"Number of events passing px cut: {len(DF_cut)}")
    p_tru = np.sqrt(DF_cut["tru_xhat"]**2 + DF_cut["tru_yhat"]**2 + DF_cut["tru_zhat"]**2)
    p_shwr = np.sqrt( DF_cut["shw_xhat"]**2 + DF_cut["shw_yhat"]**2 + DF_cut["shw_zhat"]**2)
    plt.figure(figsize=(8,6))

    plt.scatter(p_tru, p_shwr, s=20, alpha=0.4)
    #plt.plot([p_tru.min(), p_tru.max()],[p_tru.min(), p_tru.max()],linestyle="--")
    plt.xlabel(r"True $\hat{p}$", fontsize=16)
    plt.ylabel(r"Shower $\hat{p}$", fontsize=16)
    plt.title(r"Shower direction magnitude vs True direction magnitude ($\Delta p_x \geq 0.25$)", fontsize=16)
    plt.savefig("Shower_p_vs_True_p.png")
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.hist(energy_pass, bins=20,alpha=0.5)
    plt.xlabel(r"$E_\nu$ (GeV)", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title("Energy distribution after px cut", fontsize=20)
    plt.savefig("Energy_after_px_cut.png")
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.hist(DF["mcNuE"], bins=20,alpha=0.5)
    plt.xlabel(r"$E_\nu$ (GeV)", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title("Energy distribution before px cut", fontsize=20)
    plt.savefig("Energy_before_px_cut.png")
    plt.clf()


    plt.figure(figsize=(8,6))
    plt.hist(DF["mcPDG"], bins=20,alpha=0.5)
    plt.xlabel("pdg", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.title("pdg", fontsize=20)
    plt.savefig("pdg.png")
    plt.clf()
  
    plt.figure(figsize=(8,6))

    plt.hist2d(DF["mcPDG"],DF["mcNuE"], bins=[50,50])
    plt.xlabel("PDG number", fontsize=16)
    plt.ylabel(r"$E_\nu$ (GeV)", fontsize=16)
    plt.title("Energy vs PDG number", fontsize=20)

    plt.colorbar(label="Events")

    plt.savefig("Energy_vs_PDG_2D.png")
    plt.clf()
    '''  

    DF = DF.filter(like='hat')


    kinds = ['tru', 'int', 'trk', 'trke','shw']
    titles={'tru':'Truth Direction', 'int':'Interface Direction', 'trk': 'Track Fit Start Direction', 'trke': 'Track Fit End Direction',  'shw':'Shower Fit Direction' }

    for i,k in enumerate(kinds):

        print(f"Doing {kinds[i]}")
        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        figname_xy = k+"_xy.png"


        kDF = DF.filter(like=k)

        x = kDF.filter(like='_xhat').to_numpy()
        y = kDF.filter(like='_yhat').to_numpy()
        z = kDF.filter(like='_zhat').to_numpy()

        colors = cm.ScalarMappable(cmap='twilight')
        colors.set_array(z)

        plt.scatter(x,y, s=30, c=z, cmap='twilight', alpha=0.4)
    
        ax.set_xlabel(r"$\hat{p}_x$", fontsize=16)
        ax.set_ylabel(r"$\hat{p}_y$", fontsize=16)

        
        cbar = plt.colorbar(colors,ax=ax)
        cbar.ax.set_ylabel(r"$\hat{p}_z$", rotation=90, fontsize=16,labelpad=20)
        ax.set_title(titles[k], fontsize=20)

        print(f"  * {figname_xy} saving...")
        plt.savefig(figname_xy)
        plt.clf()


    # --------------------------------------------------------------------
    # Plots of true versus int,shw,trk for x,y,z separately
    # --------------------------------------------------------------------
    dirs = ['x','y','z']
    k1='tru'
    kinds = ['int', 'trk', 'trke', 'shw']
    titlesK={'tru':'Truth ', 'int':'Interface ', 'trk': 'Track Fit Start ', 'trke': 'Track Fit End ', 'shw':'Shower Fit ' }
    titlesV={'x': r'$\hat{p}_x$', 'y': r'$\hat{p}_y$','z': r'$\hat{p}_z$'}
    
    for v in dirs:

        for k2 in kinds:

            print(f"Doing {k2}")
            figname= f"{v}_{k1}_{k2}.png"

            v1 = DF.filter(like=f'{k1}_{v}hat').to_numpy()
            v2 = DF.filter(like=f'{k2}_{v}hat').to_numpy()

            fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

            plt.scatter(v1, v2, s=50, c='rebeccapurple',  alpha=0.2)
        
            ax.set_xlabel(f"{titlesK[k1]+titlesV[v]}", fontsize=16)
            ax.set_ylabel(f"{titlesK[k2]+titlesV[v]}", fontsize=16)

    
            #ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f"{titlesK[k1]} versus {titlesK[k2]}: {v} ", fontsize=20)

            print(f"  * {figname} saving...")
            plt.savefig(figname)



    # --------------------------------------------------------------------
    # 1D Plots of angle wrt z-axis for tru, int, shw, trk on same axes
    # 3 plots: solid angle, angle_xz_plane, angle_yz_plane
    # --------------------------------------------------------------------

    def angles(px,py):
        pT = np.sqrt(px**2 + py**2)

        return np.asin(px), np.asin(py), np.asin(pT)


    figA, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), layout='constrained')

    figname="Angles.png"
    
    kinds = ['tru', 'int', 'trk', 'trke' , 'shw']
    klabels=['Truth', 'Interface', 'TrackFitStart','TrackFitEnd','ShowerFit']
    styles=['-','--',':','-',':-']
    cols = ['black', 'rebeccapurple','blue',  'silver','hotpink']
    
    for i,k in enumerate(kinds):
        kDF = DF.filter(like=k)

        ffs = cols[i]

        print(f"Doing {kinds[i]} with style {styles[i]} and color {ffs}")


        x = kDF.filter(like='_xhat').to_numpy()
        y = kDF.filter(like='_yhat').to_numpy()

        # for 1D histogram of angles 
        theta_xz, theta_yz, theta_solid = angles(x,y)

        theta_solid=np.array(theta_solid).ravel()
        theta_xz=np.array(theta_xz).ravel()
        theta_yz=np.array(theta_yz).ravel()



        print(f"type(theta_solid) : {type(theta_solid)}")
        print(f"theta_solid.shape : {theta_solid.shape}")
        print(f"theta_solid : {theta_solid}")
        # solid angle
        a0=axs[0]
        
        a0.hist(theta_solid, histtype='stepfilled', bins=18, alpha=0.05, color=cols[i])
        a0.hist(theta_solid, histtype='step', bins=18,lw=2, ls=styles[i], label=klabels[i], color=cols[i])
        a0.set_xlabel(r"$\theta$ (rad)", fontsize=16)
        a0.set_title("Angle in z,T plane", fontsize=20)
        
        a0.legend()

        # xz
        a1=axs[1]
        a1.hist(theta_xz, histtype='stepfilled', bins=18,alpha=0.05, color=cols[i])
        a1.hist(theta_xz, histtype='step', bins=18,lw=2, ls=styles[i], label=klabels[i], color=cols[i])
        a1.set_xlabel(r"$\theta_{zx}$ (rad)", fontsize=16)
        a1.set_title("Angle in z,x plane", fontsize=20)


        # yz
        a2=axs[2]
        a2.hist(theta_yz, histtype='stepfilled', bins=18,alpha=0.05, color=cols[i])
        a2.hist(theta_yz, histtype='step', bins=18,lw=2, ls=styles[i], label=klabels[i], color=cols[i])
        a2.set_xlabel(r"$\theta_{zy}$ (rad)", fontsize=16)
        a2.set_title("Angle in z,y plane", fontsize=20)

    print(f"  * {figname} saving...")
    plt.savefig(figname)
    plt.clf()




rootTreeToDataFrame()
