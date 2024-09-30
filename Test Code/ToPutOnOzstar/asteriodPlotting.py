"""
Plotting functions for asteroid analysis

Import each function as needed, keeps other code for being a mess with plotting everywhere, and can be much more optional.

All have some shared optional inputs

    saving: bool
        whether or not to save the file as a pdf, default is false
    
    savePath: str
        /path/to/image/  the will be  called fName.pdf. trailing slash is needed
    
    nameChange: str
        String to add to the file name. The fName of each function already has a decription of the plot, but if this wants to be changed this can be done here. Default is '', a blank str 

BL 30/9/24
"""


import numpy as np
import numpy.typing as npt
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.cm as mpcm

import itertools
from scipy import ndimage

from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time

from astroquery.jplhorizons import Horizons
from photutils.aperture import CircularAperture

plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})



def _name_cut(df, name:str, colName:str="Name"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn



def plotAEI(propDF:pd.DataFrame, saving:bool=False, savePath:str="../Figures/", nameChange=""):
    ps=2
    col="tab:blue"
    fig, ax = plt.subplots(1,2, sharex=True, figsize = (12,6))
    ax[0].set(xlabel=f"a [Au]", ylabel="e")
    ax[1].set(xlabel=f"a [Au]", ylabel=f"i [^{{\circ}}]")
    ax[0].scatter(propDF["a"], propDF["e"], c=col,s=ps)
    ax[0].scatter(propDF["a"], propDF["i"], c=col,s=ps)
    fig.tight_layout()

    if saving:
        fig.savefig(f"{savePath}AEIplot{nameChange}.pdf")

    return fig



def plotExpectedPos(posDf: pd.DataFrame, timeList: npt.ArrayLike, targetPos: list, magLim: float = 20.0, scaleAlpha: bool = False, minLen: int = 0, saving:bool=False, hsList:list = None, cutList=[22,1,3,7], savePath:str="../Figures/", nameChange="") -> plt.Figure:
    """
    Takes the output of a query to SkyBot, sets up a WCS, and plots the objects position, coloured by time, in Ra/Dec and ecLon/ecLat space. 

    Inputs
        posDf: DataFrame
            The positions of each object with the associated time. Should have structure of output of querySB  
        timeList: arrayLike of Times
            astropy.time array like of the times of queries, also in 'epoch' collumn in posDf, but makes sure that the whole queried time range is acounted for, instead of maybe nothing was found at one time.
        targetPos: list of len=3, [float, float, Time]
            list of initial ra (float), dec (float) and time (astropy.Time value), the center point of the query and the first time of observations 

        magLim: float
            Limiting magnitude of the query, default 20.0 
        scaleAlpha: bool
            If alpha of the symbols should be scaled to the average brigtness of the object
        minLen: int
            Minimum number of hits on each object for it to be plotted, default 0
        saving: bool
            If saving the figure is required, default False
    Output
        fig: Figure
            The figure of the objects position, coloured by time, in Ra/Dec and ecLon/ecLat space. Faint grey lines connect the points to guide the eye.
    """
    sector, cam, ccd, cut = cutList
    ra_i, dec_i, t_i = targetPos  # gets _i values from the input list

    w = wcs.WCS(naxis=2)  # ? sets up blank WCS

    # setting FITS keywords #!guessing what they mean
    w.wcs.crpix = [0, 0]
    w.wcs.crval = [posDf["RA"].min(), posDf["Dec"].min()]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.mjdref = [posDf["MJD"].mean(), 0]

    # proj. things #! I don't understand
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])

    # Set up figure with WCS
    # fig = plt.figure(figsize=(12, 12))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection=w)
    ax.grid(color='black', ls='solid')
    ax.set_xlabel("RA")
    ax.coords[0].set_format_unit(u.deg)
    ax.set_ylabel("DEC")

    # Set up overlay coords
    overlay = ax.get_coords_overlay('geocentrictrueecliptic')
    overlay.grid(color='black', ls='dotted')
    overlay[0].set_axislabel('ecLon')
    overlay[1].set_axislabel('ecLat')

    cmap = "plasma"
    # The delta T of the data, and makes a norm for a cmap
    cnorm = mplc.Normalize(np.min(timeList), np.max(timeList))
    markers = itertools.cycle((".", "x", "^", "*", "D", "+", "v", "o", "<", ">", "H", "1", "2",
                              "3", "4", "8", "X", "p", "d", "s", "P", "h"))  # lots of symbols so they aren't repeated much

    unqNames = pd.unique(posDf["Name"])
    badNames = []  # stops index problem later
    if scaleAlpha:
        # setup for possible alpha scaling
        scaleMult = 0.90  # changes how small alpha can get
        brightest = posDf["Mv"].min()  # * Mag. scale is backwards...
        deltaMag = magLim-brightest
        
        if hsList == None:
            hs = {}
            for name in unqNames:
                try:
                    horizQ =Horizons(id = name, epochs = t_i.jd, location= "500@10")
                    hs[name]=float(horizQ.elements()['H'])
                except Exception as e:
                    print(f"Name= {name} didn't work, error {str(e)}")#some names aren't in jpl?
                    badNames.append(name)

            hsList = list(hs.values())
        
        # when alpha scale is by H
        brightest = np.min(hsList)
        deltaMag = np.max(hsList)-brightest

        scaleStr = f"Alpha Scaling by $H$:\n$H$ = {brightest}, $\\alpha$ = 1\n$H$= {deltaMag+brightest}, $\\alpha$ = {round((1-scaleMult),1)}"

    for i, name in enumerate(unqNames):  # does each name seperatly
        if name in badNames:
            continue
        nameIDs = posDf.index[posDf["Name"] == name]

        #*should have x,y anyway
        # coords = SkyCoord(posDf.loc[nameIDs]["RA"],
        #                   posDf.loc[nameIDs]["Dec"], unit=(u.deg, u.deg))
        # pixels = coords.to_pixel(w)

        pixels=(posDf["X"], posDf["Y"])


        # scales each objects alpha

        if scaleAlpha:
            avgMag = hsList[i]
            # scales the alpha of plotting to the brightness of the object, to give some idea of what might be detected
            alpha = (1-scaleMult*((avgMag-brightest)/deltaMag))
        else:
            alpha = 1

        # plotting
        if len(nameIDs) > minLen:
            # plots line to help eye track objects
            ax.plot(pixels[0], pixels[1], c="grey", alpha=0.3)
            # scatters the pixel coords on wsc set axis, uses different symbol for each, with cmap scaled by the time in the sector.
            ax.scatter(pixels[0], pixels[1], marker=next(markers),s=5, c=(
                posDf.loc[nameIDs]["MJD"]), cmap=cmap, norm=cnorm, label=name, alpha=alpha)
    clb = fig.colorbar(mpcm.ScalarMappable(
        norm=cnorm, cmap=cmap), ax=ax, pad=0.1)
    clb.ax.set_title("Time in MJD", fontsize=16, y=1.03)
    clb.ax.ticklabel_format(useOffset=False)
    centerPix = SkyCoord(ra_i, dec_i, unit=(u.deg, u.deg)).to_pixel(w)  # center of seach area in pixels
    ax.scatter(centerPix[0], centerPix[1], marker="+",
               s=100, c="g")  # center marker
    
    #*BOX to check
    # corners = np.array([[ra_i+1.6, dec_i+1.6],[ra_i-1.6, dec_i+1.6],[ra_i-1.6, dec_i-1.6],[ra_i+1.6, dec_i-1.6],[ra_i+1.6, dec_i+1.6]])

    # cornersPix =  SkyCoord(corners, unit=(u.deg, u.deg)).to_pixel(w) 

    # ax.plot(cornersPix[0], cornersPix[1])


    if scaleAlpha: ax.text(-25, 3, s=scaleStr, fontsize=12)

    fig.tight_layout()

    if saving: fig.savefig(f"{savePath}interpPos_{sector}_{cam}_{ccd}_{cut}{nameChange}.png") #!CHECK
    return fig #retunrs instead of saves, #? is this useful?
    # fig.savefig(f"./ecCoordsandAlphascaled_ra{ra_i}_dec{dec_i}_t{t_i.mjd}_Mv{magLim}.png") #!CHECK



def recoveredHist(propDF, bkgLim, saving:bool=False, savePath:str="../Figures/", nameChange=""):
    """Takes the mean and deviation of the fluxes, and compares to a background Limit. A histogram of those greater than the limit is made. If no asteroid is in a mag bin, then 'all' are recovered (i.e. 0/0=1 here)"""
    
    fig, ax = plt.subplots(figsize = (8,6))

    magMeans = propDF["Mv(mean)"].values

    maxMv = np.ceil(np.max(magMeans))
    minMv = np.floor(np.min(magMeans))
    histRange = (minMv, maxMv)
    binSize = 0.5
    histBins = int((maxMv-minMv)//binSize) 

    vMagHist = np.histogram(magMeans, range=histRange, bins=histBins)

    overBkgData = np.where((propDF["Mean COM Flux"]-propDF["STD COM Flux"])>=bkgLim)[0]

    ax.set(xlabel="Mean V mag", ylabel="Fraction recovered")
    #* Note recovered is when mean-std > bkg, 84% of data above noise floor
    #* recoverd is 1 when no points in bin anyway. 
    
    passedBkgLim = np.where((propDF["Mean COM Flux"]-propDF["STD COM Flux"])>=bkgLim, True, False)

    # astrProperties["Over Background Limit"] = passedBkgLim

    overBkgMags = magMeans[passedBkgLim]

    overBkgHist = np.histogram(overBkgMags, range=histRange, bins=histBins)

    normedHist = overBkgHist[0]/vMagHist[0]

    #*Here 0/0 = 1, as we aren't losing the asteroids to the mag cut if there aren't any in the bin to beging with
    for i, val in enumerate(normedHist):
        if str(val) =="nan": 
            normedHist[i] = 1

    ax.stairs(normedHist, vMagHist[1])

    if saving:
        fig.savefig(f"{savePath}recoverdHist{nameChange}.pdf")

    return passedBkgLim



def plot_lc_from_name(name, lcDF,matchDF = None, saving:bool=False, savePath:str="../Figures/", nameChange=""):
    ps = 5
    nCut = _name_cut(lcDF,name)
    fig, ax = plt.subplots(1,1,sharex=True,figsize = (8,6))
    ax.scatter(nCut["MJD"], nCut["Flux"], label="Interp Flux", s=ps, c="tab:blue", marker="o")
    ax.scatter(nCut["MJD"], nCut["COM Flux"], label="COM Flux", s=ps, c="tab:orange", marker="d")
    ax.set(xlabel="Time [MJD]", ylabel="Flux [counts]")

    if matchDF is not None:
        matchCut = _name_cut(matchDF,name)
        ax.scatter(matchCut["MJD"], matchCut["COM Flux"], label="Matches Flux", s=ps, c="k", marker="s")

    if saving:
        fig.savefig(f"{savePath}DetectMatchPos{nameChange}.pdf")



def matchesAndInterp(matchDF, interpDF, saving=False, savePath="../Figures/", nameChange=""):
        
    fig, ax = plt.subplots(1,2,figsize = (12,6), sharey=True)

    ax[0].scatter(interpDF["RA"], interpDF["MJD"], label="Interpolated Position")
    ax[0].scatter(matchDF["RA"], matchDF["Time"], label="Matched Position")
    ax[1].scatter(interpDF["Dec"], interpDF["MJD"])
    ax[1].scatter(matchDF["Dec"], matchDF["Time"])

    ax[0].set(xlabel="RA", ylabel="Time [MJD]")
    ax[1].set(xlabel="Dec")
    ax[0].legend()

    if saving:
        fig.savefig(f"{savePath}DetectMatchPos{nameChange}.pdf")



def matchHist(matchDF, interpDF, astrData, obsForDetect=3,saving=False, savePath="../Figures/", nameChange=""):

    unqNames = pd.unique(interpDF["Name"])

    obsForDetect=3 #number of observations needed to count a detection

    interpedMv = []
    foundMv = []

    for name in unqNames:
        avgItrpMv = astrData.at[astrData.index[astrData["Name"]==name]]["Mv(mean)"]
        interpedMv.append(avgItrpMv) 
        cutMtchDF = _name_cut(matchDF,name)
        if len(cutMtchDF.index)>=obsForDetect:
            avgResMv = cutMtchDF["MvMatch"].mean()
            foundMv.append(avgResMv)


    maxMv = np.ceil(np.max(interpedMv))
    minMv = np.floor(np.min(interpedMv))
    binSize = 0.5
    nBins = int((maxMv-minMv)//binSize) 

    interpMvHist, itrpBins = np.histogram(interpedMv, bins = nBins, range=(minMv, maxMv))

    foundMvHist, foundBins = np.histogram(foundMv, bins = itrpBins)
    # print(f"The number of matched detections with >= {obsForDetect} observations is {np.sum(foundMvHist)}") #// TODO get this number out



    completenessMvHist = foundMvHist/interpMvHist
    for i, val in enumerate(completenessMvHist): #*0/0=1
            if str(val) =="nan": 
                completenessMvHist[i] = 1

    fig,ax = plt.subplots()
    ax.stairs(completenessMvHist, itrpBins)
    ax.set(xlabel="Mean V Mag", ylabel="Fraction With Matches")
    fig.tight_layout()

    if saving:
        fig.savefig(f"{savePath}matchesHist{nameChange}.pdf")
    
    return



def class_Bar(propDF, saving=False, savePath="../Figures/", nameChange=""):

    classes, classCounts = np.unique(propDF["Class"], return_counts=True)

    fig, ax = plt.subplots(figsize = (12,6))
    ax.bar(classes, classCounts)
    ax.set(xlabel="Asteroid Class", ylabel="Number")
    ax.tick_params(axis='x', labelrotation=45)

    if saving:
        fig.savefig(f"{savePath}classesBar{nameChange}.pdf")

    return



def view_Asteroid(lcCut, reducedFluxes,name,numPx = 8, pointID = 0, cutList = [22,1,3,7], saving=False, savePath="../Figures/", nameChange=""):
    sector, cam, ccd, cut = cutList

    fig, ax = plt.subplots(figsize=(8,8))
    f = lcCut.at[pointID,"FrameIDs"]
    y =lcCut.at[pointID,"Y"]
    x =lcCut.at[pointID,"X"]
    flux= lcCut.at[pointID,"Flux"]
    intAp = CircularAperture((x,y), r=1.5)  #need to get aperture back for plot

    targetWSC = fits.open(f"../OzData/{sector}_{cam}_{ccd}_{cut}_wcs.fits")[0]
    w = wcs.WCS(targetWSC.header)
    interpCoords=SkyCoord(ra = lcCut.at[pointID,"RA"], dec = lcCut.at[pointID,"Dec"], unit="deg")
    floX, floY  = w.all_world2pix(interpCoords.ra, interpCoords.dec,0)
    redFluxCut = reducedFluxes[int(f),y-numPx:y+numPx,x-numPx:x+numPx]
    
    #*4x4 box,
    checkUp = 3
    checkDown = 2
    comY = lcCut.at[pointID,"COM Y"]
    comX = lcCut.at[pointID,"COM X"]
    comAp = CircularAperture((comX,comY), r=1.5)  
    comFlux = lcCut.at[pointID,"COM Flux"]

    ax.imshow(redFluxCut, extent =[x-numPx,x+numPx,y+numPx,y-numPx], vmin=np.percentile(redFluxCut, 3), vmax=np.percentile(redFluxCut,97))
    ax.scatter(floX,floY, marker="d", c="tab:purple", label=f"Float Coord")
    ax.scatter(x,y, marker="x", c="tab:red", label=f"Int Coord")
    ax.scatter(x+0.5,y+0.5, marker="o", c="k", label=f"{x+0.5},{y+0.5} \n Box centre")
    ax.scatter(comX, comY,c="tab:green", marker ="s", label="COM (float)")
    ax.plot([x-checkDown,x+checkUp,x+checkUp,x-checkDown,x-checkDown],[y-checkDown,y-checkDown,y+checkUp,y+checkUp,y-checkDown], c="tab:green", linestyle =":", label=f"COM check box")
    # ax.plot([x-1,x+2,x+2,x-1,x-1],[y-1,y-1,y+2,y+2,y-1], c="k", linestyle ="-.", label=f"Sum +2,-1 box \n F= {round(lcCut.at[pointID,'Flux'],1)}") #!box not used, ap at x,y is.
    intAp.plot(ax=ax, color="k", label=f"COM Aperture F={round(flux,1)}")
    comAp.plot(ax = ax, color = "tab:Green", label = f"COM Aperture F={round(comFlux,1)}")
    ax.legend(fontsize=10)
    ax.set(xlabel="X", ylabel="Y")
    if saving:
        fig.savefig(f"{savePath}_One_Frame_View_Of{name}in{nameChange}.pdf")
    return



#TODO add more as needed
