import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy import wcs
from astropy.utils.data import download_file
from astropy.io import fits
from astropy.stats import sigma_clip

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


%matplotlib widget

plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})


timeOffset = 2400000.5


def load_matches(sector, cam, ccd, cut, dirPath = "./"):
    matchesDF = pd.read_csv(f"{dirPath}{sector}_{cam}_{ccd}_{cut}_InterpAndDetect_Matches.csv")
    matchesDF.drop(columns="Unnamed: 0", inplace=True)
    return matchesDF

def load_interpolations(sector, cam, ccd, cut, dirPath = "./"):
    interpDF = pd.read_csv(f"{dirPath}InterpolatedQuerryResult_{sector}_{cam}_{ccd}_{cut}.csv")
    interpDF.drop(columns="Unnamed: 0", inplace=True)
    return interpDF

def load_fluxes(sector, cam, ccd, cut, dirPath = "../OzData/"):
    reduFlux = np.load(f"{dirPath}sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_ReducedFlux.npy")
    return reduFlux

def load_times(sector, cam, ccd, cut, dirPath = "../OzData/"):
    timesFromOz = np.load(f"{dirPath}sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_Times.npy")
    return timesFromOz

def name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn

def sum_fluxes(frames, ys, xs):
    """Sums 3x3 box around all f,y,x, retuns list of fluxes in frame order"""

    down = 1
    up = 2
    total = []

    for f,y,x in zip(frames,ys,xs):   
        #making sure everything is in bounds
        ymin = y-down
        ymax = y+up
        xmin = x-down
        xmax = x+up

        if ymin < 0: ymin = 0
        if ymax >= fluxBounds[1]: ymax = fluxBounds[1] -1
        if xmin < 0: xmin = 0
        if xmax >= fluxBounds[2]: xmax = fluxBounds[2]-1
        total.append(np.sum(reducedFluxes[f,ymin:ymax,xmin:xmax]))

    return total


def lightcurve_from_name(name):
    nameDf = name_cut(interpDF,name,"Name")
    
    #frameNums = [ np.argmin(np.abs(frameTimes -(time-timeOffset))) for time in nameDf["epoch"]]
    frameNums = []
    badIds = []

    for i, time in enumerate(nameDf["epoch"]):
        time = time-timeOffset
        


        thisFrame = np.argmin(np.abs(frameTimes -time))
        frameNums.append(thisFrame)

        if ((time - frameTimes.min()) < 0) or ((time - frameTimes.max())>0):
            
            badIds.append(i) # makes sure they are in the right times
            continue

        try:
            lastFrame = frameNums[-2]
        except:
            lastFrame = None

        if lastFrame == thisFrame: #2nd to last frame, as have added this frame already
            oldTime = frameTimes[frameNums[-1]] - time
            newTime = frameTimes[thisFrame] - time
            #shorter dt gets taken if the go to the same frame

            if np.abs(newTime)>=np.abs(oldTime): #New is longer (or ==) dt, so keep old one
                badIds.append(i)
            else:   #new dt is shorter, so use  new one
                badIds.append((i-1))
    
    targetWSC = fits.open(f"../OzData/{sector}_{cam}_{ccd}_{cut}_wcs.fits")[0]
    w = wcs.WCS(targetWSC.header)
    
    interpCoords=SkyCoord(ra = nameDf["RA"], dec = nameDf["Dec"], unit="deg")

    interpedX, interpedY  = w.all_world2pix(interpCoords.ra, interpCoords.dec,0)

    interpedX = interpedX.round().astype(int)
    interpedY = interpedY.round().astype(int)
    

    #add X, Y, Fs in
    nameDf["X"] = interpedX
    nameDf["Y"] = interpedY
    nameDf["Frames"] = frameNums

    #takes the flux sums, and 
    sumedFluxes = sum_fluxes(nameDf["Frames"], nameDf["Y"], nameDf["X"])
    nameDf["Flux"] = sumedFluxes

    badIds+= (np.where((interpedX<0) | (interpedX>= fluxBounds[2])| (interpedY<0) | (interpedY>= fluxBounds[1]))[0].tolist()) #! ugly af, but takes all out of bounds values in X or Y, and gets them into the badIDs

    badIds += np.nonzero( sigma_clip(sumedFluxes, masked=True, maxiters=5, sigma = 3).mask)[0].tolist() #! also ugly takes all sigma clipped values and gets them into the badIDs


    badIds = np.unique(badIds) #takes only the unique bads, as to not remove any twice (shouldn't matter, as no reindexing)

    #remove bad rows from nameDF, when X, Y, F are out of bounds, or F is repeat, or values are sigma clipped
    nameDf.drop(index=badIds, inplace=True)

    return nameDf    

def plot_lc_from_name(name):
    nCut = name_cut(totalLcDf,name=name, colName="Name")
    fig, ax = plt.subplots(1,1,sharex=True,figsize = (8,6))
    ax.scatter(nCut["epoch"]-timeOffset, nCut["Flux"])
    # ax.set_ylim(-200,200)
    


#*Global variables
#TODO take these from an input

sector = 22
cam = 1
ccd = 3
cut = 7

reducedFluxes = load_fluxes(sector, cam, ccd, cut)

fluxBounds = reducedFluxes.shape

frameTimes = load_times(sector, cam, ccd, cut)

interpDF = load_interpolations(sector, cam, ccd, cut)

extendedDfList = []

for name in np.unique(interpDF["Name"]):
    lcRes = lightcurve_from_name(name)
    extendedDfList.append(lcRes)   


totalLcDf = pd.concat(extendedDfList)
totalLcDf.reset_index(inplace=True, drop=True)
totalLcDf.to_csv(f"Interps_with_lc_{sector}_{cam}_{ccd}_{cut}.csv")

plot_lc_from_name(" Ruff ")




