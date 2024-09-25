import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import ndimage
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)


def load_matches(sector, cam, ccd, cut, dirPath = ""):
    matchesDF = pd.read_csv(f"{dirPath}MatchesToDetections.csv")
    matchesDF.drop(columns="Unnamed: 0", inplace=True)
    return matchesDF

def load_interpolations(sector, cam, ccd, cut, dirPath = ""):
    interpDF = pd.read_csv(f"{dirPath}asteroid_interpolated_positions.csv")
    interpDF.drop(columns="Unnamed: 0", inplace=True)
    return interpDF

def load_fluxes(sector, cam, ccd, cut, dirPath = ""):
    reduFlux = np.load(f"{dirPath}sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_ReducedFlux.npy")
    return reduFlux

def load_times(sector, cam, ccd, cut, dirPath = ""):
    timesFromOz = np.load(f"{dirPath}sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_Times.npy")
    return timesFromOz

def name_cut(df, name:str, colName:str="Name"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn

def sum_fluxes(frames, ys, xs, fluxBounds:list,reducedFluxes):
    """
    1.5 px aperture photometry around the COM coord
    
    Sums 3x3 box around all f,y,x, retuns list of fluxes in frame order
    
    """

    down= 1
    up = 2

    checkUp = 3
    checkDown = 2
    total = []
    comTotal = []

    comCoordsList = []

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
        
        photTable = aperture_photometry(reducedFluxes[int(f),:,:],CircularAperture((x,y), r=1.5)) 
        apFlux = photTable["aperture_sum"][0]
        total.append(apFlux)

        ymin = y-checkDown
        ymax = y+checkUp
        xmin = x-checkDown
        xmax = x+checkUp

        if ymin < 0: ymin = 0
        if ymax >= fluxBounds[1]: ymax = fluxBounds[1] -1
        if xmin < 0: xmin = 0
        if xmax >= fluxBounds[2]: xmax = fluxBounds[2]-1

        
        com = ndimage.center_of_mass(reducedFluxes[int(f),ymin:ymax,xmin:xmax])
        if (com[0]>=0) &(com[0]<=7) &(com[1]>=0) & (com[1]<=7): #make sure COM in 7x7 box
            comX =x-checkDown+com[1]
            comY = y-checkDown+com[0]
        else:
            #reset defaults
            comX =x
            comY = y
        
        try:
            comX = int(round(comX))
            comY = int(round(comY))
        except:
            comTotal.append(np.nan)
            continue
        
        photTable = aperture_photometry(reducedFluxes[int(f),:,:],CircularAperture((comX,comY), r=1.5))

        comAppFlux = photTable["aperture_sum"][0]
        comTotal.append(comAppFlux)
        
        comCoordsList.append([comX,comY])
    
    comCoordsArr = np.array(comCoordsList)

    return total, comTotal, comCoordsArr


def lightcurve_from_name(name, interpDF,fluxBounds,reducedFluxes,basePath):
    nameDf = name_cut(interpDF,name,"Name")
    frameNums = [] #*sidesteped by keeping frame ID in from interpolation
    badIds = []
        
    targetWSC = fits.open(f"{basePath}wcs.fits")[0]
    w = wcs.WCS(targetWSC.header)
    
    interpCoords=SkyCoord(ra = nameDf["RA"], dec = nameDf["Dec"], unit="deg")

    interpedX, interpedY  = w.all_world2pix(interpCoords.ra, interpCoords.dec,0)

    #rounds to nearest whole number, which returns #.0 as a float. then int converts. if just int masked, it floors the number, not rounds it.
    interpedX = interpedX.round().astype(int)
    interpedY = interpedY.round().astype(int)
    

    #add X, Y, Fs in
    nameDf["X"] = interpedX
    nameDf["Y"] = interpedY
    # nameDf["Frames"] = frameNums

    #takes the flux sums, and 
    fluxes, comFluxes, comCoords = sum_fluxes(nameDf["FrameIDs"], nameDf["Y"], nameDf["X"], fluxBounds, reducedFluxes)

    comXs = comCoords[:,0]
    comYs = comCoords[:,1]

    nameDf["Flux"] = fluxes
    nameDf["COM Flux"] = comFluxes 
    nameDf["COM X"] = comXs
    nameDf["COM Y"] = comYs


    badIds+= (np.where((interpedX<0) | (interpedX>= fluxBounds[2])| (interpedY<0) | (interpedY>= fluxBounds[1]))[0].tolist()) #! ugly af, but takes all out of bounds values in X or Y, and gets them into the badIDs


    badIds += np.nonzero( sigma_clip(comFluxes, masked=True, maxiters=5, sigma = 3).mask)[0].tolist() #! also ugly takes all sigma clipped values and gets them into the badIDs


    badIds = np.unique(badIds) #takes only the unique bads, as to not remove any twice (shouldn't matter, as no reindexing)

    #remove bad rows from nameDF, when X, Y, F are out of bounds, or F is repeat, or Flux values are sigma clipped
    nameDf.drop(index=badIds, inplace=True)

    return nameDf    


def makeLCs(sector,cam,ccd,cut):
    """
    Only Flux is COM flux
    """
    basePath = f"../TESSdata/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of16/" 

    reducedFluxes = load_fluxes(sector, cam, ccd, cut,basePath)

    fluxBounds = reducedFluxes.shape

    # frameTimes = load_times(sector, cam, ccd, cut, basePath)

    interpDF = load_interpolations(sector, cam, ccd, cut,basePath)

    unqNames = np.unique(interpDF["Name"])

    extendedDfList = []

    for name in unqNames:
        lcRes = lightcurve_from_name(name, interpDF,fluxBounds,reducedFluxes,basePath)
        extendedDfList.append(lcRes)   

    print(len(extendedDfList))

    totalLcDf = pd.concat(extendedDfList)
    totalLcDf.reset_index(inplace=True, drop=True)

    totalLcDf.to_csv(f"{basePath}InterpedLC.csv")
