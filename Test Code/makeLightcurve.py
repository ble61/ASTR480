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
from scipy import ndimage
import numpy.typing as npt
import matplotlib.colors as mplc
import matplotlib.cm as mpcm
import itertools
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from astropy.stats import SigmaClip as astroSC
from photutils.background import Background2D, MedianBackground

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)

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
    checkUp = 4
    checkDown = 3
    total = []
    comTotal = []

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
        total.append(np.sum(reducedFluxes[int(f),ymin:ymax,xmin:xmax]))

        ymin = y-checkDown
        ymax = y+checkUp
        xmin = x-checkDown
        xmax = x+checkUp

        if ymin < 0: ymin = 0
        if ymax >= fluxBounds[1]: ymax = fluxBounds[1] -1
        if xmin < 0: xmin = 0
        if xmax >= fluxBounds[2]: xmax = fluxBounds[2]-1


        #???? comTotal.append(np.sum(reducedFluxes[int(f),ymin:ymax,xmin:xmax]))
        

        com = ndimage.center_of_mass(reducedFluxes[int(f),ymin:ymax,xmin:xmax])
        comX = x-checkDown+com[1]
        comY = y-checkDown+com[0]
        
        try:
            comX = int(round(comX))
            comY = int(round(comY))
        except:
            comTotal.append(np.nan)
            continue
        
        photTable = aperture_photometry(reducedFluxes[int(f),:,:],CircularAperture((comX,comY), r=1.5))

        comAppFlux = photTable["aperture_sum"][0]
        comTotal.append(comAppFlux)
        continue

        ymin = comY-down
        ymax = comY+up
        xmin = comX-down
        xmax = comX+up

        if ymin < 0: ymin = 0
        if ymax >= fluxBounds[1]: ymax = fluxBounds[1] -1
        if xmin < 0: xmin = 0
        if xmax >= fluxBounds[2]: xmax = fluxBounds[2]-1

        comTotal.append(np.sum(reducedFluxes[int(f),ymin:ymax,xmin:xmax]))
        

    return total, comTotal


def lightcurve_from_name(name):
    nameDf = name_cut(interpDF,name,"Name")
    frameNums = [] #*sidesteped by keeping frame ID in from interpolation
    badIds = []


        # for i, time in enumerate(nameDf["MJD"]):
        #     # time = time-timeOffset

        #     thisFrame = np.argmin(np.abs(frameTimes -time))
        #     frameNums.append(thisFrame)

        #     if ((time - frameTimes.min()) < 0) or ((time - frameTimes.max())>0):
                
        #         badIds.append(i) # makes sure they are in the right times
        #         continue

        #     try:
        #         lastFrame = frameNums[-2]
        #     except:
        #         lastFrame = None

        #     if lastFrame == thisFrame: #2nd to last frame, as have added this frame already
        #         oldTime = frameTimes[frameNums[-1]] - time
        #         newTime = frameTimes[thisFrame] - time
        #         #shorter dt gets taken if the go to the same frame

        #         if np.abs(newTime)>=np.abs(oldTime): #New is longer (or ==) dt, so keep old one
        #             badIds.append(i)
        #         else:   #new dt is shorter, so use  new one
        #             badIds.append((i-1))
        
    targetWSC = fits.open(f"../OzData/{sector}_{cam}_{ccd}_{cut}_wcs.fits")[0]
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
    sumedFluxes, comFluxes = sum_fluxes(nameDf["FrameIDs"], nameDf["Y"], nameDf["X"])
    nameDf["Flux"] = sumedFluxes
    nameDf["COM Flux"] = comFluxes

    badIds+= (np.where((interpedX<0) | (interpedX>= fluxBounds[2])| (interpedY<0) | (interpedY>= fluxBounds[1]))[0].tolist()) #! ugly af, but takes all out of bounds values in X or Y, and gets them into the badIDs

    # badIds += np.nonzero( sigma_clip(sumedFluxes, masked=True, maxiters=5, sigma = 3).mask)[0].tolist() #! also ugly takes all sigma clipped values and gets them into the badIDs

    badIds += np.nonzero( sigma_clip(comFluxes, masked=True, maxiters=5, sigma = 3).mask)[0].tolist() #sigma clip gets rid of nans


    badIds = np.unique(badIds) #takes only the unique bads, as to not remove any twice (shouldn't matter, as no reindexing)

    #remove bad rows from nameDF, when X, Y, F are out of bounds, or F is repeat, or Flux values are sigma clipped
    nameDf.drop(index=badIds, inplace=True)

    return nameDf    

def plot_lc_from_name(name):
    nCut = name_cut(totalLcDf,name=name, colName="Name")
    fig, ax = plt.subplots(1,1,sharex=True,figsize = (8,6))
    ax.scatter(nCut["MJD"], nCut["Flux"])
    ax.scatter(nCut["MJD"], nCut["COM Flux"])
    # ax.set_ylim(-200,200)
    
def plotExpectedPos(posDf: pd.DataFrame, timeList: npt.ArrayLike, targetPos: list, magLim: float = 20.0, scaleAlpha: bool = False, minLen: int = 0, saving=False) -> plt.Figure:
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

        coords = SkyCoord(posDf.loc[nameIDs]["RA"],
                          posDf.loc[nameIDs]["Dec"], unit=(u.deg, u.deg))
        pixels = coords.to_pixel(w)

        # scales each objects alpha

        if scaleAlpha:
            avgMag = posDf.loc[nameIDs]["Mv"].mean()
            avgMag = hs[name]
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

    if saving: fig.savefig(f"./Testing Figures/interpPos_{sector}_{cam}_{ccd}_{cut}.png") #!CHECK
    return fig #retunrs instead of saves, #? is this useful?
    # fig.savefig(f"./ecCoordsandAlphascaled_ra{ra_i}_dec{dec_i}_t{t_i.mjd}_Mv{magLim}.png") #!CHECK

def setupQuery(sector,cam,ccd,cut,dirPath="../OzData/"):

    fname = f"{dirPath}{sector}_{cam}_{ccd}_{cut}_wcs.fits"

    targetWSC = fits.open(fname)[0]

    targetRa_i = targetWSC.header["CRVAL1"]
    targetDec_i = targetWSC.header["CRVAL2"]
    targetTime_i = Time(targetWSC.header["DATE-OBS"])

    myTargetPos = [targetRa_i, targetDec_i, targetTime_i]

    return myTargetPos


def recoveredHist(astrProperties, bkgLim):
    """Takes the mean and deviation of the fluxes, and compares to a background Limit. A histogram of those greater than the limit is made. If no asteroid is in a mag bin, then 'all' are recovered (i.e. 0/0=1 here)"""
    
    fig, ax = plt.subplots()

    magMeans = astrProperties["Mv(mean)"].values

    histRange = (13,20)
    histBins = 14

    vMagHist = np.histogram(magMeans, range=histRange, bins=histBins)

    overBkgData = np.where((astrProperties["Mean COM Flux"]-astrProperties["STD COM Flux"])>=bkgLim)[0]

    ax.set(xlabel="Mean V mag", ylabel="Fraction recovered")
    #* Note recovered is when mean-std > bkg, 84% of data above noise floor
    #* recoverd is 1 when no points in bin anyway. 
    passedBkgLim = np.where((astrProperties["Mean COM Flux"]-astrProperties["STD COM Flux"])>=bkgLim, True, False)

    # astrProperties["Over Background Limit"] = passedBkgLim

    overBkgMags = magMeans[passedBkgLim]

    overBkgHist = np.histogram(overBkgMags, range=histRange, bins=histBins)

    normedHist = overBkgHist[0]/vMagHist[0]

    #*Here 0/0 = 1, as we aren't losing the asteroids to the mag cut if there aren't any in the bin to beging with
    for i, val in enumerate(normedHist):
        if str(val) =="nan": 
            normedHist[i] = 1

    ax.stairs(normedHist, vMagHist[1])

    return passedBkgLim




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

unqNames = np.unique(interpDF["Name"])
print(len(unqNames))

extendedDfList = []

for name in np.unique(interpDF["Name"]):
    lcRes = lightcurve_from_name(name)
    extendedDfList.append(lcRes)   

print(len(extendedDfList))

totalLcDf = pd.concat(extendedDfList)
totalLcDf.reset_index(inplace=True, drop=True)

namesAfter =np.unique(totalLcDf["Name"])

print(len(namesAfter))

namesDroped = np.setdiff1d(unqNames, namesAfter) 

# print(namesDroped)

plotExpectedPos(totalLcDf,frameTimes,setupQuery(sector,cam,ccd,cut))


totalLcDf.to_csv(f"Interps_with_lc_{sector}_{cam}_{ccd}_{cut}.csv")


fig, ax = plt.subplots(figsize=(12,6))

count = 0

medBkgs = []

for i in np.random.randint(0,fluxBounds[0], size=100):

    sc = astroSC(sigma=3.0)
    bkg_estimator = MedianBackground()
    try:
        bkg = Background2D(reducedFluxes[i,:,:], (50, 50), filter_size=(3, 3),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    except:
        continue
    medBkgs.append(bkg.background_median)


bkgLim = np.mean(medBkgs)

meanFluxes = []
stdFluxes = []

for i,name in enumerate(namesAfter):
    nCut = name_cut(totalLcDf,name,"Name")
    cflux = nCut["COM Flux"] 
    mean = np.mean(cflux)
    meanFluxes.append(mean)
    
    sdev = np.std(cflux)
    stdFluxes.append(sdev)

    if mean - sdev >= bkgLim :
        count+=1
        ax.errorbar(i,mean,sdev, markersize=10, capsize=3)

astrData = pd.read_csv(f"./asteroids_in_{sector}_{cam}_{ccd}_{cut}_properties.csv")

astrData.drop(columns = ["Unnamed: 0"], inplace=True)

astrData["Mean COM Flux"] = meanFluxes
astrData["STD COM Flux"] = stdFluxes

astrData["Over Background Limit"] = recoveredHist(astrData, bkgLim)


astrData.to_csv(f"./asteroids_in_{sector}_{cam}_{ccd}_{cut}_properties.csv")

print(count)

plot_lc_from_name("Bernoulli")


#* sawtooth trials
trialCut = name_cut(totalLcDf,"Bernoulli", colName="Name")

timeCut = trialCut.loc[np.where((trialCut["MJD"]>=58900.58) & (trialCut["MJD"]<=58901.11))]

timeCut.reset_index(drop=True, inplace=True)

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(timeCut["MJD"], timeCut["Flux"], label = "Flux from interpolated position")
ax.set(xlabel="Time [MJD]", ylabel="Flux",ylim=(100,280))

fig2, ax2 = plt.subplots(5,5, figsize =(15,15))

ax2 = np.ravel(ax2)

numPx = 6

targetWSC = fits.open(f"../OzData/{sector}_{cam}_{ccd}_{cut}_wcs.fits")[0]
w = wcs.WCS(targetWSC.header)


comSums = []
comApSums = []

for pointID in range(timeCut.index.max()+1):

    f = timeCut.at[pointID,"FrameIDs"]
    y =timeCut.at[pointID,"Y"]
    x =timeCut.at[pointID,"X"]

    interpCoords=SkyCoord(ra = timeCut.at[pointID,"RA"], dec = timeCut.at[pointID,"Dec"], unit="deg")

    floX, floY  = w.all_world2pix(interpCoords.ra, interpCoords.dec,0)

    redFluxCut = reducedFluxes[int(f),y-numPx:y+numPx,x-numPx:x+numPx]

    ax2[pointID].scatter(x,y, marker="x",s=10, c="tab:red", label=f"{f}, {x},{y}")
    ax2[pointID].scatter(floX,floY, marker="d",s=10, c="tab:purple", label=f"float coord")
    ax2[pointID].imshow(redFluxCut, extent =[x-numPx,x+numPx,y+numPx,y-numPx], vmin=np.percentile(redFluxCut, 3), vmax=np.percentile(redFluxCut,97))


    ax2[pointID].scatter(x+0.5,y+0.5, marker="o",s=5, c="k", label=f"{x+0.5},{y+0.5}")
    ax2[pointID].plot([x-1,x+2,x+2,x-1,x-1],[y-1,y-1,y+2,y+2,y-1], c="k", linestyle = "-.",label=f"Sum Box, F={round(timeCut.at[pointID,'Flux'],1)}") 

    #4x4 box,
    checkUp = 4
    checkDown = 3
    checkBox=reducedFluxes[int(f),y-checkDown:y+checkUp,x-checkDown:x+checkUp]
    ax2[pointID].plot([x-checkDown,x+checkUp,x+checkUp,x-checkDown,x-checkDown],[y-checkDown,y-checkDown,y+checkUp,y+checkUp,y-checkDown], c="tab:green", linestyle =":")

    com = ndimage.center_of_mass(checkBox)
    ax2[pointID].scatter(x-checkDown+com[1], y-checkDown+com[0],marker = "s", s=10,c="tab:green", label="COM")

    comX = x-checkDown+com[1]
    comY = y-checkDown+com[0]


    #Photutils appeture

    thisAp = CircularAperture((comX,comY), r=1.5)  

    photTable = aperture_photometry(reducedFluxes[int(f),:,:],thisAp)

    comApFlux = photTable["aperture_sum"][0]
    comApSums.append(comApFlux)


    comX = int(x-checkDown+round(com[1]))
    comY = int(y-checkDown+round(com[0]))
    comSum = np.sum(reducedFluxes[int(f), comY-1:comY+2, comX-1:comX+2])
    comSums.append(comSum)
    ax2[pointID].scatter(comX+0.5,comY+0.5, label="COM sum centre",c="tab:orange",s=10)
    ax2[pointID].plot([comX-1,comX+2,comX+2,comX-1,comX-1],[comY-1,comY-1,comY+2,comY+2,comY-1], c="tab:orange",linestyle=":", label=f"COM sum box F={round(comSum,1)}")
    # ax2[pointID].legend(fontsize=5)


ax2 = np.reshape(ax2,(5,5))

ax.scatter(timeCut["MJD"],comSums, marker="d",label="Flux from Centre of Mass")
ax.scatter(timeCut["MJD"],comApSums, marker="s",label="Aperture flux from Centre of Mass")
ax.legend()


numPx = 6
pointID = 24

fig, ax3 = plt.subplots(figsize=(8,8))
f = timeCut.at[pointID,"FrameIDs"]
y =timeCut.at[pointID,"Y"]
x =timeCut.at[pointID,"X"]

interpCoords=SkyCoord(ra = timeCut.at[pointID,"RA"], dec = timeCut.at[pointID,"Dec"], unit="deg")

floX, floY  = w.all_world2pix(interpCoords.ra, interpCoords.dec,0)

redFluxCut = reducedFluxes[int(f),y-numPx:y+numPx,x-numPx:x+numPx]



#4x4 box,
checkUp = 4
checkDown = 3
checkBox=reducedFluxes[int(f),y-checkDown:y+checkUp,x-checkDown:x+checkUp]


com = ndimage.center_of_mass(checkBox)
comX = int(x-checkDown+round(com[1]))
comY = int(y-checkDown+round(com[0]))
comSum = np.sum(reducedFluxes[int(f), comY-1:comY+2, comX-1:comX+2])


thisAp = CircularAperture((x-checkDown+com[1],y-checkDown+com[0]), r=1.5)  

photTable = aperture_photometry(reducedFluxes[int(f),:,:],thisAp)

comApFlux = photTable["aperture_sum"][0]

ax3.imshow(redFluxCut, extent =[x-numPx,x+numPx,y+numPx,y-numPx], vmin=np.percentile(redFluxCut, 3), vmax=np.percentile(redFluxCut,97))
ax3.scatter(floX,floY, marker="d", c="tab:purple", label=f"Float Coord")
ax3.scatter(x,y, marker="x", c="tab:red", label=f"Int Coord")
ax3.scatter(x+0.5,y+0.5, marker="o", c="k", label=f"{x+0.5},{y+0.5} \n Box centre")

ax3.scatter(x-checkDown+com[1], y-checkDown+com[0],c="tab:green", marker ="s", label="COM (float)")

 
ax3.scatter(comX+0.5,comY+0.5, label="COM sum centre",c="tab:orange")

ax3.plot([x-checkDown,x+checkUp,x+checkUp,x-checkDown,x-checkDown],[y-checkDown,y-checkDown,y+checkUp,y+checkUp,y-checkDown], c="tab:green", linestyle =":", label=f"COM check box")
#!what it is, when TOP left origin
ax3.plot([x-1,x+2,x+2,x-1,x-1],[y-1,y-1,y+2,y+2,y-1], c="k", linestyle ="-.", label=f"Sum +2,-1 box \n F= {round(timeCut.at[pointID,'Flux'],1)}") 
ax3.plot([comX-1,comX+2,comX+2,comX-1,comX-1],[comY-1,comY-1,comY+2,comY+2,comY-1], c="tab:orange",linestyle=":", label=f"COM sum box \n F={round(comSum,1)}")
thisAp.plot(ax = ax3, color = "tab:Green", label = f"Aperture F= {round(comApFlux,1)}")

ax3.legend(fontsize=10)
ax3.set(xlabel="X", ylabel="Y")






# #! Origin is TOP LEFT, not bottom left!!!!!!
# #! Shown bellow... 
# fig4,ax4 = plt.subplots()

# toShow = reducedFluxes[0,0:15,0:15]

# ax4.imshow(toShow, extent =[0,15,15,0], vmin=np.percentile(toShow, 3), vmax=np.percentile(toShow,97)) 

# x=3
# y=3


# ax4.scatter(x,y,c="k")
# ax4.plot([x-1,x+2,x+2,x-1,x-1],[y-1,y-1,y+2,y+2,y-1], c="k")


# fig5,ax5 = plt.subplots()

# sumSlice = reducedFluxes[0,y-1:y+2,x-1:x+2]

# ax5.imshow(sumSlice, extent = [x-1,x+2,y-1,y+2],vmin=np.percentile(toShow, 3), vmax=np.percentile(toShow,97), cmap="magma")