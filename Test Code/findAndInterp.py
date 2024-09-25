import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.cm as mpcm
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy import wcs
from astropy.utils.data import download_file
from astropy.io import fits
from astroquery.jplhorizons import Horizons
import itertools
from tqdm import tqdm
from scipy.spatial import KDTree
import shutil


plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})



timeOffset = 2400000.5

def _Skybotquery(ra, dec, times, radius=10/60, location='C57', cache=False):
    """Returns a list of asteroids/comets given a position and time.
    This function relies on The Virtual Observatory Sky Body Tracker (SkyBot)
    service which can be found at http://vo.imcce.fr/webservices/skybot/
        Geert's magic code
            Ryans Magic code

    Parameters
    ----------https://github.com/bray217/tessts.git
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    times : array of float
        Times in Julian Date.
    radius : float
        Search radius in degrees.
    location : str
        Spacecraft location. Options include `'kepler'` and `'tess'`.
    cache : bool
        Whether to cache the search result. Default is True.
    Returns
    -------
    result : `pandas.DataFrame`
        DataFrame containing the list of known solar system objects at the
        requested time and location.
    """
    url = 'http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?'
    url += '-mime=text&'
    url += '-ra={}&'.format(ra)
    url += '-dec={}&'.format(dec)
    url += '-rd={}&'.format(radius)
    url += '-loc={}&'.format(location)
    # TODO add more things here?

    df = None
    times = np.atleast_1d(times)
    responses =[]
    for time in tqdm(times, desc='Querying for SSOs'):
        url_queried = url + 'EPOCH={}'.format(time)
        response = download_file(url_queried, cache=cache)
        responses.append(response)
        if open(response).read(10) == '# Flag: -1':  # error code detected?
            raise IOError("SkyBot Solar System query failed.\n"
                          "URL used:\n" + url_queried + "\n"
                          "Response received:\n" + open(response).read())
        res = pd.read_csv(response, delimiter='|', skiprows=2)
        if len(res) > 0:
            res['epoch'] = time
            res.rename({'# Num ': 'Num', ' Name ': 'Name', ' RA(h) ': 'RA', ' DE(deg) ': 'Dec',
                       ' Class ': 'Class', ' Mv ': 'Mv'}, inplace=True, axis='columns')
            res = res[['Num', 'Name', 'RA', 'Dec', 'Class',
                       'Mv', 'epoch']].reset_index(drop=True)
            if df is None:
                df = res
            else:
                df = pd.concat([df, res], ignore_index=True)
    if df is not None:
        # // ! should have inplace=True...
        df.reset_index(drop=True, inplace=True)
    return df, responses


def setupQuery(sector,cam,ccd,cut,dirPath="../OzData/"):

    fname = f"{dirPath}{sector}_{cam}_{ccd}_{cut}_wcs.fits"

    targetWSC = fits.open(fname)[0]

    targetRa_i = targetWSC.header["CRVAL1"]
    targetDec_i = targetWSC.header["CRVAL2"]
    targetTime_i = Time(targetWSC.header["DATE-OBS"])

    myTargetPos = [targetRa_i, targetDec_i, targetTime_i]

    return myTargetPos

def querySB(targetPos: list, qRad: float = 10.0, qLoc: str = "C57", numTimesteps: int = 27, magLim: float = 20.0) -> tuple[pd.DataFrame, npt.ArrayLike]:
    """
    Queries SkyBot in a box of width qRad after making a list of the times to query at. returns this timelist after the query has been made and down selected to just the objects brighter than magLim

    Inputs
        targetPos: list of len=3, [float, float, Time]
            list of initial ra (float), dec (float) and time (astropy.Time value), the center point of the query and the first time of observations 

        qRad: float
            size of query box in degrees, default 10.0
        qLoc: string
            the obscode of the observatory, default 'C57', which is TESS
        numTimesteps: int
            the number of times to query SB. default 27
        magLim: float
            Limiting magnitude of the query, default 20.0

    Outputs
        brightResult: DataFrame
            the magLim cut df from the query. With Collumns ['Num', 'Name','RA','Dec', 'Class', 'Mv', 'epoch']. 
        timeList: arrayLike of Times
            astropy.time array like of the times of queries, also in 'epoch' collumn in brigtResult, but makes sure that the whole queried time range is acounted for, instead of maybe nothing was found at one time.
    """
    ra_i, dec_i, t_i = targetPos
    dt = (27/numTimesteps)*u.day  # a dt in days
    # list of times constructed sequentially
    timeList = t_i + dt*np.arange(0, numTimesteps)
    result= pd.DataFrame()
    
    while len(result)==0: #so if query timesout, it will restart with the cache
        try:
            result, responses = _Skybotquery(ra_i, dec_i, timeList.jd,
                          radius=qRad, location=qLoc, cache=True) #timeout throws error, so need to be in try/except to not close
        except Exception as e:
            print(e)
            continue #restarts query with cache

        if str(type(result)) != "<class 'pandas.core.frame.DataFrame'>": 
            #// ! need something for when no asteroids are in cut.
            #// ! else everthing else breaks too.
            result = pd.DataFrame(data =[[np.nan,"There Were No Asteroids",np.nan,np.nan,np.nan,np.nan]], columns = ["Num", "Name", "RA","Dec","Mv","epoch"])
            break #to get out of while

    

    brightResult = result.loc[result["Mv"] <= magLim].reset_index(drop=True)

    brightResult["Name"] = [name.strip() for name in brightResult["Name"]]

    brightResult["MJD"] = brightResult["epoch"]-timeOffset

    brightResult.drop(columns="epoch", inplace=True)

    coords = SkyCoord(
        brightResult["RA"], brightResult["Dec"], unit=(u.hourangle, u.deg))
    brightResult["RA"] = coords.ra.deg
    brightResult["Dec"] = coords.dec.deg

    return brightResult, timeList, responses


def get_properties_Horizons(asteroidsDf, time, loc:str="500@10")->pd.DataFrame:
    
    eleList = []

    unitTime = Time(time, format="mjd")

    for name in asteroidsDf["Name"]:
        try:
            horizQ = Horizons(id = name, epochs = unitTime.jd, location= loc, id_type = "asteroid_name")
            elements = horizQ.elements()
            a = elements["a"].value[0] #subscriting to get number instead of list of 1 value????
            e = elements["e"].value[0]
            i = elements["incl"].value[0]
            H = elements["H"].value[0]
            #? phaseAng = elements["alpha"].value[0]

        except Exception as ex:
            print(f"{name} failed horizons check because:\n {ex}")
            a = np.NaN
            e = np.NaN
            i = np.NaN
            H = np.NaN
            # phaseAng = np.NaN

        #TODO lcdb querry
        #lcdbQ = querryfunc()...
        #knownP = lcbdQ.Period()...
        #if knownP = None:
            #knownP = np.NaN


        eleList.append([name,a,e,i,H]) #,knownP

    eledf = pd.DataFrame(eleList,columns=["Name Horizons","a","e","i","H"]) #knownP

    astrDfwEle = pd.concat([asteroidsDf, eledf], axis=1)

    return astrDfwEle

def find_asteroids(sector,cam,ccd,cut):

    res, timeList, responses = querySB(setupQuery(sector, cam, ccd, cut), numTimesteps=54, qRad=3.05)

    targetWSC = fits.open(f"../OzData/{sector}_{cam}_{ccd}_{cut}_wcs.fits")[0]
    w = wcs.WCS(targetWSC.header)
    
    coords=SkyCoord(ra = res["RA"], dec = res["Dec"], unit="deg")

    xCoord, yCoord  = w.all_world2pix(coords.ra, coords.dec,0)

    #rounds to nearest whole number, which returns #.0 as a float. then int converts. if just int masked, it floors the number, not rounds it.
    xCoord = xCoord.round().astype(int)
    yCoord = yCoord.round().astype(int)
    

    #add X, Y, Fs in
    res["X"] = xCoord
    res["Y"] = yCoord

    fluxBounds = 513

    badIds = []
    badIds+= (np.where((xCoord<-1) | (xCoord>= fluxBounds)| (yCoord<-1) | (yCoord>= fluxBounds))[0].tolist()) #! ugly af, but takes all out of bounds values in X or Y, and gets them into the badIDs
    #give extra space for interploations to work with
    res.drop(index=badIds, inplace=True)

    interpRes = interplolation_of_pos(res, sector)

    interpRes.drop(columns=["X", "Y"], inplace=True)

    coords=SkyCoord(ra = interpRes["RA"], dec = interpRes["Dec"], unit="deg")

    xCoord, yCoord  = w.all_world2pix(coords.ra, coords.dec,0)

    #rounds to nearest whole number, which returns #.0 as a float. then int converts. if just int masked, it floors the number, not rounds it.
    xCoord = xCoord.round().astype(int)
    yCoord = yCoord.round().astype(int)
    

    #add X, Y, Fs in
    interpRes["X"] = xCoord
    interpRes["Y"] = yCoord

    fluxBounds = 512

    badIds = []
    badIds+= (np.where((xCoord<0) | (xCoord>= fluxBounds)| (yCoord<0) | (yCoord>= fluxBounds))[0].tolist()) #! ugly af, but takes all out of bounds values in X or Y, and gets them into the badIDs
    #give extra space for interploations to work with
    interpRes.drop(index=badIds, inplace=True)



    unqNames = np.unique(interpRes["Name"])

    if list(unqNames) == []:
        
        interpRes = pd.DataFrame(data=[[np.nan,f"There Were No Asteroids in {sector}_{cam}_{ccd}_{cut}",np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]], columns = ["Num", "Name","RA","Dec","Mv","MJD","FrameIDs","X","Y"])

        asteroidPropertiesDf = pd.DataFrame([[np.nan,f"There Were No Asteroids in {sector}_{cam}_{ccd}_{cut}",np.nan,np.nan,np.nan,f"There Were No Asteroids in {sector}_{cam}_{ccd}_{cut}",np.nan,np.nan,np.nan, np.nan]],columns=["Num","Name","Mv(mean)","Class", "Number of Points","Name Horizons","a","e","i","H"])

        fname = f"asteroids_in_{sector}_{cam}_{ccd}_{cut}_properties"
        asteroidPropertiesDf.to_csv(f"{fname}.csv")

        return interpRes, responses
    else:
        propertiesList = []

        for name in unqNames:
            nCut = name_cut(res, name, colName="Name")
            numPoints = len(nCut.index)
            avgMv = np.mean(nCut["Mv"]).round(3)
            num = nCut.at[0,"Num"]
            astrClass = nCut.at[0,"Class"]  

            propertiesList.append([num,name,avgMv,astrClass, numPoints])

        asteroidPropertiesDf = pd.DataFrame(propertiesList,columns=["Num","Name","Mv(mean)","Class", "Number of Points"])

        withEles = get_properties_Horizons(asteroidPropertiesDf, timeList[0],"500@10")

        fname = f"asteroids_in_{sector}_{cam}_{ccd}_{cut}_properties"
        withEles.to_csv(f"{fname}.csv")


        return interpRes, responses, withEles


def name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn


def interplolation_of_pos(posDf, sector):
    
    #* Seems to be sec 27 and 56 when it changes
    #12hr queries constant
    if sector <27: 
        interpPoints=48  #1/2hr ffi
    elif sector>=27 and sector<56:
        interpPoints = 144 #1/6hr (10 min) ffi
    else:
        interpPoints = 432   # 200 s ffi
    

    unqNames = np.unique(posDf["Name"])
    dfsList = []


    frameTimes = np.load(f"../OzData/sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_Times.npy")

    for name in unqNames:
        
        underSampledPos = name_cut(posDf, name, colName="Name")

        underSampledPos["QueriedPoint"] = np.ones_like(underSampledPos["MJD"])
        minTime = underSampledPos["MJD"].min()
        maxTime = underSampledPos["MJD"].max()
        deltaTime = maxTime-minTime

        #// TODO interp times at frame times.
        # interpTimes = np.linspace(minTime,maxTime, int(interpPoints*deltaTime))#linspace to sample
        # #Ra and Dec samples

        frameIDs = np.where((frameTimes>=minTime) & (frameTimes<=maxTime))[0]

        interpTimes = frameTimes[frameIDs] #*gets IDs and times of frames all at once

        interpRAs = np.interp(x=interpTimes, xp=underSampledPos["MJD"], fp=underSampledPos["RA"])
        interpDecs = np.interp(x=interpTimes, xp=underSampledPos["MJD"], fp=underSampledPos["Dec"])
        interpDf = pd.DataFrame({"RA":interpRAs, "Dec":interpDecs, "MJD":interpTimes, "QueriedPoint":np.zeros_like(interpRAs), "FrameIDs":frameIDs.astype(int)}) #make into DF

        concatedDF = pd.concat([underSampledPos, interpDf])
        concatedDF.sort_values(by=['MJD'], inplace=True)
        concatedDF.reset_index(drop=True, inplace=True)
        concatedDF.ffill(inplace=True) #foward fill mag etc
        
        #\\TODO remove origonal points...
        concatedDF.drop(concatedDF[concatedDF["QueriedPoint"] ==1].index, inplace=True)
        concatedDF.drop(columns=["QueriedPoint"], inplace=True)
        
        dfsList.append(concatedDF)
    
    if dfsList == []:
        interpRes = pd.DataFrame(data=[[np.nan,f"There Were No Asteroids in {sector}_{cam}_{ccd}_{cut}",np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan]], columns = ["Num", "Name","RA","Dec","Mv","MJD","FrameIDs","X","Y","Class"])
    else:
        interpRes = pd.concat(dfsList) #puts everything back together
    
    # namesAfter =np.unique(interpRes["Name"])
    # namesDroped = np.setdiff1d(unqNames, namesAfter) #!names with only 1 point from query will be missed, as there is nothing to interpolate between.
    # print(namesDroped)

    #final clean
    interpRes.reset_index(drop=True, inplace=True)
 
    interpRes.drop(columns=["Class"], inplace=True)



    return interpRes


    
def plotExpectedPos(posDf: pd.DataFrame, timeList: npt.ArrayLike, targetPos: list, magLim: float = 20.0, scaleAlpha: bool = False, minLen: int = 0, saving=False, hsList = None) -> plt.Figure:
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
        scaleMult = 0.95  # changes how small alpha can get
        if str(type(hsList)) == "<class 'NoneType'>": #doesn't like asking for None when it is a array
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
            thisH = hsList[i]
            # scales the alpha of plotting to the brightness of the object, to give some idea of what might be detected
            alpha = (1-scaleMult*((thisH-brightest)/deltaMag))
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
        norm=cnorm, cmap=cmap), ax=ax, pad=0.12)
    clb.ax.set_title("Time in MJD", fontsize=16, y=1.03)
    clb.ax.ticklabel_format(useOffset=False)
    centerPix = SkyCoord(ra_i, dec_i, unit=(u.deg, u.deg)).to_pixel(w)  # center of seach area in pixels
    ax.scatter(centerPix[0], centerPix[1], marker="+",
               s=100, c="g")  # center marker
    
    #*BOX to check
    # corners = np.array([[ra_i+1.6, dec_i+1.6],[ra_i-1.6, dec_i+1.6],[ra_i-1.6, dec_i-1.6],[ra_i+1.6, dec_i-1.6],[ra_i+1.6, dec_i+1.6]])

    # cornersPix =  SkyCoord(corners, unit=(u.deg, u.deg)).to_pixel(w) 

    # ax.plot(cornersPix[0], cornersPix[1])


    if scaleAlpha: ax.text(-15, 1, s=scaleStr, fontsize=12)

    fig.tight_layout()

    if saving: fig.savefig(f"./Testing Figures/interpPos_{sector}_{cam}_{ccd}_{cut}.pdf") #!CHECK
    return fig #retunrs instead of saves, #? is this useful?
    # fig.savefig(f"./ecCoordsandAlphascaled_ra{ra_i}_dec{dec_i}_t{t_i.mjd}_Mv{magLim}.png") #!CHECK

#// One that breaks [22, 4, 1, 7]

sector = 22
cam = 2
ccd =3
cut = 4


resDf , responses, astrProps = find_asteroids(sector,cam,ccd,cut)
# interpDf = interplolation_of_pos(resDf,sector) Interp now done inside of the finding, so lower horizons queries
fname = f"InterpolatedQuerryResult_{sector}_{cam}_{ccd}_{cut}"
resDf.to_csv(f"{fname}.csv")

import os



#TO DELETE CACHE
# for respon in responses:
    # shutil.rmtree(respon[:-9])

frameTimes = np.load(f"../OzData/sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_Times.npy")

plotExpectedPos(resDf,frameTimes,setupQuery(sector,cam,ccd,cut), scaleAlpha=True, hsList=astrProps["H"].values, saving=True)

#TODO 
#//save out df of pos properties name:pos(t,x,y) from interpolations (name repeated)
#//save out df of name:num:avgMag:a:e:i:H + other properties of asteroid ?known period? 1 row per name. Other file will add found Period, lc properties etc 


unqNames = np.unique(resDf["Name"])

numNames = len(unqNames)




#*to force just one:
unqNames = [unqNames[0]]

#*or pick a name
# unqNames=["Bernoulli"] 

for name in unqNames:
    # if np.random.rand()< 1/numNames:
    nameCut = name_cut(resDf, name,"Name")
    # timeStart = Time(nameCut["epoch"].min(), format="jd")
    # timeEnd = Time(nameCut["epoch"].max(), format="jd")
    
    # horizQ = Horizons(id = name, epochs = {"start":str(tforHorz(timeStart)), "stop":str(tforHorz(timeEnd)), "step":"30m"}, location= "500@-95")
    rasActs = []
    decsActs = []
    for time in nameCut["MJD"]:
        try:
            horizQ = Horizons(id = name, epochs =time, location= "500@-95")
            eph = horizQ.ephemerides()
            rasAct = float(eph["RA"][0])
            decsAct = float(eph["DEC"][0])
            rasActs.append(rasAct)
            decsActs.append(decsAct)

        except Exception as e:
            print(e)

    rasInterp = nameCut["RA"]
    decsInterp = nameCut["Dec"]
    limAngle = 21/2
    deltaRa = (rasActs - rasInterp)*60*60
    deltaDec = (decsActs - decsInterp)*60*60
    fig, ax = plt.subplots(1,figsize=(10,8))
    # ax.set_title(name)
    ax.set_xlabel("Delta RA [$''$]")
    ax.set_xlim((-limAngle,limAngle))
    ax.set_ylabel("Delta Dec [$''$]")
    ax.set_ylim((-limAngle,limAngle))

    cmap = "winter"
    # The delta T of the data, and makes a norm for a cmap
    cnorm = mplc.Normalize(np.min(nameCut["MJD"]), np.max(nameCut["MJD"]))
    clb = fig.colorbar(mpcm.ScalarMappable(
        norm=cnorm, cmap=cmap), ax=ax, pad=0.12)
    clb.ax.set_title("Time in MJD", fontsize=16, y=1.03)
    clb.ax.ticklabel_format(useOffset=False)

    ax.scatter(deltaRa, deltaDec, c=nameCut["MJD"], cmap=cmap,norm=cnorm, s=5)
    fig.savefig(f"./Testing Figures/{name}PosCheck.pdf")

