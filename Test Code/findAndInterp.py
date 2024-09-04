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
    for time in tqdm(times, desc='Querying for SSOs'):
        url_queried = url + 'EPOCH={}'.format(time)
        response = download_file(url_queried, cache=cache)
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
    return df


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
    result = pd.DataFrame()
    
    while len(result)<numTimesteps: #so if query timesout, it will restart with the cache
        try:
            result = _Skybotquery(ra_i, dec_i, timeList.jd,
                          radius=qRad, location=qLoc, cache=True) #timeout throws error, so need to be in try/except to not close
        except:
            continue #restarts query with cache

        if type(result) != "pandas.core.frame.DataFrame": 
            #! need something for when no asteroids are in cut.
            #! else everthing else breaks too. 
            break #to get out of while
    

    brightResult = result.loc[result["Mv"] <= magLim].reset_index(drop=True)

    brightResult["Name"] = [name.strip() for name in brightResult["Name"]]

    brightResult["MJD"] = brightResult["epoch"]-timeOffset

    brightResult.drop(columns="epoch", inplace=True)

    coords = SkyCoord(
        brightResult["RA"], brightResult["Dec"], unit=(u.hourangle, u.deg))
    brightResult["RA"] = coords.ra.deg
    brightResult["Dec"] = coords.dec.deg

    return brightResult, timeList


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

        except Exception as ex:
            print(f"{name} failed horizons check because:\n {ex}")
            a = np.NaN
            e = np.NaN
            i = np.NaN
            H = np.NaN

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

    res, timeList = querySB(setupQuery(sector, cam, ccd, cut), numTimesteps=54, qRad=3.05)

    unqNames = np.unique(res["Name"])
    print(len(unqNames))
    propertiesList = []

    for name in unqNames:
        nCut = name_cut(res, name, colName="Name")
        avgMv = np.mean(nCut["Mv"]).round(3)
        num = nCut.at[0,"Num"]
        astrClass = nCut.at[0,"Class"]  

        propertiesList.append([num,name,avgMv,astrClass])

    asteroidPropertiesDf = pd.DataFrame(propertiesList,columns=["Num","Name","Mv(mean)","Class"])

    withEles = get_properties_Horizons(asteroidPropertiesDf, timeList[0])

    fname = f"asteroids_in_{sector}_{cam}_{ccd}_{cut}_properties"
    withEles.to_csv(f"{fname}.csv")

    return res


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
    print(len(unqNames))
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
    
    # print(len(dfsList))
    interpRes = pd.concat(dfsList) #?puts everything back together
    
    
    namesAfter =np.unique(interpRes["Name"])
    # print(len(namesAfter))
    namesDroped = np.setdiff1d(unqNames, namesAfter) #!names with only 1 point from query will be missed, as there is nothing to interpolate between.

    # print(namesDroped)

    #final clean
    interpRes.reset_index(drop=True, inplace=True)
    interpRes.drop(columns=["Mv","Class"], inplace=True)
    


    return interpRes


sector = 22
cam = 1
ccd =3
cut = 7

resDf = find_asteroids(sector,cam,ccd,cut)
interpDf = interplolation_of_pos(resDf,sector)
fname = f"InterpolatedQuerryResult_{sector}_{cam}_{ccd}_{cut}"
interpDf.to_csv(f"{fname}.csv")

#TODO 
#//save out df of pos properties name:pos(t,x,y) from interpolations (name repeated)
#//save out df of name:num:avgMag:a:e:i:H + other properties of asteroid ?known period? 1 row per name. Other file will add found Period, lc properties etc 


unqNames = np.unique(interpDf["Name"])

numNames = len(unqNames)

# unqNames=["Bernoulli"]

# for name in unqNames:
#     # if np.random.rand()< 1/numNames:
#     ids = interpDf.index[interpDf["Name"]==name]
#     nameCut = interpDf.loc[ids]
#     # timeStart = Time(nameCut["epoch"].min(), format="jd")
#     # timeEnd = Time(nameCut["epoch"].max(), format="jd")
    
#     # horizQ = Horizons(id = name, epochs = {"start":str(tforHorz(timeStart)), "stop":str(tforHorz(timeEnd)), "step":"30m"}, location= "500@-95")
#     rasActs = []
#     decsActs = []
#     for time in nameCut["MJD"]:
#         try:
#             horizQ = Horizons(id = name, epochs =time, location= "500@-95")
#             eph = horizQ.ephemerides()
#             rasAct = float(eph["RA"][0])
#             decsAct = float(eph["DEC"][0])
#             rasActs.append(rasAct)
#             decsActs.append(decsAct)

#         except Exception as e:
#             print(e)

#     rasInterp = nameCut["RA"]
#     decsInterp = nameCut["Dec"]

#     deltaRa = rasActs - rasInterp
#     deltaDec = decsActs - decsInterp
#     fig, ax = plt.subplots(1)
#     ax.set_title(name)
#     ax.set_xlabel("Delta RA")
#     ax.set_ylabel("Delta Dec")
#     ax.scatter(deltaRa, deltaDec, c=nameCut["MJD"], cmap="magma")

