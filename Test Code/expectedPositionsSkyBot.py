"""expectedPostionsSkyBot.py

Queries SkyBot and gets the positions of asteriods in a TESS field in a defined number of timesteps. Plots them in RA and Dec space

B Leicester, 16/4/24

Last Edited 16/4/24

"""
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
from astroquery.jplhorizons import Horizons
import itertools
from tqdm import tqdm

plt.rcParams.update({
    "font.size": 16,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})


def _Skybotquery(ra, dec, times, radius=10/60, location='C57',cache=False):
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
    url += '-bd={}&'.format(radius)
    url += '-loc={}&'.format(location)
    #TODO add more things here?

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
            res.rename({'# Num ':'Num', ' Name ':'Name',' RA(h) ':'RA', ' DE(deg) ':'Dec', ' Class ':'Class', ' Mv ':'Mv'}, inplace=True, axis='columns')
            res = res[['Num', 'Name','RA','Dec', 'Class', 'Mv', 'epoch']].reset_index(drop=True)
            if df is None:
                df = res
            else:
                df = pd.concat([df,res],ignore_index=True)
    if df is not None:
        df.reset_index(drop=True, inplace=True) #// ! should have inplace=True...
    return df 


def querySB(targetPos:list, qRad:float =10.0, qLoc:str="C57", numTimesteps:int = 27, magLim:float = 20.0)->tuple[pd.DataFrame,npt.ArrayLike]:
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
    dt = (27/numTimesteps)*u.day #a dt in days
    timeList = t_i + dt*np.arange(0,numTimesteps) #list of times constructed sequentially
    result = _Skybotquery(ra_i, dec_i, timeList.jd, radius=qRad, location=qLoc, cache=True)
    brightResult = result.loc[result["Mv"]<= magLim].reset_index(drop=True)

    coords = SkyCoord(brightResult["RA"], brightResult["Dec"], unit=(u.hourangle,u.deg))
    brightResult["RA"] = coords.ra.deg
    brightResult["Dec"] = coords.dec.deg

    return brightResult, timeList


def plotExpectedPos(posDf:pd.DataFrame, timeList:npt.ArrayLike, targetPos:list, magLim:float = 20.0, scaleAlpha:bool = False, minLen:int = 0)->plt.Figure:
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
    Output
        fig: Figure
            The figure of the objects position, coloured by time, in Ra/Dec and ecLon/ecLat space. Faint grey lines connect the points to guide the eye.
    """

    ra_i, dec_i, t_i = targetPos #gets _i values from the input list
    
    w = wcs.WCS(naxis=2) #? sets up blank WCS

    #setting FITS keywords #!guessing what they mean
    w.wcs.crpix = [0,0]
    w.wcs.crval = [posDf["RA"].min(), posDf["Dec"].min()]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.mjdref = [posDf["epoch"].mean(),0]

    #proj. things #! I don't understand
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])

    #Set up figure with WCS
    fig= plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111,projection = w)
    ax.grid(color='black', ls='solid')
    ax.set_xlabel("RA")
    ax.coords[0].set_format_unit(u.deg)
    ax.set_ylabel("DEC")

    #Set up overlay coords
    overlay = ax.get_coords_overlay('geocentrictrueecliptic')
    overlay.grid(color='black', ls='dotted')
    overlay[0].set_axislabel('ecLon')
    overlay[1].set_axislabel('ecLat')

    cmap = "plasma"
    cnorm = mplc.Normalize(np.min(timeList.mjd), np.max(timeList.mjd))# The delta T of the data, and makes a norm for a cmap 
    markers = itertools.cycle((".","x","^","*","D","+","v","o", "<", ">","H","1","2","3","4","8","X","p","d","s","P","h"))#lots of symbols so they aren't repeated much

    #setup for possible alpha scaling
    scaleMult = 0.7 #changes how small alpha can get
    brightest = posDf["Mv"].min() #* Mag. scale is backwards...
    deltaMag = magLim-brightest

    unqNames = pd.unique(posDf["Name"])
    #for a vs e or i plots
    fig2, ax2 = plt.subplots(2, figsize=(6,8))
    ax2[1].set_xlabel("a [au]")
    ax2[0].set_ylabel("e")
    ax2[1].set_ylabel("i [Degrees]")
    gridA = 0.5
    ax2[0].grid(alpha=gridA)
    ax2[1].grid(alpha=gridA)
    ms=8
    hs = {}
    badNames = [] #stops index problem later
    for name in unqNames:
        try:
            horizQ = Horizons(id = name, epochs = t_i.jd, location= "500@10")
            a=horizQ.elements()['a']
            e=horizQ.elements()['e']
            i=horizQ.elements()['incl']
            hs[name]=float(horizQ.elements()['H'])
            ax2[0].scatter(a,e, c="tab:blue", s=ms)
            ax2[1].scatter(a,i, c="tab:blue", s=ms)

        except Exception as e:
            print(f"Name= {name} didn't work, error {str(e)}")
            badNames.append(name)

    hsList = list(hs.values())
    #when alpha scale is by H
    brightest = np.min(hsList) 
    deltaMag = np.max(hsList)-brightest

    scaleStr =f"Alpha Scaling by H:\nH = {brightest}, $\\alpha$ = 1\nH= {deltaMag+brightest}, $\\alpha$ = {round((1-scaleMult),1)}" 

    for i, name in  enumerate(unqNames): #does each name seperatly
        if name in badNames:
            continue
        nameIDs = posDf.index[posDf["Name"]==name]

        coords = SkyCoord(posDf.loc[nameIDs]["RA"],
                        posDf.loc[nameIDs]["Dec"], unit=(u.deg,u.deg))
        pixels = coords.to_pixel(w)

        #scales each objects alpha
        avgMag = posDf.loc[nameIDs]["Mv"].mean()
        avgMag = hs[name]
        if scaleAlpha:
            alpha = (1-scaleMult*((avgMag-brightest)/deltaMag)) #scales the alpha of plotting to the brightness of the object, to give some idea of what might be detected
        else:
            alpha=1

        #plotting
        if len(nameIDs)>minLen:
            ax.plot(pixels[0], pixels[1], c="grey", alpha=0.3) #plots line to help eye track objects
            ax.scatter(pixels[0], pixels[1], marker=next(markers), c=(posDf.loc[nameIDs]["epoch"]-2400000), cmap=cmap, norm=cnorm, label=name, alpha=alpha) #scatters the pixel coords on wsc set axis, uses different symbol for each, with cmap scaled by the time in the sector.
    clb = fig.colorbar(mpcm.ScalarMappable(norm=cnorm, cmap=cmap), ax=ax, pad=0.1)
    clb.ax.set_title("Time in MJD",fontsize=16,y=1.03) 
    clb.ax.ticklabel_format(useOffset=False)   
    centerPix = SkyCoord(ra_i,dec_i, unit=(u.deg,u.deg)).to_pixel(w) #center of seach area in pixels
    ax.scatter(centerPix[0], centerPix[1], marker="+", s=100, c="g") #center marker
    ax.text(-37,-10,s=scaleStr, fontsize=18)

    fig.tight_layout()

    return fig, fig2 #retunrs instead of saves, #? is this useful?
    # fig.savefig(f"./ecCoordsandAlphascaled_ra{ra_i}_dec{dec_i}_t{t_i.mjd}_Mv{magLim}.png") #!CHECK

myTargetPos = [301.60, -38.68, Time("2020-04-17T00:00:00.000", format='isot', scale='utc')]
magLim = 20.0
res, times = querySB(myTargetPos, magLim=magLim)

resFig,resFig2 = plotExpectedPos(res, times, myTargetPos, magLim=magLim, scaleAlpha=True)

resFig.savefig(f"./AlphaScaleText_ra{myTargetPos[0]}_dec{myTargetPos[1]}_t{myTargetPos[2].mjd}_Mv{magLim}.png")
resFig2.savefig(f"./PointSize_ra{myTargetPos[0]}_dec{myTargetPos[1]}_t{myTargetPos[2].mjd}_Mv{magLim}.png")