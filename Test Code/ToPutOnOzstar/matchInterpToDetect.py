import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def _load_lcs(sector, cam, ccd, cut, dirPath = "./"):
    lcDF = pd.read_csv(f"{dirPath}InterpLC.csv")
    #!need to not reset index lcDF.drop(columns="Unnamed: 0", inplace=True)
    return lcDF

def _load_fluxes(sector, cam, ccd, cut, dirPath = "../OzData/"):
    reduFlux = np.load(f"{dirPath}sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_ReducedFlux.npy")
    return reduFlux

def _load_times(sector, cam, ccd, cut, dirPath = "../OzData/"):
    timesFromOz = np.load(f"{dirPath}sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_Times.npy")
    return timesFromOz

def _load_detection(dirPath = "../OzData/"):
    detectedSourcesAll = pd.read_csv(f"{dirPath}detected_sources.csv", usecols=["ra", "dec", "mjd", "flux", "Type"])

    indexsOfInterest = detectedSourcesAll.index[(detectedSourcesAll["Type"] == "0")]

    indexsOfInterest = detectedSourcesAll.index[(detectedSourcesAll["Type"] == "0") | (detectedSourcesAll["Type"] == "Asteroid")] #Just the unknown ones

    detectedSources = detectedSourcesAll.loc[indexsOfInterest]


    detectedSources.rename(columns={"ra":"RA", "dec":"Dec", "mjd":"MJD"}, inplace=True)

    detectedSources.reset_index(drop=True, inplace=True)

    return detectedSources

def _load_properties(dirPath = "../OzData/"):
    propDF = pd.read_csv(f"{dirPath}asteroid_properties_cat.csv")
    propDF.drop(columns="Unnamed: 0", inplace=True)
    return propDF

def _name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn

def _use_KD_tree(df1, df2, cols, maxDist:float=0.01,maxTimeSep:float=0.1,k:int=1):
    tree = KDTree(df1[cols].values)
    dists, indices =  tree.query(df2[cols].values, k=k)

    fts = [c for c in df1.columns]
    df2["distToMatch"] = dists 
    for c in fts:
        df2[f"{c}Match"]= df1[c].values[indices]


    df2.rename(columns={'Unnamed: 0Match':"IDMatch"}, inplace=True, errors="raise")

    #// TODO compare time at indices returned
    timeInds = df2.index[np.abs((df2["Time"]-df2["TimeMatch"]))<maxTimeSep]


    df2=df2.loc[timeInds]


    #// TODO take only nearest match for each point
    smallDistInds = df2.index[df2["distToMatch"]<maxDist]
    
    df2= df2.loc[smallDistInds]

    df2 = df2.sort_values("distToMatch").drop_duplicates(subset=["IDMatch"]).sort_index() #drops all the duplicate IDs, as very point should be 1:1 not n:1

    df2.reset_index(drop=False, inplace=True)
    df2.rename(columns={"index":"DetectIDs"}, inplace=True)

    return df2


def find_matches(sector, cam, ccd, cut):
    """Uses a KDTree to find the matches to the interpolated positions"""
    
    basePath = f"../TESSdata/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of16/" 

    lcDF = _load_lcs(sector, cam, ccd, cut, basePath)

    if len(lcDF.index)==1:
                try:
                    if lcDF.at[0, "Name"][:12] == "There Were N":
                        print(f"No asteroids at {basePath}") #TODO save a blank one?
                        return
                except:
                    pass

    detectDF = _load_detection(sector, cam, ccd, cut, basePath)

    colsToUse = ["RA", "Dec"]

    matchDF = _use_KD_tree(lcDF.copy(deep=True), detectDF.copy(deep=True), cols=colsToUse, maxDist=21/3600, maxTimeSep=0.025)

    matchDF.to_csv(f"{basePath}Asteroid_InterpAndDetect_Matches.csv")
    
    #*Updating properties with a new col: Num Matches
    propDF = _load_properties(basePath)
    numMatches = []
    for name in propDF["Name"]:
        nCut = _name_cut(matchDF)
        numMatch = len(nCut.index) #* this can be 0, and that is what is desired for an asteroid with no match
        numMatches.append(numMatch)

    propDF["Num Matches"] = numMatches

    propDF.to_csv(f"{basePath}asteroid_properties_cat.csv")

    return



