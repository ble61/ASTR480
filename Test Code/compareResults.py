'''compareResults.py
Uses a KDTree to compare interpolated results of previous querry to other data. 

Currently faking random data. 

B Leicester 26/4/24

Last edited 26/4/24
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from scipy.spatial import KDTree

from expectedPositionsSkyBot import querySB
# from sklearn.cluster import DBSCAN


def useKDTree(df1, df2, cols, maxDist:float=0.01,maxTimeSep:float=0.1,k:int=1):
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
 

# def cross_match_DB(cat1,cat2,distance=2*21,njobs=-1):
#     all_ra = np.append(cat1['ra'].values,cat2['ra'].values)
#     all_dec = np.append(cat1['dec'].values,cat2['dec'].values)
#     cat2_ind = len(cat1)

#     p = np.array([all_ra,all_dec]).T
#     cluster = DBSCAN(eps=distance/60**2,min_samples=2,n_jobs=njobs).fit(p)
#     labels = cluster.labels_
#     unique_labels = set(labels)
#     cat1_id = []
#     cat2_id = []matches.rename(columns={'Unnamed: 0Match':"IDMatch"}, inplace=True, errors="raise")y():
#                 if len(inds) > 2:
#                     dra = all_ra[np.where(labels == label)[0]]
#                     ddec = all_dec[np.where(labels == label)[0]]
#                     d = (dra - dra[0])**2 + (ddec - ddec[0])**2
#                     args = np.argsort(d)
#                     inds = inds[args[:2]]
#                 cat1_id += [inds[0]]
#                 cat2_id += [inds[1] - cat2_ind]
#     return cat1_id,cat2_id


sector = 22
cam = 1
ccd = 3
cut = 7


interpRes = pd.read_csv(f"./InterpolatedQuerryResult_{sector}_{cam}_{ccd}_{cut}.csv")


# #// TODO get csv from Ryan of problems. 
# # will have to rename vars to be readable
# #*gets random offset to make matches from
# obsPos = [301.60, -38.68, Time("2020-04-17T00:00:00.000", format='isot', scale='utc')]

# resForRand, randTimesOut = querySB(targetPos=obsPos, numTimesteps=12, magLim=19.5) #querry at differnt time scale

# #change at 0.001 place for RA and Dec
# #21 arcsec is ~0.006 of a degree which is size of TESS pixel 
# #p/m 0.018 deg is pm 3 pixels ((18-36r)/1000=p/m 0.018)
# randRAs = resForRand["RA"] + (18-36*np.random.rand(len(resForRand.index)))/1000
# randDecs = resForRand["Dec"] + (18-36*np.random.rand(len(resForRand.index)))/1000

# #Slightly larger change in time, about error in 1/2 hr. (1/48 of a day is 0.02)
# randTimes = resForRand["epoch"] + (3-6*np.random.rand(len(resForRand.index)))/100

# randRes = pd.DataFrame({"Namerand": resForRand["Name"],"RArand":randRAs, "Decrand":randDecs, "epochrand":randTimes}) #into a df.
#detectedSources = randRes

detectedSourcesAll = pd.read_csv(f"../OzData/{sector}_{cam}_{ccd}_{cut}_detected_sources.csv", usecols=["ra", "dec", "mjd", "flux", "Type"])

#     indexs= interpRes.index[interpRes["Name"]==name]
#     cutItrpDf = interpRes.loc[indexs]

print(pd.unique(detectedSourcesAll["Type"]))

indexsOfInterest = detectedSourcesAll.index[(detectedSourcesAll["Type"] == "0")]

indexsOfInterest = detectedSourcesAll.index[(detectedSourcesAll["Type"] == "0") | (detectedSourcesAll["Type"] == "Asteroid")] #// TODO figure out what code is for asteroid.

detectedSources = detectedSourcesAll.loc[indexsOfInterest]

detectedSources["jd"] = detectedSources["mjd"] +2400000.5
detectedSources.reset_index(drop=True, inplace=True)

colsToUse = ["RA", "Dec"]

interpRes.rename(columns={"epoch":"Time"}, inplace=True)


detectedSources.rename(columns={"ra":"RA", "dec":"Dec", "jd":"Time"}, inplace=True)

matches = useKDTree(df1=interpRes.copy(deep=True), df2=detectedSources.copy(deep=True), cols=colsToUse, maxDist=21/3600, maxTimeSep=0.025)

matches.to_csv(f"./{sector}_{cam}_{ccd}_{cut}_InterpAndDetect_Matches.csv")


# #* use kd tree to compare the interpolated values to the ones with a random change to them
# #! stolen from stack overflow: https://stackoverflow.com/questions/67099008/matching-nearest-values-in-two-dataframes-of-different-lengths 

# tree = KDTree(interpRes[['RA','Dec','epoch']].values) #make tree

# dists, indices = tree.query(detectedSources[['ra','dec','jd']].values, k=1) #querry tree for closest matches #!check col names

# fts = [c for c in interpRes.columns]

# for c in fts:
#     detectedSources[c] = interpRes[c].values[indices]

# #!end stolen

# detectedSources.rename(columns={"Unnamed: 0":"interpIndex"}, errors="raise", inplace=True) #fix col name issue

# print(randRes['Namerand'].compare(randRes["Name"]))
# makes sure the names are the same. sanity check for now, as asteroids in TESS won't have the name, which is the point. should be empty. 
# with larger errors, only a the same few are getting confused.
#even with higher sampling and dimmer maglim

#TODO completeness

unqNames = pd.unique(interpRes["Name"])

obsForDetect=40 #number of observations needed to count a detection


interpedMv = []
foundMv = []


for name in unqNames:
    indexs= interpRes.index[interpRes["Name"]==name]
    cutItrpDf = interpRes.loc[indexs]
    avgItrpMv = cutItrpDf["Mv"].mean()
    interpedMv.append(avgItrpMv) 

    resIds = matches.index[matches['NameMatch']==name]
    if len(resIds)>=obsForDetect:
        cutResDf = matches.loc[resIds]
        avgResMv = cutResDf["MvMatch"].mean()
        foundMv.append(avgResMv)
    # try:
    #     resIds = randRes.index[randRes['Namerand']==name]
    #     cutResDf = randRes.loc[resIds]
    #     avgResMv = cutResDf["Mv"].mean()
    #     foundMv.append(avgResMv)
    # except:
    #     #someing about name not being found
    #     continue

maxMv = np.ceil(np.max(interpedMv))
minMv = np.floor(np.min(interpedMv))
binSize = 0.25
nBins = int((maxMv-minMv)//binSize)


interpMvHist, itrpBins = np.histogram(interpedMv, bins = nBins, range=(minMv, maxMv))

foundMvHist, foundBins = np.histogram(foundMv, bins = nBins, range=(minMv, maxMv))

print(f"The number of matched detections with >= {obsForDetect} observations is {np.sum(foundMvHist)}")

completenessMvHist = foundMvHist/interpMvHist

plt.stairs(completenessMvHist, itrpBins)



for name in unqNames:
    nameIDs = matches.index[matches["NameMatch"]==name]
    nameCut = matches.loc[nameIDs]
    
    if len(nameIDs)> obsForDetect:
        plt.figure()
        plt.scatter(nameCut["Time"], nameCut["flux"], label=name)
        plt.legend()

    if name == " Ruff " or name ==" Lincoln " or name ==" 1999 JE82 " or name == " Henry ":
        nameCut.to_csv(f"./{name}Matches.csv")





