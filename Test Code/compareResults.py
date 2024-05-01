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

interpRes = pd.read_csv("./InterpolatedQuerryResultTESSdata_Sector2_Cam1_Ccd1_Cut1of16_wcs")



# #TODO get csv from Ryan of problems. 
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

detectedSources = pd.read_csv("./TESSdata_Sector2_Cam1_Ccd1_Cut1of16_detected_sources.csv", usecols=["ra", "dec", "mjd", "mag", "Type"])

#     indexs= interpRes.index[interpRes["Name"]==name]
#     cutItrpDf = interpRes.loc[indexs]

indexsOfInterest = detectedSources.index[detectedSources["Type"] == "0"]

detectedSources = detectedSources.loc[indexsOfInterest]

detectedSources["jd"] = detectedSources["mjd"] +2400000


#* use kd tree to compare the interpolated values to the ones with a random change to them
#! stolen from stack overflow: https://stackoverflow.com/questions/67099008/matching-nearest-values-in-two-dataframes-of-different-lengths 

tree = KDTree(interpRes[['RA','Dec','epoch']].values) #make tree

dists, indices = tree.query(detectedSources[['ra','dec','jd']].values, k=1) #querry tree for closest matches #!check col names

fts = [c for c in interpRes.columns]

for c in fts:
    detectedSources[c] = interpRes[c].values[indices]

#!end stolen

detectedSources.rename(columns={"Unnamed: 0":"interpIndex"}, errors="raise", inplace=True) #fix col name issue

# print(randRes['Namerand'].compare(randRes["Name"]))
# makes sure the names are the same. sanity check for now, as asteroids in TESS won't have the name, which is the point. should be empty. 
# with larger errors, only a the same few are getting confused.
#even with higher sampling and dimmer maglim

#TODO completeness

# unqNames = pd.unique(interpRes["Name"])

# obsForDetect = 1 #number of observations needed to count a detection

# interpedMv = []
# foundMv = []


# for name in unqNames:
#     indexs= interpRes.index[interpRes["Name"]==name]
#     cutItrpDf = interpRes.loc[indexs]
#     avgItrpMv = cutItrpDf["Mv"].mean()
#     interpedMv.append(avgItrpMv) 

#     resIds = detectedSources.index[detectedSources['Namerand']==name]
#     if len(resIds)>=obsForDetect:
#         cutResDf = detectedSources.loc[resIds]
#         avgResMv = cutResDf["Mv"].mean()
#         foundMv.append(avgResMv)
#     # try:
#     #     resIds = randRes.index[randRes['Namerand']==name]
#     #     cutResDf = randRes.loc[resIds]
#     #     avgResMv = cutResDf["Mv"].mean()
#     #     foundMv.append(avgResMv)
#     # except:
#     #     #someing about name not being found
#     #     continue

# maxMv = np.ceil(np.max(interpedMv))
# minMv = np.floor(np.min(interpedMv))
# binSize = 0.25
# nBins = int((maxMv-minMv)/binSize)


# interpMvHist, itrpBins = np.histogram(interpedMv, bins = nBins, range=(minMv, maxMv))

# foundMvHist, foundBins = np.histogram(foundMv, bins = nBins, range=(minMv, maxMv))

# print(np.sum(foundMvHist))

# completenessMvHist = foundMvHist/interpMvHist

# plt.stairs(completenessMvHist, itrpBins)
