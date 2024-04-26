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

interpRes = pd.read_csv("./InterpolatedQuerryResult_ra301.6_dec-38.68_t58956.0_Mv20.csv")

#TODO get csv from Ryan of problems. 
# will have to rename vars to be readable
#*gets random offset to make matches from
myTargetPos = [301.60, -38.68, Time("2020-04-17T00:00:00.000", format='isot', scale='utc')]

resForRand, randTimesOut = querySB(targetPos=myTargetPos, numTimesteps=12, magLim=20) #querry at differnt time scale

#change at 0.001 place for RA and Dec
#21 arcsec is ~0.006 of a degree which is size of TESS pixel 
#p/m 0.018 deg is pm 3 pixels ((18-36r)/1000=p/m 0.018)
randRAs = resForRand["RA"] + (18-36*np.random.rand(len(resForRand.index)))/1000
randDecs = resForRand["Dec"] + (18-36*np.random.rand(len(resForRand.index)))/1000

#Slightly larger change in time, about error in 1/2 hr. (1/48 of a day is 0.02)
randTimes = resForRand["epoch"] + (3-6*np.random.rand(len(resForRand.index)))/100

randRes = pd.DataFrame({"Namerand": resForRand["Name"],"RArand":randRAs, "Decrand":randDecs, "epochrand":randTimes}) #into a df.

#* use kd tree to compare the interpolated values to the ones with a random change to them
#! stolen from stack overflow: https://stackoverflow.com/questions/67099008/matching-nearest-values-in-two-dataframes-of-different-lengths 

tree = KDTree(interpRes[['RA','Dec','epoch']].values) #make tree

dists, indices = tree.query(randRes[['RArand','Decrand','epochrand']].values, k=1) #querry tree for closest matches

fts = [c for c in interpRes.columns]

for c in fts:
    randRes[c] = interpRes[c].values[indices]

#!end stolen

randRes.rename(columns={"Unnamed: 0":"interpIndex"}, errors="raise", inplace=True) #fix col name issue

print(randRes['Namerand'].compare(randRes["Name"]))
# makes sure the names are the same. sanity check for now, as asteroids in TESS won't have the name, which is the point. should be empty. 
# with larger errors, only a the same few are getting confused.
#even with higher sampling and dimmer maglim