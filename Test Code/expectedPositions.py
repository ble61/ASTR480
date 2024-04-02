"""expectedPostions.py

Queries MPC and gets the positions of asteriods in a TESS field in a defined number of timesteps. Plots them in RA and Dec space

B Leicester, 2/4/2024

Last Edited 2/4/24

"""
#TODO speed up MPC query.
#! problem is keeping them sequential

#TODO colour by time better
#! problem is colour is chosen from cmap uniformly from number of coord pair for that asteriod, not from times themselves

import numpy as np
import matplotlib.pyplot as plt
import itertools
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from find_asteroid import MPCquery
import pandas as pd


def plotExpectedPos(posArr, times, lims, names):
    """Plots each asteroid as a sepeate (hopfully) marker
    Input:
        posArr: A 3d ndArray, with each object having its own RA and DEC collumns that are the lenght of *times*, to get each asteriod as a "line" of points

        times: A 1d arrayLike, meant as a sequence for colouring, actual time is not needed, as that is defined by TESS sector. Though it can be used as the same list should be used to query MPC

        lims: 4 float list, for the x and y min and max limits for the plot

        names: 1d arrayLike, should be the same lenght as the 1st dimension of *posArr*    
    
        
    Outputs:
        plt.figure of the tracks of the asteroids
    """
    #Sets up plot
    fig, ax = plt.subplots(1,figsize =(12,12))
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_xlim([lims[0],lims[1]])
    ax.set_ylim([lims[2],lims[3]])

    markers = itertools.cycle((".","x","^","*","D","+","v","o", "<", ">","H","1","2","3","4","8","X","p","d","s","P","h")) #marker cycle
    
    #plots each asteroid
    for i in range(posArr.shape[0]):
        ax.scatter(posArr[i,0],posArr[i,1], marker=next(markers), c=times, cmap="plasma", label = names[i])
    
    plt.scatter(ra_i, dec_i, marker="+", s=100, c="g") #Center of seach area

    # ax.legend() #! Not really working, stops after a few symbols

    fig.savefig(f"./posTracks_ra{ra_i}_dec{dec_i}_t{t_i}.png")

#sets up positions and time
#TODO get _i vals as inputs?    
ra_i, dec_i = 301.60, -38.68
numTimesteps = 27 #! often small as takes a long time
t_i=Time("2020-04-17T00:00:00.000", format='isot', scale='utc')
dt = (27/numTimesteps)*u.day
timeList = t_i + dt*np.arange(0,numTimesteps)

#MPC query/ies
resList = []
for pos, time in enumerate(timeList.jd):  
    queryRes = MPCquery(ra_i,dec_i,time, 300, limit="18.0",obscode="C57") #! This is the time sink, especially with so many times
    resList.append(queryRes)

#Gets the unique asteroid names, to be used to get tracks of the same asteroid with time
namesList = [resList[i]["name"] for i in range(len(resList))]
names = pd.concat(namesList)
unqNames = pd.unique(names)
unqNames = np.delete(unqNames, np.where(unqNames==""))

#To get name and the RA and Dec with time into a list of lists to be made into an array later
valList = []
for j, name in enumerate(unqNames):    
    if name == "": #Incase any blanks slip through, shouldn't be needed
        continue
    raList = []
    decList = []
    for i in range(len(resList)):
        df = resList[i]
        index= df.index[df["name"]==name]
        try:
            raList.append(df.iloc[index[0]]["RA"])
            decList.append(df.iloc[index[0]]["Dec"])
        except:
            raList.append(np.nan)
            decList.append(np.nan)
    valList.append([name, raList, decList])

#puts list values into an array, of the right shape to be plotted
numObjects = unqNames.shape[0]
objPosArr = np.empty([numObjects,2,numTimesteps])
for i in range(numObjects):
    ra = valList[i][1]
    dec = valList[i][2]
    coords = SkyCoord(ra, dec, unit=(u.hourangle,u.deg)) #converts from annoying unitless values with spaces in them into values that can be plotted
    objPosArr[i,0] = coords.ra
    objPosArr[i,1] = coords.dec


limits = [np.nanmin(objPosArr[:,0,:])-0.5,np.nanmax(objPosArr[:,0,:])+0.5,np.nanmin(objPosArr[:,1,:])-0.5,np.nanmax(objPosArr[:,1,:])+0.5]#sets lims based on min and max values, with a 0.5 deg buffer for visualisation


plotExpectedPos(objPosArr, timeList.jd, limits, unqNames) #calls plotting function
