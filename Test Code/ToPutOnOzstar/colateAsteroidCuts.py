"""
Takes all the files on a per cut basis, and gets them together in a per sector basis. 
"""

import pandas as pd
import os
import numpy as np
# from glob import glob #! slow recursivly

def _concatDFList(dfList):
    """Takes list of dataframes, returns concated DF with index reset"""
    allDFs = pd.concat(dfList)
    allDFs.reset_index(inplace=True, drop=True)
    allDFs.rename(columns={"Unnamed: 0":"ID in Cut"}, inplace=True)
    return allDFs

def _dropNans(multiProp):
    for prop in multiProp:
            try:
                propInt = int(prop)
                return prop
            except:
                continue
    
    return prop

def _name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn


def colate_cuts(sector):
    """Take the data from a per cut level to a per sector level. This is ok as all analysis wants to be on a sector level. Does make large files though"""
    # allPropPaths = glob(f"/fred/oz335/TESSdata/Sector{sector}/**/asteroid_properties_cat.csv", recursive=True) #!TOO SLOW

    #* Faster to just itterate and check
    possibleCams = [1,2,3,4]
    possibleCCDs = [1,2,3,4]
    possibleCuts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    basePathsList = []

    for cam in possibleCams:
        for ccd in possibleCCDs:
            for cut in possibleCuts:
                thisPath = f"/fred/oz335/TESSdata/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of16/"
                basePathsList.append(thisPath)

    propDfs = []
    lcDfs = []
    matchDfs = []

    for path in basePathsList:
        propPath = f"{path}asteroid_properties_cat.csv"
        if os.path.exists(propPath):
            propDfs.append(pd.read_csv(propPath))
        
        lcPath = f"{path}asteroid_interpolated_positions.csv"
        if os.path.exists(lcPath):
            lcDfs.append(pd.read_csv(lcDfs))
        
        matchPath = f"{path}Asteroid_InterpAndDetect_Matches.csv"
        if os.path.exists(matchPath):
            matchDfs.append(pd.read_csv(matchPath))
        
    allProps = _concatDFList(propDfs)
    allLCs = _concatDFList(lcDfs)
    allMatches = _concatDFList(matchDfs)


    #TODO clean up duplicate props requery will be somewhere else

    unqNames = np.unique(allProps["Name"])

    nameLists = []

    for name in unqNames:
        
        nCut = _name_cut(allProps,name)
        totPoints = np.sum(nCut["Number of Points"]) 
        newMean = (np.sum(nCut["Mv(mean)"]*nCut["Number of Points"])/totPoints).round(3)
        
        colapsedEles = []

        for eleTitle in ["a", "e", "i", "H"]:
            ele = pd.unique(nCut[eleTitle])
            if len(ele) != 1:
                ele = _dropNans(ele)
            else:
                ele = ele[0]
            colapsedEles.append(ele)

        nameList = [nCut.at[0,"Num"],nCut.at[0,"Name"],newMean,nCut.at[0,"Class"],totPoints, colapsedEles[0], colapsedEles[1], colapsedEles[2], colapsedEles[3]]
        nameLists.append(nameList)

    unqNamesPropDF = pd.DataFrame(nameLists, columns=["Num","Name","Mv(mean)","Class","Number of Points","a","e","i","H"])
    

    allProps.to_csv(f"/fred/oz335/bleicester/Data/{sector}_All_Properties.csv")
    allLCs.to_csv(f"/fred/oz335/bleicester/Data/{sector}_All_LCs.csv")
    allMatches.to_csv(f"/fred/oz335/bleicester/Data/{sector}_All_Matches.csv")


    #TODO, remove the cut files
    #! want to make sure its working properly before doing that

    return f"DONE {sector}"



print(f"Starting colation")
print(colate_cuts(29)) #*To run in just one file 