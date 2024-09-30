import pandas as pd
import os
# from glob import glob #! slow recursivly

def _concatDFList(dfList):
    """Takes list of dataframes, returns concated DF with index reset"""
    allDFs = pd.concat(dfList)
    allDFs.reset_index(inplace=True, drop=True)
    allDFs.rename(columns={"Unnamed: 0":"ID in Cut"}, inplace=True)
    return allDFs

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
                thisPath = f"/fred/oz335/TESSdata/Sector{sector}/Cam{cam}/Ccd{ccd}/cut{cut}of16/"
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


    allProps.to_csv(f"/fred/oz335/bleicester/Data/{sector}_All_Properties.csv")
    allLCs.to_csv(f"/fred/oz335/bleicester/Data/{sector}_All_LCs.csv")
    allMatches.to_csv(f"/fred/oz335/bleicester/Data/{sector}_All_Matches.csv")


