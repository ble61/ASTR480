import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy import wcs
from astropy.utils.data import download_file
from astropy.io import fits

# %matplotlib widget

plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})


timeOffset = 2400000.5


def load_matches(sector, cam, ccd, cut, dirPath = "./"):
    matchesDF = pd.read_csv(f"{dirPath}{sector}_{cam}_{ccd}_{cut}_InterpAndDetect_Matches.csv")
    matchesDF.drop(columns="Unnamed: 0", inplace=True)
    return matchesDF

def load_interpolations(sector, cam, ccd, cut, dirPath = "./"):
    interpDF = pd.read_csv(f"{dirPath}InterpolatedQuerryResult_{sector}_{cam}_{ccd}_{cut}.csv")
    interpDF.drop(columns="Unnamed: 0", inplace=True)
    return interpDF

def load_fluxes(sector, cam, ccd, cut, dirPath = "../Ozdata/"):
    reduFlux = np.load(f"{dirPath}sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_ReducedFlux.npy")
    return reduFlux

def load_times(sector, cam, ccd, cut, dirPath = "../Ozdata/"):
    timesFromOz = np.load(f"{dirPath}sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of16_Times.npy")
    return timesFromOz

def name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    return df.loc[df.index[df[colName]==name]]



#*Global variables
#TODO take these from an input

sector = 22
cam = 1
ccd = 3
cut = 8

reducedFluxes = load_fluxes(sector, cam, ccd, cut)
frameTimes = load_times(sector, cam, ccd, cut)



