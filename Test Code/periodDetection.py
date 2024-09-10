import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import lightkurve as lk

import astropy.units as u
from astropy.timeseries import LombScargle as lsp

%matplotlib widget 


plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})

#TODO Nyquist criterion
#TODO LC database

def detect_period_lk(times, fluxes, fluxErrs):
    """
    DEPRICATED
    
    Calculates the periodogram using LightKurve and returns the period at the max power.
    """
    lc =lk.LightCurve(data=None, time=times, flux=fluxes, flux_err=fluxErrs)
    # lc = lc.normalize()
    pg = lc.to_periodogram()
    period = pg.period_at_max_power

    return period.value

def detect_period_ap(times, fluxes, plotting = False, knownFreq = None):
    """
    Calculates the periodogram with astropy and returns the period at the max power.
    """
    #* Seems to be sec 27 and 56 when it changes
    #12hr queries constant
    if sector <27: 
        nyquistP = 1*60*60*u.s #1/2hr ffi, so 1hr nyqist
    elif sector>=27 and sector<56:
        nyquistP = 20*60*u.s #1/6hr (10 min) ffi
    else: 
        nyquistP = 400*u.s   # 200 s ffi
    

    minFreq = 1/(2*(times.max()-times.min())*u.day).to(u.s)
    # minFreq = 1e-5/u.s

    maxFreq = 1/nyquistP

    lsper = lsp((u.Quantity(times, u.day)).to(u.s), fluxes)
    freqs, powers = lsper.autopower(minimum_frequency=minFreq, maximum_frequency=maxFreq)
    bestPow = np.max(powers)
    bestFreq = freqs[np.argmax(powers)]

    t_fit=(np.linspace(times.min(),times.max(),1000)*u.day).to(u.s)
    y_fit = lsper.model(t_fit, bestFreq)
    modelParams = lsper.model_parameters(bestFreq)
    f_a_prob = lsper.false_alarm_probability(np.max(powers),samples_per_peak=50, minimum_frequency=minFreq, maximum_frequency=maxFreq)

    if plotting:
        fig, ax = plt.subplots(2, figsize = (8,10))

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set(xlabel="Log Frequency", ylabel="Power")
        ax[0].plot(freqs, powers, c="k", label="Periodogram")
        ax[0].scatter(bestFreq,bestPow, c="gold",marker="*", s=100, label=f"Best Frequency = {bestFreq.round(7)} \nFalse Alarm Probability = {f_a_prob.round(3)}")

        if knownFreq is not None:
            ax[0].axvline(knownFreq, linestyle=":", c="tab:blue", label = f"Known F={knownFreq.round(7)}")
            ax[0].axvline(2*knownFreq, linestyle=":", c="tab:purple", label = f"Known F alias (2*)")
            ax[0].axvline(0.5*knownFreq, linestyle=":", c="tab:red", label = f"Known F alias (0.5*)")

        ax[0].legend(fontsize=10)
        meanFlux = np.mean(fluxes)
        ax[1].set(xlabel="Time [JD]", ylabel="Flux")#, ylim=(meanFlux-100, meanFlux+100))
        ax[1].scatter(times, fluxes, label = "Light curve", c="tab:orange", marker = "d", s=10)
        ax[1].plot(t_fit.to(u.day),y_fit, ls="--", c="k", label="Model")


    return (1/bestFreq)/u.s, f_a_prob, modelParams, bestPow


def load_interps(sector, cam, ccd, cut):
    interpDf = pd.read_csv(f"./InterpolatedQuerryResult_{sector}_{cam}_{ccd}_{cut}.csv")
    interpDf.drop(columns="Unnamed: 0", inplace=True)
    return interpDf

def load_interps(sector, cam, ccd, cut):
    lcDF = pd.read_csv(f"./Interps_with_lc_{sector}_{cam}_{ccd}_{cut}.csv")
    lcDF.drop(columns="Unnamed: 0", inplace=True)
    return lcDF

def load_matches(sector, cam, ccd, cut):
    matchesDF = pd.read_csv(f"./{sector}_{cam}_{ccd}_{cut}_InterpAndDetect_Matches.csv")
    matchesDF.drop(columns="Unnamed: 0", inplace=True)
    return matchesDF

def name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn

numObser = 3 #*any less than 3 and NaNs come up, or it completely breaks
#TODO, what should it be for a reliable P

def compute_periods_lk(posDF):
    """DEPRECATED"""
    unqNames = pd.unique(posDF["Name"]) #!name col better
    periodList_lk = []
    badCount = 0

    for name in unqNames:
        nameDf = name_cut(posDF, name, colName="Name")
        if nameDf.shape[0]<=numObser:
            badCount+=1
            # print(f"{badCount}. not enough points (<={numObser}) for {name}")
            continue
        time = nameDf["MJD"].values
        flux = nameDf["Flux"].values
        fluxErr = np.ones_like(flux)*1e-4 #!Have to make up err for lk
        period_lk = detect_period_lk(time,flux,fluxErr)
        periodList_lk.append([name, period_lk,np.nan, np.nan]) #NaNs as LK doesn't spit those values out

    lkPeriods = pd.DataFrame(periodList_lk, columns=["Name", "Best Period","False Alarm Probability", "Model Parameters"])

    return lkPeriods, badCount

def compute_periods_ap(posDF):
    unqNames = pd.unique(posDF["Name"])
    periodList_ap = []
    badCount = 0

    for name in unqNames:
        nameDf = name_cut(posDF, name, colName="Name")
        
        nameProperties = name_cut(astrData, name, colName="Name")
        
        if nameDf.shape[0]<=numObser:# or nameProperties["Over Background Limit"][0] ==False:
            badCount+=1
            # print(f"{badCount}. not enough points (<={numObser}) for {name}")
            continue
        time = nameDf["MJD"].values
        flux = nameDf["COM Flux"].values

        period_ap, f_a_Prob, theta, maxPow = detect_period_ap(time, flux)
        periodList_ap.append([name, period_ap, period_ap/(60*60), period_ap/(60*60*24), f_a_Prob, theta, maxPow])

    apPeriods = pd.DataFrame(periodList_ap, columns=["Name", "Best Period [Seconds]", "Best Period [Hours]","Best Period [Days]",  "False Alarm Probability", "Model Parameters", "Max Power"])

    return apPeriods, badCount

def singleNameLSP(posDf, name, knownFreq=None):
    cut = name_cut(posDf,name, colName="Name")
    detect_period_ap(cut["MJD"],cut["COM Flux"],plotting=True, knownFreq=knownFreq)

def model_from_params(theta:list, period:float, times, avgFlux:float):
    """
    Period in same time unit as times... conversion is not done implicitly (Tho all should be floats, not quantities)
    """
    trigArg = times*(2*np.pi/(period))

    model = avgFlux+ theta[0] + theta[1]*np.sin(trigArg) +theta[2]*np.cos(trigArg)

    return model



sector = 22
cam = 1
ccd = 3
cut = 7


interpLcDF = load_interps(22,1,3,7)

astrData = pd.read_csv(f"./asteroids_in_{sector}_{cam}_{ccd}_{cut}_properties.csv")

astrData.drop(columns = ["Unnamed: 0"], inplace=True)


unqNames = np.unique(interpLcDF["Name"])

# lkPer, badCountlk = compute_periods_lk(interpLcDF)

apPer, badCountap = compute_periods_ap(interpLcDF)


#Combine with what we already have for each asteroid



allData = astrData.merge(apPer, how="outer", on="Name")

allData.to_csv(f"asteroids_in_{sector}_{cam}_{ccd}_{cut}_with_Periods.csv")



# numPerlk = len(lkPer.index)
numPerap = len(apPer.index)
# print(badCountlk, numPerlk)
print(badCountap, numPerap)

fig, ax = plt.subplots()
# ax.errorbar(lkPer.index, lkPer["Best Period"], fmt=".", c="tab:blue", capsize = 2, label="Lightkurve")

#! Not error in period, but level of uncertanty due to f_a_Prob
ax.errorbar(apPer.index, apPer["Best Period [Days]"], apPer["False Alarm Probability"]*10, fmt=".",c="tab:orange", capsize=2, label="All")

bigPowLim = 0.2

bigPows = apPer.iloc[apPer.index[apPer["Max Power"] > bigPowLim]]


#?WHY DO I HAVE TO DO THIS????
ids = bigPows.index.values
peri = bigPows["Best Period [Days]"].values
falAP = bigPows["False Alarm Probability"].values*10
ax.errorbar(ids, peri, falAP, fmt=".",c="tab:blue", capsize=2, label=f"Max Power >{bigPowLim}")

ax.set_ylabel("Period [days]")
ax.set_xlabel("Index")
ax.legend()


# maxPlk = lkPer.loc[lkPer["Best Period"].idxmax()]
maxPap = apPer.loc[apPer["Best Period [Seconds]"].idxmax()]
# print(maxPlk)
print("")
#* print(maxPap)


# singleNameLSP(interpLcDF,maxPap["Name"])


# minPlk = lkPer.loc[lkPer["Best Period"].idxmin()]
minPap = apPer.loc[apPer["Best Period [Seconds]"].idxmin()]
print("")
# print(minPlk)
print("")
#* print(minPap)


# singleNameLSP(interpLcDF,minPap["Name"])





lcdb = pd.read_csv("lcdbUseful.csv", sep=",")

inLCDBList = []

for i,name in enumerate(apPer["Name"]):
    name = name.strip()

    if name in lcdb["Name"].values:
        foundP = apPer.at[i,"Best Period [Hours]"].value
        falAlP = apPer.at[i, "False Alarm Probability"].value
        knownRow = lcdb.index[lcdb["Name"]==name]
        knownP = lcdb.at[knownRow.values[0], "Period [Hours]"]
        
        inLCDBList.append([name, knownP, foundP, falAlP])

compedPs = pd.DataFrame(inLCDBList, columns=["Name", "Known Period", "Found Period","False Alarm Probability"])

print(compedPs)


trialName = "Ruff"

try:
    knownFreq = 1/(compedPs.at[compedPs.index[compedPs["Name"]==trialName].values[0],"Known Period"]*60*60)
except:
    knownFreq=None


singleNameLSP(interpLcDF,trialName, knownFreq)

trialCut = name_cut(interpLcDF, trialName, colName="Name")

knownP = compedPs.at[compedPs.index[compedPs["Name"]==trialName].values[0], "Known Period"]

foundP = compedPs.at[compedPs.index[compedPs["Name"]==trialName].values[0], "Found Period"]

theta = apPer.at[apPer.index[apPer["Name"]==trialName].values[0], "Model Parameters"]

times = np.linspace(trialCut["MJD"].min(), trialCut["MJD"].max(), 1000)

fig, ax = plt.subplots(figsize = (8,5))

ax.set(xlabel="Time [MJD]", ylabel="Flux")

model = model_from_params(theta=theta, times=times, period=foundP/24, avgFlux=np.median(trialCut["COM Flux"]))

ax.plot(times, model,  linestyle= "--", c="k", label ="Model and found P")


ax.plot(times, np.sin(times*(2*np.pi/(knownP/24)))+np.mean(trialCut["Flux"]), linestyle= "-.", label = "Known P")  

ax.scatter(trialCut["MJD"], trialCut["Flux"],c="tab:blue", marker = "o", s=10, label = "Light Curve")

ax.scatter(trialCut["MJD"], trialCut["COM Flux"],c="tab:orange", marker = "d", s=10, label = "COM Light Curve")


matchesDf = load_matches(sector,cam, ccd, cut)

matchCut = name_cut(matchesDf, trialName)

ax.scatter(matchCut["Time"],matchCut["flux"], label="Matched Flux",c="Pink", marker="^")


# ax.set_ylim(100,800)
ax.legend()



fig2, ax2 = plt.subplots(figsize = (8,6))



tessZP = 20.44 #* From Clarinda
tessZP = 20.6 #* From Clarinda


comMags = tessZP - 2.5*np.log10(trialCut["COM Flux"]) #????


detectMags = tessZP - 2.5*np.log10(matchCut["flux"]) 


ax2.scatter(trialCut["MJD"], comMags, label = "Mag from COM Flux")



ax2.scatter(matchCut["Time"], detectMags, label = "Mag from detections flux")

try:
    magsCSV = pd.read_csv(f"MPC_{trialName}_mags.csv")
    ax2.scatter(magsCSV["Date"], magsCSV["Mag"]-0.486, label= "MPC mags (G band)")
except:
    print(f"MPC mags not avalible for {trialName}")

ax2.set(xlabel="Time [MJD]", ylabel="Mag [T?]")
ax2.invert_yaxis()
ax2.legend()




#* to get just a few periods of the named asteroid. sensitive to what times you cut

# downCut =trialCut.loc[trialCut.index[(trialCut["MJD"]>=58900.58) & (trialCut["MJD"]<=58901.11)]]

# detect_period_ap(downCut["MJD"], downCut["Flux"], plotting=True, knownFreq=knownFreq)

# detect_period_ap(downCut["MJD"], downCut["COM Flux"], plotting=True, knownFreq=knownFreq)

