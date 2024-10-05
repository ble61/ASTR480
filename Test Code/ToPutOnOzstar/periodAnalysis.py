import numpy as np
import pandas as pd 
import astropy.units as u
from astropy.timeseries import LombScargle as lsp
import matplotlib as plt
import nifty_ls

def _name_cut(df, name:str, colName:str="Name"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn

def _load_lcs(sector, dirPath = "../Data/"):
    lcDF = pd.read_csv(f"{dirPath}{sector}_All_LCs.csv")
    lcDF.drop(columns="Unnamed: 0", inplace=True)
    return lcDF

def _load_properties(sector, dirPath = "../Data/"):
    propDF = pd.read_csv(f"{dirPath}{sector}_All_Properties.csv")
    propDF.drop(columns="Unnamed: 0", inplace=True)
    return propDF


def detect_period_ap(times, fluxes, sector, plotting = False, knownFreq = None):
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
    


    centredFlux = fluxes - np.mean(fluxes)

    minFreq = 1/(2*(times.max()-times.min())*u.day).to(u.s)
    
    # minFreq = 1e-5/u.s

    # minFreq = 1/(1468800*u.s)

    maxFreq = 1/nyquistP 

    lsper = lsp((u.Quantity(times, u.day)).to(u.s), centredFlux)
    freqs, powers = lsper.autopower(minimum_frequency=minFreq, maximum_frequency=maxFreq, method="fastnifty")
    bestPow = np.max(powers)
    bestFreq = freqs[np.argmax(powers)]

    t_fit=(np.linspace(times.min(),times.max(),1000)*u.day).to(u.s)
    y_fit = lsper.model(t_fit, bestFreq) + np.mean(fluxes)
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


def compute_periods_ap(lcDF, minNumObser=3):
    unqNames = pd.unique(lcDF["Name"])
    periodList_ap = []
    badCount = 0

    for name in unqNames:
        nameDf = _name_cut(lcDF, name, colName="Name")
        
        if nameDf.shape[0]<=minNumObser:# or nameProperties["Over Background Limit"][0] ==False:
            badCount+=1
            # print(f"{badCount}. not enough points (<={numObser}) for {name}")
            continue
        time = nameDf["MJD"].values
        flux = nameDf["COM Flux"].values

        period_ap, f_a_Prob, theta, maxPow = detect_period_ap(time, flux)
        periodList_ap.append([name, period_ap, period_ap/(60*60), period_ap/(60*60*24), f_a_Prob, theta, maxPow])

    apPeriods = pd.DataFrame(periodList_ap, columns=["Name", "Best Period [Seconds]", "Best Period [Hours]","Best Period [Days]",  "False Alarm Probability", "Model Parameters", "Max Periodogram Power"])

    return apPeriods, badCount



def period_analysis(sector):

    lcDF = _load_lcs(sector)

    periodData, badCounts = compute_periods_ap(lcDF)

    print(f"The bad count is {badCounts}")

    propDF = _load_properties(sector)

    propDF.merge(periodData, how="outer", on="Name")

    propDF.to_csv(f"../Data/{sector}Properties_and_Periods.csv")



period_analysis(29)