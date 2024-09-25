from makeAsteroidLightcurve import makeLCs
import numpy as np
from itertools import product
from multiprocessing import Pool

thisSector = 22

#*For Full runs
# possibleCams = [1,2,3,4]
# possibleCCDs = [1,2,3,4]
# possibleCuts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#For testing run
possibleCams = [1]
possibleCCDs = [3]
possibleCuts = [7]        


if __name__ == '__main__':
    with Pool(12) as p:
        p.starmap(makeLCs, product([thisSector],possibleCams,possibleCCDs,possibleCuts))
        # p.starmap(thisPrint, product([thisSector],possibleCams,possibleCCDs,possibleCuts))

