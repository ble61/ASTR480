from asteroidCatalouge import make_asteroid_cat
import numpy as np

thisSector = 22

#*For Full runs
# possibleCams = [1,2,3,4]
# possibleCCDs = [1,2,3,4]
# possibleCuts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#For testing run
possibleCams = [1]
possibleCCDs = [3]
possibleCuts = [7]

for i in  possibleCams:
    for j in possibleCCDs:
        for k in possibleCuts:
            make_asteroid_cat(thisSector,i,j,k)
