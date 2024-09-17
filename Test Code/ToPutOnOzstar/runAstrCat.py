from asteroidCatalouge import make_asteroid_cat
import numpy as np
from itertools import product


thisSector = 22

#*For Full runs
# possibleCams = [1,2,3,4]
# possibleCCDs = [1,2,3,4]
# possibleCuts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#For testing run
possibleCams = [1]
possibleCCDs = [3]
possibleCuts = [7]

# for i in  possibleCams:
#     for j in possibleCCDs:
#         for k in possibleCuts:
#             print(thisSector,i,j,k)
#             


from multiprocessing import Pool
# import time

# def thisPrint(sector,cam,ccd,cut):
#     time.sleep(0.5)
#     print(sector,cam,ccd,cut)

if __name__ == '__main__':
    with Pool(12) as p:
        p.starmap(make_asteroid_cat, product([thisSector],possibleCams,possibleCCDs,possibleCuts))
        # p.starmap(thisPrint, product([thisSector],possibleCams,possibleCCDs,possibleCuts))



