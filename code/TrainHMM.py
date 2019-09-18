import csv
import numpy as np
import sys
from HMM import *
from DataSet import *


filename = sys.argv[1]
dSet = DataSet(filename)
dSet.readFile(200, "train")
hmm = HMM(16, 4, dSet.trainState, dSet.trainOutput)
hmm.train()

