import sys
from DataSet import *
from HMM import *
from Viterbi import *


# This defines a simple class for running your Viterbi code. As with
# all the code, feel free to modify it as you see fit, or to write
# your own outright.


class RunViterbi(object):

    def __init__(self):
        self.maxSequence = []


    def trainHMM(self, filename):
        print ("Reading training data from %s" % (filename))

        # Read in the training data from the file
        self.dataset = DataSet(filename)
        self.dataset.readFile(200, "train")

        # Instatiate and train the HMM
        self.hmm = HMM(self.dataset.numStates, self.dataset.numOutputs, self.dataset.trainState, self.dataset.trainOutput)
        self.hmm.train()
        
        return

    def estMaxSequence(self, filename):

        print ("Reading testing data from %s" % (filename))

        # Read in the testing dta from the file
        self.dataset = DataSet(filename)
        self.dataset.readFile(200, "test")

        # Run Viterbi to estimate most likely sequence
        viterbi = Viterbi(self.hmm)
        self.maxSequence = viterbi.mostLikelySequence(self.dataset.testOutput)


if __name__ == '__main__':

    # This function should be called with two arguments: trainingdata.txt testingdata.txt
    viterbi = RunViterbi()
    viterbi.trainHMM(sys.argv[1])
    viterbi.estMaxSequence(sys.argv[2])
    
    # calculate the misses and accuracy of the implemented viterbi algorithm
    total_miss = 0
    for i in range(len(viterbi.maxSequence)):
        miss = 0
        for t in range(201):
            if viterbi.maxSequence[i][t] != viterbi.dataset.testState[i][t]:
                miss += 1
                total_miss += 1
        print("test seq " + str(i) + " misses " + str(miss))

    print("overall target percentage:")
    print(str(100-(total_miss/400)) + "%")
    




