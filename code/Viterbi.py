# This is a template for a Vitirbi class that can be used to compute
# most likely sequences.
import numpy as np

class Viterbi(object):
    # Construct an instance of the viterbi class
    # based upon an instantiatin of the HMM class
    def __init__(self, hmm):
        self.hmm = hmm

        self.numStates = hmm.numStates
        self.numOutputs = hmm.numOutputs
        self.states = hmm.states
        self.outputs = hmm.outputs

        self.prior = hmm.prior
        self.trans = hmm.trans
        self.emit = hmm.emit

        self.pre = {}

    def deltapre_mat(self, output, seqnum):
        
        ZT = output[seqnum]
        delta_m = np.zeros((201, 16))
        delta_m[0, :] = self.emit[:, ZT[0]] * self.prior

        for t in range(1, 201):
            for i in range(16):
                temp_set = []
                for j in range(16):
                    temp_set.append(self.trans[j, i] * delta_m[t-1, j])
                maxx = max(temp_set)
                max_j = temp_set.index(maxx)
                delta_m[t, i] = self.emit[i, ZT[t]] * maxx
                self.pre[(t, i, seqnum)] = max_j
        return delta_m


    # Returns the most likely state sequence for a given output
    # (observation) sequence, i.e.,
    #    arg max_{X_1, X_2, ..., X_T} P(X_1,...,X_T | Z_1,...Z_T)
    # according to the HMM model that was passed to the constructor.
    def mostLikelySequence(self, output):
        # result[i][j] indicates the jth state prediction of the ith observation sequence 
        result = []

        for seqnum in range(len(output)):
            print("test seqnum: " + str(seqnum))
            delta_m = self.deltapre_mat(output, seqnum)
            cur_result = []
            temp_set = []
            for i in range(16):
                temp_set.append(delta_m[200, i])
            maxx = max(temp_set)
            max_i = temp_set.index(maxx)
            cur_result.append(max_i)

            t = 199
            while t >= 0:
                cur_result.append(self.pre[t+1, cur_result[len(cur_result)-1] ,seqnum])
                t = t - 1
            cur_result.reverse()
            result.append(cur_result)
        return result
