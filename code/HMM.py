import numpy as np
import matplotlib.pyplot as plt

class HMM(object):
    # Construct an HMM with the following set of variables
    #    numStates:          The size of the state space
    #    numOutputs:         The size of the output space
    #    trainStates[i][j]:  The jth element of the ith state sequence
    #    trainOutputs[i][j]: Similarly, for output
    def __init__(self, numStates, numOutputs, states, outputs):

        self.numStates = numStates
        self.numOutputs = numOutputs
        self.states = states
        self.outputs = outputs

        # Initialize all parameters as stated
        self.prior = np.array([0, 1/12, 1/12, 0, 1/12, 0, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 0, 1/12])
        self.trans = np.zeros((16, 16))

        self.trans[1, 1] = 0.2
        self.trans[1, 2] = 0.8
        self.trans[2, 2] = 0.2
        self.trans[2, 1] = 0.4
        self.trans[2, 6] = 0.4
        self.trans[4, 4] = 0.2
        self.trans[4, 8] = 0.8
        self.trans[6, 6] = 0.2
        self.trans[6, 2] = 0.8/3.0
        self.trans[6, 7] = 0.8/3.0
        self.trans[6, 10] = 0.8/3.0
        self.trans[7, 7] = 0.2
        self.trans[7, 6] = 0.4
        self.trans[7, 11] = 0.4
        self.trans[8, 8] = 0.2
        self.trans[8, 4] = 0.8/3.0
        self.trans[8, 9] = 0.8/3.0
        self.trans[8, 12] = 0.8/3.0
        self.trans[9, 9] = 0.2
        self.trans[9, 8] = 0.8/3.0
        self.trans[9, 10] = 0.8/3.0
        self.trans[9, 13] = 0.8/3.0
        self.trans[10, 10] = 0.2
        self.trans[10, 6] = 0.8/3.0
        self.trans[10, 9] = 0.8/3.0
        self.trans[10, 11] = 0.8/3.0
        self.trans[11, 11] = 0.2
        self.trans[11, 7] = 0.8/3.0
        self.trans[11, 10] = 0.8/3.0
        self.trans[11, 15] = 0.8/3.0
        self.trans[12, 12] = 0.2
        self.trans[12, 8] = 0.4
        self.trans[12, 13] = 0.4
        self.trans[13, 13] = 0.2
        self.trans[13, 9] = 0.4
        self.trans[13, 12] = 0.4
        self.trans[15, 15] = 0.2
        self.trans[15, 11] = 0.8

        self.trans[0, 0] = 1.0
        self.trans[3, 3] = 1.0
        self.trans[5, 5] = 1.0
        self.trans[14, 14] = 1.0


        self.emit = np.zeros((16, 4))

        self.emit[1, 0] = 0.7
        self.emit[1, 1] = 0.1
        self.emit[1, 2] = 0.1
        self.emit[1, 3] = 0.1
        self.emit[2, 0] = 0.1
        self.emit[2, 1] = 0.7
        self.emit[2, 2] = 0.1
        self.emit[2, 3] = 0.1
        self.emit[4, 0] = 0.1
        self.emit[4, 1] = 0.7
        self.emit[4, 2] = 0.1
        self.emit[4, 3] = 0.1
        self.emit[6, 0] = 0.1
        self.emit[6, 1] = 0.1
        self.emit[6, 2] = 0.7
        self.emit[6, 3] = 0.1
        self.emit[7, 0] = 0.7
        self.emit[7, 1] = 0.1
        self.emit[7, 2] = 0.1
        self.emit[7, 3] = 0.1
        self.emit[8, 0] = 0.1
        self.emit[8, 1] = 0.1
        self.emit[8, 2] = 0.1
        self.emit[8, 3] = 0.7
        self.emit[9, 0] = 0.1
        self.emit[9, 1] = 0.7
        self.emit[9, 2] = 0.1
        self.emit[9, 3] = 0.1
        self.emit[10, 0] = 0.7
        self.emit[10, 1] = 0.1
        self.emit[10, 2] = 0.1
        self.emit[10, 3] = 0.1
        self.emit[11, 0] = 0.1
        self.emit[11, 1] = 0.1
        self.emit[11, 2] = 0.1
        self.emit[11, 3] = 0.7
        self.emit[12, 0] = 0.1
        self.emit[12, 1] = 0.1
        self.emit[12, 2] = 0.7
        self.emit[12, 3] = 0.1
        self.emit[13, 0] = 0.1
        self.emit[13, 1] = 0.1
        self.emit[13, 2] = 0.1
        self.emit[13, 3] = 0.7
        self.emit[15, 0] = 0.1
        self.emit[15, 1] = 0.1
        self.emit[15, 2] = 0.7
        self.emit[15, 3] = 0.1

        self.emit[0, 0] = 0.25
        self.emit[0, 1] = 0.25
        self.emit[0, 2] = 0.25
        self.emit[0, 3] = 0.25
        self.emit[3, 0] = 0.25
        self.emit[3, 1] = 0.25
        self.emit[3, 2] = 0.25
        self.emit[3, 3] = 0.25
        self.emit[5, 0] = 0.25
        self.emit[5, 1] = 0.25
        self.emit[5, 2] = 0.25
        self.emit[5, 3] = 0.25
        self.emit[14, 0] = 0.25
        self.emit[14, 1] = 0.25
        self.emit[14, 2] = 0.25
        self.emit[14, 3] = 0.25

    # compute the forward matrix for a given sequence
    def alpha_mat(self, seqnum):
        
        forward_mat = np.zeros((201, 16))
        normal_factor = np.zeros(201)
        ZT = self.outputs[seqnum]

        forward_mat[0, :] = self.prior * self.emit[:, ZT[0]]
        # normalize
        normal_factor[0] = sum(forward_mat[0, :])
        forward_mat[0, :] = forward_mat[0, :]/sum(forward_mat[0, :])

        for t in range(1, 201):
            for xi in range(16):
                summ = 0
                for xt_minus_1 in range(16):
                    summ += self.trans[xt_minus_1, xi] * forward_mat[t-1, xt_minus_1]
                forward_mat[t, xi] = self.emit[xi, ZT[t]] * summ
            # normalize
            normal_factor[t] = sum(forward_mat[t, :])
            forward_mat[t, :] = forward_mat[t, :]/sum(forward_mat[t, :])

        return (forward_mat, normal_factor)

    # compute the backward matrix for a given sequence and the normalization factors from forward matrix
    def beta_mat(self, seqnum, normal_factor):

        backward_mat = np.zeros((201, 16))
        ZT = self.outputs[seqnum]

        backward_mat[200, :] = np.array([1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200], 1/normal_factor[200]])

        t = 199
        while t >= 0:
            for xi in range(16):
                summ = 0
                for xt_plus_1 in range(16):
                    summ += self.emit[xt_plus_1, ZT[t+1]] * self.trans[xi, xt_plus_1] * backward_mat[t+1, xt_plus_1]
                backward_mat[t, xi] = summ
                # normalize with normal_factor of forward matrix
                backward_mat[t, xi] = backward_mat[t, xi]/normal_factor[t]
            t = t - 1

        return backward_mat

    # compute the gemma matrix from forward/backward matrices for a given sequence
    def gemma_mat(self, forward_mat, backward_mat, seqnum):

        gemma = forward_mat * backward_mat
        # normalize
        if gemma.sum(axis=1)[:, np.newaxis].all() != 0:
            gemma /= gemma.sum(axis=1)[:, np.newaxis]
        return gemma

    # compute the transition probability matrix for all time for a given sequence
    def eps_mat(self, forward_mat, backward_mat, seqnum):

        ZT = self.outputs[seqnum]
        temp = np.transpose(self.trans)
        eps_mat = np.zeros((200, 16, 16))
        for t in range(200):
            tempp = np.transpose(temp * forward_mat[t])
            eps_mat[t, :, :] = tempp * self.emit[:, ZT[t+1]] * backward_mat[t+1, :]
            # normalize
            if np.sum(eps_mat[t, :, :]) != 0:
                eps_mat[t, :, :] = eps_mat[t, :, :] / np.sum(eps_mat[t, :, :])
        return eps_mat

    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self):

        # approximate observation log probability for all iterations
        obs_log = np.zeros(50)
        # iteration as x-axis
        x = np.zeros(50)
        # good convergence at 50 iterations
        for iterr in range(50):
            print("Epoch: " + str(iterr))
            x[iterr] = iterr
            # initialize of new iteration
            new_prior = np.zeros(16)
            trans_sum = np.zeros((16, 16))
            transdenom_sum = np.zeros((16, 16))
            emit_sum = np.zeros((16, 4))
            emitdenom_sum = np.zeros((16, 4))
            obs_sum = 0
            # go through training data
            for seqnum in range(200):
                if seqnum%10 == 0:
                    print(seqnum)

                ZT = self.outputs[seqnum]
                (forward_mat, normal_factor) = self.alpha_mat(seqnum)
                # observation probability
                obs_sum += np.prod(normal_factor)
                backward_mat = self.beta_mat(seqnum, normal_factor)
                gemma_mat = self.gemma_mat(forward_mat, backward_mat, seqnum)
                eps_mat = self.eps_mat(forward_mat, backward_mat, seqnum)
                # take in gemma at time = 0 for every gemma matrices
                new_prior = new_prior + gemma_mat[0, :]

                # calculate values that help with updates
                for i in range(16):
                    denom = np.sum(gemma_mat[0:200, i])
                    for j in range(16):
                        trans_sum[i, j] = trans_sum[i, j] + np.sum(eps_mat[:, i, j])
                        transdenom_sum[i, j] = transdenom_sum[i, j] + denom

                for i in range(16):
                    denom = np.sum(gemma_mat[:, i])
                    for m in range(4):
                        summ = 0
                        for t in range(201):
                            if ZT[t] == m:
                                summ += gemma_mat[t, i]
                        emit_sum[i, m] = emit_sum[i, m] + summ
                        emitdenom_sum[i, m] = emitdenom_sum[i, m] + denom

            print("Current observation log probability:")
            print(np.log(obs_sum))
            obs_log[iterr] = np.log(obs_sum)
            # update transition matrix
            for i in range(16):
                for j in range(16):
                    if transdenom_sum[i, j] != 0:
                        self.trans[i, j] = trans_sum[i, j]/transdenom_sum[i, j]
            # update emittion matrix
            for i in range(16):
                for m in range(4):
                    if emitdenom_sum[i, m] != 0:
                        self.emit[i, m] = emit_sum[i, m]/emitdenom_sum[i, m]
            # update prior
            self.prior = new_prior/np.sum(new_prior)

        plt.plot(x, obs_log)
        plt.xlabel('Iteration')
        plt.ylabel('Log Probability')
        plt.show()




    # # Returns the log probability associated with a transition from
    # # the dummy start state to the given state according to this HMM
    # def getLogStartProb (state):

    #     # Your code goes here
    #     print ("Please add code")

    # # Returns the log probability associated with a transition from
    # # fromState to toState according to this HMM
    # def getLogTransProb (fromState, toState):

    #     # Your code goes here
    #     print ("Please add code")

    # # Returns the log probability of state state emitting output
    # # output
    # def getLogOutputProb (state, output):

    #     # Your code goes here
    #     print ("Please add code")
    
        
