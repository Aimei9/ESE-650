import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        
        cmap = np.rot90(cmap,-1)
        belief = np.rot90(belief,-1)
        A = np.array(cmap).shape[0]
        B = np.array(cmap).shape[1]
        go = np.zeros((A,B))
     
        for i in range(A):
            for j in range(B):
                # Not out of bounds
                if i+action[0]>=0 and i+action[0]<A and j+action[1]>=0 and j+action[1]<B: 
                    go[i+action[0]][j+action[1]] += belief[i][j]*0.9
                    go[i][j] += belief[i][j]*0.1
                else:
                    go[i][j] += belief[i][j]*1   # Condition remains unchanged
        # Sensory data
        res = 0
        for i in range(A):
            for j in range(B):
                if cmap[i][j] == observation:  # The right data
                    go[i][j] = 0.9*go[i][j]
                    res += go[i][j]
                else:                          # Incorrect figures
                    go[i][j] = 0.1*go[i][j]
                    res += go[i][j]
        go /= res
        temp = 0
        for i in range(A):
            for j in range(B):
                if go[i][j] > temp:
                    temp = go[i][j]
                    state = [i,j]
        return np.rot90(go, k=1), np.array(state)
