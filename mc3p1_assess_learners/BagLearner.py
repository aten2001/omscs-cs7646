"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
from random import randint


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        learners = []
        for i in range(0, bags):
            learners.append(learner(**kwargs))

        self.learners = learners

        if self.verbose:
            print "Learner Type is :" + str(type(learner))
            print "KWARGS : " + str(kwargs)
            print "Bags : " + str(bags)
            print "Boosting : " + str(boost)

        pass  # move along, these aren't the drones you're looking for

    def author(self):
        return 'jlee3259'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # slap on 1s column so linear regression finds a constant term
        n = dataX.shape[0]
        n_prime = n * 0.6
        if self.verbose:
            print "n : " + str(n)
            print "n prime: " + str(n_prime)

        count = 1
        for learner in self.learners:
            if self.verbose:
                print "Processing {}th Learner...".format(count)
            sample_x = []
            sample_y = []
            for i in range(0, int(n_prime)):
                random_i = randint(0, n-1)
                sample_x.append(dataX[random_i])
                sample_y.append(dataY[random_i])
            learner.addEvidence(sample_x, sample_y)
            count +=1

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result = np.fromiter(self.predict_loop(points), dtype=float)
        if self.verbose:
            print result
        return result

    def predict_loop(self, points):
        predictions = []
        count = 1
        for learner in self.learners:
            if self.verbose:
                print "Querying from {}th Learner..".format(count)
            predicted = learner.query(points)
            predictions.append(predicted)
            count +=1
        return np.mean(predictions, axis=0)



if __name__ == "__main__":
    # import RTLearner as rt
    # learner = rt.RTLearner(leaf_size=1, verbose=False)  # constructor
    # learner.addEvidence(Xtrain, Ytrain)  # training step
    # Y = learner.query(Xtest)  # query

    """
    Where "leaf_size" is the maximum number of samples to be aggregated at a leaf.
    While the tree is being constructed recursively, if there are leaf_size or fewer elements at the time of the recursive call,
    the data should be aggregated into a leaf.
    """
    print "the secret clue is 'zzyzx'"