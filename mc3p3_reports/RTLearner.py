"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
from random import randint


class RTLearner(object):
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        if self.verbose:
            print "leaf_size is : " + str(leaf_size)

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
        combined = np.column_stack((dataX, dataY))
        if self.verbose:
            print "Combined Data Table:"
            print combined

        self.tree = self.build_tree(combined)

        if self.verbose:
            print "Constructed Tree in Tabular Format:"
            print self.tree

    def build_tree(self, data):
        #If there is only one row left
        if data.shape[0] == 1:
            return np.array([[-1, data[0, -1], np.NAN, np.NAN]])

        #If the size of the subtree is less than the leaf_size, you avg the y to create a leaf
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:,-1]), np.NAN, np.NAN]])

        #If all the Y data are the same
        if len(np.unique(data[:, -1])) == 1:
            return np.array([[-1, data[0, -1], np.NAN, np.NAN]])
        else:
            left, right, i, split_val = self.split(data)
            left_tree = self.build_tree(left)
            right_tree = self.build_tree(right)
            root = np.array([i, split_val, 1, left_tree.shape[0] + 1])
            return np.vstack([root, left_tree, right_tree])

    def split(self, data, retry=50):
        count = 0
        while count < retry:
            i = randint(0, data.shape[1] - 2)
            rand_1 = randint(0, data.shape[0] - 1)
            rand_2 = randint(0, data.shape[0] - 1)
            split_val = (data[rand_1, i] + data[rand_2, i]) / 2
            left = data[data[:, i] <= split_val]
            right = data[data[:, i] > split_val]
            if left.shape[0] > 0 and right.shape[0]:
                return left, right, i, split_val
            else:
                count +=1
        raise ValueError('Unable to Split the Data into non-empty Left and non-empty Right.. Max Retry has reached')

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
        for p in points:
            predict = self.predict(0, self.tree[0], p)
            yield predict

    def predict(self, i, row, point):
        if int(row[0]) == -1:
            return row[1]
        feat = int(row[0])
        split_val = row[1]
        j = i+int(row[2]) if point[feat] <= split_val else i+int(row[3])
        return self.predict(j, self.tree[j], point)


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
