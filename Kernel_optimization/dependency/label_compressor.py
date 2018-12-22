import os
import pickle

import numpy as np


class LabelCompressor:
    def __init__(self, file):
        if not os.path.exists(file):
            self.label_set = {}
            self.counter = 0
        else:
            self.label_set = pickle.load(open(file, "rb"))
            self.counter = max(list(self.label_set.values()))
            print("label_counter", self.counter)

    def compress(self, label):
        if label in self.label_set:
            return self.label_set[label]
        else:
            self.counter += 1
            self.label_set[label] = self.counter
            return self.counter

    def dump(self, file):
        pickle.dump(self.label_set, open(file, "wb"))

