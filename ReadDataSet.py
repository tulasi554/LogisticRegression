# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:25:01 2016

@author: ymlui
"""

import numpy as np
from pylab import *


class ReadDataSet:
    
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        
    def getTrainData(self):
        #return self.train_data[0:50000], self.train_labels[0:50000]
        return self.train_data[0:1000], self.train_labels[0:1000]
        
    def getTestData(self):
        #return self.test_data, self.test_labels
        return self.test_data[0:100], self.test_labels[0:100]
        
    def Read(self, datapath):
    
        import cPickle
        fp = open(datapath, 'rb')
        #fp = open('C:\Tulasi\machine learning\workspace\DigitRecognition\DigitRecognition\Data\mnist.pkl')
        [train, test, validate] = cPickle.load(fp)
        fp.close()
    
        self.train_data = train[0]
        self.train_labels = train[1]
        self.test_data = test[0]
        self.test_labels = test[1]
    
