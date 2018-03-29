# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:16:59 2016

@author: ymlui
"""

import sys, os
import numpy as np
from pylab import *

def GetCurrentPath():
    return os.path.dirname(os.path.realpath(__file__))

def SetupPath():
    newpath = GetCurrentPath() + '/Utils'
    sys.path.append(newpath)

    newpath = GetCurrentPath() + '/Algo'
    sys.path.append(newpath)

    
    
def DisplayDigit(sample, label):
    
    sample = sample.reshape(settings.WIDTH, settings.HEIGHT)
    [fig, axs] = plt.subplots(1,1)
    axs.imshow(sample, cmap='Greys_r')
    print label
    
            
############### main function #################    
if __name__ == '__main__':  
    SetupPath()
    from settings import *
    from ReadDataSet import *
    from LogisticRegression import *
    import settings

    settings.init()
    
    dataset = ReadDataSet()
    print GetCurrentPath()
    dataset.Read(GetCurrentPath() + '/Data/mnist.pkl')
    [train_data, train_labels] = dataset.getTrainData()
    
    #DisplayDigit(train_data[8], train_labels[8])
    
    LR = LogisticRegression(train_data, -1, train_labels)
    Wt = LR.train(settings.ETA, settings.EPOCH)
    
    import cPickle
    with open(GetCurrentPath() + '/Models/weights.pkl', 'wb') as fp:
        cPickle.dump(Wt, fp)
    