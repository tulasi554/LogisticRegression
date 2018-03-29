# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 18:47:42 2016

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

    settings.init()
    
    dataset = ReadDataSet()
    dataset.Read(GetCurrentPath() + '/Data/mnist.pkl')
    [test_data, test_labels] = dataset.getTrainData()
    
    #DisplayDigit(train_data[5], train_labels[5])
    
    import cPickle
    with open(GetCurrentPath() + '/Models/weights.pkl', 'rb') as fp:
        Wt = cPickle.load(fp)
    
    print shape(Wt)[0]
    LR = LogisticRegression(test_data, shape(Wt)[0])
    [predict_labels, probs] = LR.test(Wt)
    
    TP = 0
    for i in range(0, len(test_labels)):
        #print('%d %d %d %f'% (i, test_labels[i],predict_labels[i],probs[i]))
        if (test_labels[i]==predict_labels[i]):
            TP += 1
               
    acc = 100*TP/float(len(predict_labels)) 
    print('Recognition Rate = %f'%(acc))    
    