# -*- coding: utf-8 -*-


import numpy as np

class LogisticRegression:
    def __init__(self, train_data, noOfClasses, train_label = None):
        self.train_data = train_data
        self.train_label = train_label
        self.noOfClasses = noOfClasses
        self.classes = sorted(np.unique(self.train_label))
        self.classes = np.array(self.classes)
        self.noOfClasses = self.classes.shape[0]
        
        self.features = np.ones((self.train_data.shape[0],self.train_data.shape[1]+1))
        self.features[:,1:] = self.train_data

    
    def train(self, ETA, EPOCH):
        self.labelData = np.zeros((self.noOfClasses, self.train_data.shape[0] ), dtype = np.int)
        for i in range(0, self.train_data.shape[0]):
            self.labelData[self.train_label[i]][i] = 1
        
      
        self.W = np.zeros((785, 10))
        
        self.W = self.gradientDescent(ETA, EPOCH, self.W, self.classes)

   
        
        return self.W
                
   
    
   
    #to find the softmax    
    def softmax(self, W, X):
        vec = np.dot(W.T, X.T)
        ol = np.add(vec, -vec.max(axis=0))
        vec1 = np.exp(ol)
        res = (vec1)/(np.sum(vec1, axis=0))
        
        return res
        
    #finding the weights
    def gradientDescent(self, ETA, EPOCH, W, classes):
        
        for i in range(0, EPOCH):
            weights = self.softmax(W, self.features)
            error = (weights - self.labelData)
            GradientW = np.dot(self.features.T, error.T )
            W = W - ETA * GradientW
            
        return W
    
    #to test the data    
    def test(self, W):
        probability = self.softmax(W, self.features)
        predicted = np.argmax(probability, axis = 0)
       
        prob = np.amax(probability, axis = 0)
        return [predicted, prob]

            
        

