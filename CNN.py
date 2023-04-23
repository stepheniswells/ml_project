import numpy as np
class CNN:
    weights = []
    numNeurons = 0
    output = -1

    #input is array of images
    def __init__(self, input, output, numNeurons):
        self.numNeurons = numNeurons
        self.output = output
        return
    
    def train(self, input, output):
        return
    
    #input is single image, result is single feature map based on passed in filter.
    def convolution(self, input, filter):
        r = filter.shape[0]-1
        d = r
        featureMap = [0] * ((input.shape[0]-filter.shape[0]+1)*(input.shape[0]-filter.shape[0]+1))
        index = 0
        while d < input.shape[0]:
            r = 1
            while r < input.shape[1]:
                submatrix = input[d-1:d+1,r-1:r+1]
                print(submatrix[:])
                res = np.multiply(submatrix, filter)
                print(np.sum(res))
                featureMap[index] = np.sum(res)
                r+=1
                index+=1
            d+=1
        featureMap = np.reshape(featureMap, (-1, input.shape[0]-filter.shape[0]+1))
        print(featureMap)
        return featureMap

    #1D inputs passed into layer, performs sigmoid activation
    def denseLayer(self, input):
        res = np.dot(self.weights.T, input)
        return self.sigmoidActivation(res)

    #Something takes a single image, gets the prediction, gets error, backpropagates. 
    def something(self, numFilters, input, filter):
        # Pass image into convolutional layer to get feature maps
        featureMaps = self.reluActivation(self.convolution(input, filter))
        for i in range(numFilters-1):
            featureMap = self.convolution(input, filter)
            featureMap = self.reluActivation(featureMap)
            featureMaps = np.concatenate((featureMaps, featureMap))
        print(featureMaps)
        print(featureMaps.shape)

        # Flatten for dense layer and set weight matrix size
        featureMaps = featureMaps.flatten()
        self.weights = np.random.uniform(size=(featureMaps.shape[0], self.numNeurons))

        # Pass features to dense layer
        res = self.denseLayer(featureMaps)
        for el in res:
            print(el)

        # Get output/error
        error = ((res - self.output)**2)/2
        dEdO = -(self.output-res)
        dOdZ = res * (1-res)
        dZdW = featureMaps
        


    def denseLayerBackPropagation():

        return

    def sigmoidActivation(self, val):
        return 1/(1+np.exp(-val))
    
    def reluActivation(self, val):
        return np.maximum(val, 0)
