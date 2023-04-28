import numpy as np
import pandas as pd
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sn

modelPath = './trainedmodel.h5'
d = 64 #dimension of img
class ConvolutionalNeuralNet:
    model = Sequential()

    testArray = []
    testDf = pd.DataFrame()

    def __init__(self):
        # Check if trained model already exists
        if os.path.isfile(modelPath):
            print("Found previously trained model file - using that.")
            self.model = load_model(modelPath)
            return
        print("Did not find previously trained model file. Training model now..")
        
        # Get cat/dog training data from folder
        df = self.getTrainingData()
        
        #Preprocessing and train/test split
        X_train, X_test, y_train, y_test = train_test_split(df.path,df.label)
        X_train = [tf.keras.utils.load_img(path,target_size=(d,d)) for path in X_train]
        X_train = np.array([tf.keras.utils.img_to_array(img) for img in X_train])
        X_test = [tf.keras.utils.load_img(path,target_size=(d,d)) for path in X_test]
        X_test = np.array([tf.keras.utils.img_to_array(img) for img in X_test])

        # Divide by 255 to convert color to 0->1
        X_train /= 255
        X_test /= 255

        # Build the model layer by layer
        self.constructModel()

        # Train the model
        history = self.model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

        # Get some evaluation metrics
        self.confusionMatrix(self.model.predict(X_test), y_test, "Original CNN Confusion Matrix")

        # plot accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('CNN Accuracy vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel("Accuracy")
        plt.legend(['Train Accuracy', 'Validation Accuracy'])
        plt.show()

        self.model.evaluate(X_test, y_test)
        self.model.evaluate(X_train, y_train)

        # Save trained model
        self.model.save(modelPath)
        
    def populateTestData(self):
        # Populate test metrics dataframe
        cats = r"./\dataset\test\cat"
        catsPaths = [join(cats,filename) for filename in os.listdir(cats)]
        dogs = r"./\dataset\test\dog"
        dogsPaths = [join(dogs,filename) for filename in os.listdir(dogs)]

        # 1 = cat, 0 = dog
        df = pd.DataFrame({
            'path': catsPaths + dogsPaths,
            'label': [1 if path in catsPaths else 0 for path in catsPaths + dogsPaths] })
        imgs = [tf.keras.utils.load_img(path,target_size=(d,d)) for path in df.path]
        self.testArray = np.array([tf.keras.utils.img_to_array(img) for img in imgs])
        self.testArray /= 255
        self.testDf = df

    def getTrainingData(self):
        cats = r"./\dataset\training_set\cats"
        catsPaths = [join(cats,filename) for filename in os.listdir(cats)]
        dogs = r"./\dataset\training_set\dogs"
        dogsPaths = [join(dogs,filename) for filename in os.listdir(dogs)]

        # 1 = cat, 0 = dog
        df = pd.DataFrame({
            'path': catsPaths + dogsPaths,
            'label': [1 if path in catsPaths else 0 for path in catsPaths + dogsPaths] })
        return df
    
    def constructModel(self):
        self.model.add(Conv2D(d, (3, 3), activation='relu', input_shape=(d, d, 3)))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(d, (3, 3), activation='relu', input_shape=(d, d, 3)))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(d, (3, 3), activation='relu', input_shape=(d, d, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(d, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
        
    def predict(self, input):
        predictions = self.model.predict(input)
        return predictions
    
    def train(self, input, output):
        print("Training model with new data")
        history = self.model.fit(input, output, epochs=8)

    def getTestMetrics(self):
        print("Getting model metrics on test dataset:")
        self.model.evaluate(self.testArray, self.testDf.label)
        predictions = self.model.predict(self.testArray)
        self.confusionMatrix(predictions, self.testDf.label, "Confusion Matrix")
        
        # plot predictions
        plt.scatter(range(len(predictions)), predictions, c=self.testDf.label)
        plt.title("Predicted Values for Test Data")
        plt.xlabel("Datapoints")
        plt.ylabel("Predictions")
        plt.legend(['Cats', 'Dogs'])
        plt.show()

        #plot ROC
        falsePos, truePos, thresholds = roc_curve(self.testDf.label, predictions)
        auc = roc_auc_score(self.testDf.label, predictions)
        plt.plot(falsePos, truePos, label="AUC " + str(auc))
        plt.title("ROC Curve")
        plt.ylabel("True Positive")
        plt.xlabel("False Positive")
        plt.legend(loc=4)
        plt.show()

    def confusionMatrix(self, predictions, actual, title):
        predictions = np.where(predictions < 0.5, 0, 1)
        confMat = confusion_matrix(actual, predictions, normalize="pred")
        cm = pd.DataFrame(confMat)
        sn.heatmap(cm)
        plt.title(title)
        plt.show()

    # Perform random sampling s times
    def randomSampling(self, s):
        cats = r"./\dataset\test\cat1"
        catsPaths = [join(cats,filename) for filename in os.listdir(cats)]
        dogs = r"./\dataset\test\dog1"
        dogsPaths = [join(dogs,filename) for filename in os.listdir(dogs)]
        imagePaths = catsPaths + dogsPaths

        # 1 = cat, 0 = dog
        df = pd.DataFrame({
            'path': imagePaths,
            'label': [1 if path in catsPaths else 0 for path in imagePaths] })
        randomDfSample = df.sample(s)
        imgs = [tf.keras.utils.load_img(path,target_size=(d,d)) for path in randomDfSample.path]
        X_test = np.array([tf.keras.utils.img_to_array(img) for img in imgs])
        X_test /= 255
        print("Training model with random sample")
        cnn.train(X_test, randomDfSample.label)
        print("Model metrics after random sampling")
        cnn.getTestMetrics()
    
    def automaticUncertainySampling(self, s):
        cats = r"./\dataset\test\cat1"
        catsPaths = [join(cats,filename) for filename in os.listdir(cats)]
        dogs = r"./\dataset\test\dog1"
        dogsPaths = [join(dogs,filename) for filename in os.listdir(dogs)]

        # 1 = cat, 0 = dog
        df = pd.DataFrame({
            'path': catsPaths + dogsPaths,
            'label': [1 if path in catsPaths else 0 for path in catsPaths + dogsPaths] })
        randomDfSample = df.sample(s)
        imgs = [tf.keras.utils.load_img(path,target_size=(d,d)) for path in randomDfSample.path]
        arr = np.array([tf.keras.utils.img_to_array(img) for img in imgs])
        arr /= 255
        predictions = cnn.predict(arr)
        threshold = 0.3 # threshold for determining if cnn is unsure about datapoint
        count = 0
        for i in range(len(arr)):
            if abs(predictions[i][0] - 0.5) < threshold: 
                count+=1
        print("Count " + str(count))
        unsure = np.empty((count, d, d, 3))
        labels = np.empty(count)
        ind = 0
        for i in range(len(arr)):
            if abs(predictions[i][0] - 0.5) < threshold:
                unsure[ind] = arr[i]
                labels[ind] = randomDfSample.iloc[i].label
                ind+=1 
        print("Training model with uncertainty sampling")
        cnn.train(unsure, labels)
        print("Model metrics after uncertainty sampling")
        cnn.getTestMetrics()

    def uncertaintySampling(self):
        #Give the CNN unlabeled data and get its predictions
        data = r"./\dataset\mixed_set"
        paths = [join(data,filename) for filename in os.listdir(data)]
        df = pd.DataFrame({'path': paths})
        images = [tf.keras.utils.load_img(path,target_size=(d,d)) for path in df.path]
        arr = np.array([tf.keras.utils.img_to_array(img) for img in images])
        normalizedArr = arr/255
        predictions = cnn.predict(normalizedArr)

        # Perform uncertaintySampling based on previous data given to CNN
        threshold = 0.1 # threshold for determining if cnn is unsure about datapoint
        count = 0
        for i in range(len(arr)):
            if abs(predictions[i][0] - 0.5) < threshold: 
                count+=1
        unsure = np.empty((count, d, d, 3))
        ind = 0
        for i in range(len(arr)):
            if abs(predictions[i][0] - 0.5) < threshold:
                unsure[ind] = arr[i]
                ind+=1        
        if(count == 0):
            print("There were no uncertain samples")
        else:
            s = input("There were " + str(count) + " samples the CNN couldn't confidently predict. How many would you like to label?\n")
            if(int(s) != 0):
                np.random.shuffle(unsure)
                s = int(s)
                if(s > len(unsure)):
                    s = len(unsure)
                uncertainSamples = np.empty((s, d, d, 3))
                givenLabels = np.empty(s)
                for i in range(s):
                    plt.imshow(unsure[i].astype('uint8'))
                    plt.draw()
                    plt.pause(0.1)
                    while True:
                        userIn = input('Enter c for cat, d for dog.\n')
                        if userIn == "c":
                            givenLabels[i] = 1
                            uncertainSamples[i] = unsure[i]
                            break
                        elif userIn=="d":
                            givenLabels[i] = 0
                            uncertainSamples[i] = unsure[i]
                            break
                        else:
                            print("Invalid character, enter c or d")
                    plt.close()
                cnn.train(uncertainSamples/255, givenLabels)
                print("Model test metrics after training on uncertain samples")
                cnn.getTestMetrics() 

if __name__ == "__main__":
    #Create/get the CNN which is trained on cat/dog data
    cnn = ConvolutionalNeuralNet()

    # Get CNN metrics prior to human assist
    cnn.populateTestData()
    print("Model metrics against test data prior to Active learning")
    cnn.getTestMetrics()

    choice = input("What type of active learning do you want to do? Press 1 for random sampling, 2 for automatic uncertainty sampling, or 3 to manually label images")
    if(int(choice) == 1):
        cnn.randomSampling(900)
    elif(int(choice) == 2):
        cnn.automaticUncertainySampling(900)
    elif(int(choice) == 3):
        cnn.uncertaintySampling()
    else:
        print("invalid selection")
    

        