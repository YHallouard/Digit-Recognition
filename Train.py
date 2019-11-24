import os
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv1D, MaxPooling1D, Flatten, UpSampling1D, \
    Reshape, Lambda, Add,\
    Dropout, LeakyReLU, Activation, AveragePooling1D, PReLU, Softmax, Multiply
from tensorflow.keras.models import Model
import Digitclassifieur as DCL

os.environ['KMP_DUPLICATE_LIB_OK']='True'

os.chdir('/Users/yannhallouard/PycharmProjects/DigitClassification')

X = pd.read_csv("./digit-recognizer/train.csv")
test = pd.read_csv("./digit-recognizer/test.csv")
#test = test.values.reshape(test.shape[0], int(np.sqrt(test.shape[1])), int(np.sqrt(test.shape[1])), 1)
test = test.values.reshape(test.shape[0], 28, 28, 1).astype('float32')

target = X.iloc[:,0]

X = X.iloc[:,1:].values.reshape(X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])), 1)

new_target = DCL.build_target(target)



# Visualization
#plt.imshow(X[0].reshape(28, 28))

# -----------------
# Build Classifieur
# -----------------
CL = DCL.Classifieur(X=X.astype('float32'), target=new_target)
CL.classifieur.summary()
history = CL.train(epochs=20, batch_size=128, show_history=True)

CL.plot((CL.X[1:8]/255).astype('float32'), name="train")
CL.plot((test[1:8]/255).astype('float32'), name="test")

pred = CL.classifieur.predict((test/255).astype('float32'))


a = np.sort(np.max(pred, axis=1))

confidence = np.sum(a < 0.9)/len(a)
