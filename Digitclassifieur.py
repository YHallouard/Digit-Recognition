from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, UpSampling2D, \
    Reshape, Lambda, Add , GlobalAveragePooling2D,\
    Dropout, LeakyReLU, Activation, AveragePooling1D, PReLU, Softmax, Multiply
from tensorflow.keras.models import Model, Sequential

class Classifieur():
    def __init__(self, X=None, target=None):
        """

        :type X: numpy array of size (?, 28,28)
        """
        self.X = X
        self.target = target

        optimizer = optimizers.Adam(0.0004)

        self.classifieur = self.build_RESNET_discriminator(input_shape=(28, 28, 1))
        self.classifieur.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def residual_stack(self, input=None, filters=8, down_size=2):
        """
        1D Up residual stack

        :param input: the previous layer
        :param filters: number of filter of the output
        :param down_size: the output size will be input_shape*up_size (ex (128,32) --> (256, 32) with down_size=2)

        :return Model
        """
        # 1x1 conv linear
        x = Conv2D(filters=filters, kernel_size=(1,1), strides=1, padding='same', data_format='channels_last')(input)
        x = BatchNormalization()(x)
        x = Activation('linear')(x)

        # residual unit 1
        x_shortcut = x
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('linear')(x)
        # Add skip connection
        if x.shape[1:] == x_shortcut.shape[1:]:
            x=Add()([x, x_shortcut])
        else:
            raise Exception('Skip Connection Failure')

        # residual unit 2
        x_shortcut = x
        x = Conv2D(filters=filters, kernel_size=(11,11), strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=(11,11), strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('linear')(x)
        # Add skip connection
        if x.shape[1:] == x_shortcut.shape[1:]:
            x=Add()([x, x_shortcut])
        else:
            raise Exception('Skip Connection Failure')

        # Maxpooling
        x = MaxPooling2D(pool_size=(down_size,down_size), strides=None, padding='valid', data_format='channels_last')(x)
        return(x)

    def build_RESNET_discriminator(self, input_shape=(28,28,1)):
        """
        Build the generator Discriminator part, based on RESNET Architecture with attention added

        :param input_shape: the shape of every time series to discriminate (ex = (3000,1))

        :return Model
        """

        X_shape = input_shape
        ts = Input(shape=X_shape)

        # Residual_stack
        x = self.residual_stack(ts, 8)
        x = self.residual_stack(x, 16)
        x = self.residual_stack(x, 32)


        # attention = GlobalAveragePooling1D()(x)
        # attention = Reshape((1, 128))(attention)
        # #attention = Dense(128//8, activation='relu', kernel_initializer='he_normal', use_bias=False)(attention)
        # attention = Dense(128, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(attention)
        # multiply_layer = Multiply()([attention, x])
        #
        # # attention_data = Lambda(lambda x: x[:, :, :128])(x)
        # # attention_softmax = Lambda(lambda x: x[:, :, 128:])(x)
        # # attention_softmax = Activation('sigmoid')(attention_softmax)
        # # multiply_layer = Multiply()([attention_softmax, attention_data])
        # #multiply_layer = Multiply()([attention_softmax, attention_data])
        #
        # dense_layer = Dense(128, activation='sigmoid', kernel_initializer='he_normal')(multiply_layer)
        # dense_layer = BatchNormalization()(dense_layer)
        #
        # flatten_layer = Flatten()(dense_layer)
        # output_layer = Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform')(flatten_layer)

        # # Output layer
        x = Flatten()(x)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('selu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('selu')(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(10, activation='softmax', kernel_initializer='glorot_uniform')(x)

        return Model(inputs=ts, outputs=output_layer)

    def train(self, epochs=1, batch_size=128):
        #load data
        X_train = self.X/255
        Y_train = self.target

        self.classifieur.fit(X_train, Y_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_split=0.2)

    def plot(self, X=None):
        inte = np.random.randint(0, X.shape[0], 4)
        # pred = CL.classifieur.predict(CL.X[inte] / 255)

        pred = self.classifieur.predict(X / 255)

        fig = plt.figure(figsize=(10, 5))
        for j in range(len(inte)):
            axs = fig.add_subplot(2, 2, j + 1)
            plt.imshow(X[inte[j]].astype('int').reshape(28, 28))
            for i in range(10):
                plt.text(29, i * 2, '[val : %d] [pred: %f]' % (i, round(pred[j][i], 3)), va="top", family="monospace")
            plt.xlim(0, 75)
            plt.show()

def build_target(target):
    """

    :type target: bytearray
    :param target: taget vector of target digit
    """
    new_target = np.zeros((target.shape[0], 10))

    for i in range(target.shape[0]):
        new_target[i][target[i]] = 1

    return new_target



#cl = Classifieur(X)

#cl.classifieur.summary()