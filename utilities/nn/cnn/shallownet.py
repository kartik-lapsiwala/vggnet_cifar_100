# from keras.models import Sequential
# from keras.layers.convolutional import Conv2D
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dense
# from keras import backend as K


# class Shallownet:
#     @staticmethod
#     def build(width, height, depth, classes):
#         # Initialize the model along with the input shape to be 'channels_last'
#         model = Sequential()
#         input_shape = (height, width, depth)

#         # Update the image shape if 'channels_first' is being used
#         if K.image_data_format() == 'channels_first':
#             input_shape = (depth, height, width)

#         # Define the first (and only) CONV => RELU layer
#         model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
#         model.add(Activation('relu'))

#         # Add a softmax classifier
#         model.add(Flatten())
#         model.add(Dense(classes))
#         model.add(Activation('softmax'))

#         # Return the network architecture
#         return model






from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.convolutional import MaxPooling2D

class Shallownet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanelDim = -1
        """
        Batch normalization operates over the channels, so in order to apply BN, we need to
        know which axis to normalize over. Setting chanDim = -1 implies that the index of the channel
        dimension last in the input shape (i.e., channels last ordering). However, if we are using channels
        first ordering (Lines 23-25), we need need to update the inputShape and set chanDim = 1, since
        the channel dimension is now the first entry in the input shape.
        """
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanelDim = 1

        # First set of... CONV => RELU => BN => CONV => RELU => BN => POOLING => DROPOUT
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanelDim))
        model.add(Conv2D(32, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanelDim))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(rate = 0.25))

        # Second set of... CONV => RELU => BN => CONV => RELU => BN => POOLING => DROPOUT
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanelDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanelDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FULLY CONNECTED LAYER
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # SOFTMAX
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

