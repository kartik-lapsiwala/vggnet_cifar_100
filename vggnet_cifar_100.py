from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import SimplePreprocessor
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.cnn import Shallownet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
args = vars(ap.parse_args())

# Grab the list of images
print('[INFO]: Loading images....')
image_paths = list(paths.list_images(args['dataset']))

# Initialize the preprocessors
sp = SimplePreprocessor(32, 32)
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, itap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

# Split the data into training data (75%) and testing data (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

# Initialize the optimizer and model
print('[INFO]: Compiling model....')
optimizer = SGD(lr=0.005)
model = Shallownet.build(width=32, height=32, depth=3, classes=100)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the network
print('[INFO]: Training the network....')
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=40, verbose=1)

# Test the network
labelNames = [
'beaver', 'dolphin', 'otter', 'seal', 'whale',
'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
'bottles', 'bowls', 'cans', 'cups', 'plates',
'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
'bed', 'chair', 'couch', 'table', 'wardrobe',
'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
'bear', 'leopard', 'lion', 'tiger', 'wolf',
'bridge', 'castle', 'house', 'road', 'skyscraper',
'cloud', 'forest', 'mountain', 'plain', 'sea',
'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
'crab', 'lobster', 'snail', 'spider', 'worm',
'baby', 'boy', 'girl', 'man', 'woman',
'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
'maple', 'oak', 'palm', 'pine', 'willow',
'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
]

print('[INFO]: Evaluating the network....')
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()