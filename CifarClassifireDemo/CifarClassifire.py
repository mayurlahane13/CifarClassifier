import _pickle as cPickle
import numpy as np
import keras
import scipy 
from keras import optimizers
import matplotlib.image as mpimg
import png
import tensorflowjs as tfjs
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def unpickle(file):
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo, encoding='bytes')
	return dict

inputs = []
labels = []
inputs = np.array(inputs)
labels = np.array(labels, dtype = int)

for i in range(5):	
	images = unpickle('cifar-10-batches-py/data_batch_'+ str(i + 1))
	label = np.array(images[b'labels'], dtype = int)
	imgData = np.resize(images[b'data'], (10000, 32, 32, 3)) / 255
	labels = np.append(labels, label)
	inputs = np.resize(np.append(inputs, imgData), (len(inputs) + len(imgData), 32, 32, 3))

outputs = []
for label in labels:
	output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	output[label] = 1
	outputs.append(output)

outputs = np.array(outputs)


model = keras.models.Sequential()

# number of convolutional filters
#n_filters = 32

# convolution filter size
# i.e. we will use a n_conv x n_conv filter
n_conv = 3

# pooling window size
# i.e. we will use a n_pool x n_pool pooling window
n_pool = 2


model.add(keras.layers.Conv2D(64, (3, 3), padding = 'same',
		# we have a 32x32 three channel  image
        # so the input shape should be (32, 32, 3)
        strides = (1, 1),
        input_shape= (32, 32, 3),
        activation = 'relu'
	))
#model.add(keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu', strides = (1, 1)))
model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(n_pool, n_pool), strides = (n_pool, n_pool)))


model.add(keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu', strides = (1, 1)))
model.add(keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu', strides = (1, 1)))


model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(n_pool, n_pool), strides = (n_pool, n_pool)))




# flatten the data for the 1D layers
model.add(keras.layers.Flatten())

#1st dense layer
model.add(keras.layers.Dense(1024, activation = 'relu'))



#output layer
#the softmax output layer gives us a probablity for each class
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))


opt = keras.optimizers.Adam(lr=0.0007, decay=1e-6)

model.compile(
		loss = 'categorical_crossentropy',
		optimizer = 'adam',
		metrics = ['accuracy']
	)

datagen = ImageDataGenerator(
         zoom_range=0.2, # randomly zoom into images
         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


validationInputs = []
validationLabels = []
validationInputs = np.array(validationInputs)
validationLabels = np.array(validationLabels)
for i in range(5000):
	dtI = inputs[i]
	dtO = outputs[i]
	validationInputs = np.append(validationInputs, dtI)
	validationLabels = np.append(validationLabels, dtO)

validationInputs = np.resize(validationInputs, (5000, 32, 32, 3))

validationLabels = np.resize(validationLabels, (5000, 10))

testData = unpickle('cifar-10-batches-py/test_batch')

testLabels = np.array(testData[b'labels'], dtype = int)

testOutputs = []
for label in testLabels:
	testOutput = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	testOutput[label] = 1
	testOutputs.append(testOutput)

testOutputs = np.array(testOutputs)

testImages = np.resize(testData[b'data'], (10000, 32, 32, 3)) / 255


inputs = np.append(inputs, testImages)
outputs = np.append(outputs, testOutputs)
inputs = np.resize(inputs, (60000, 32, 32, 3))
outputs = np.resize(outputs, (60000, 10))



#model.fit(inputs, outputs, batch_size = 200, epochs = 30, shuffle = True, validation_data = (testImages, testOutputs))
#model.fit(inputs, outputs, batch_size = 128, epochs = 30, shuffle = True, validation_split = 0.1)
print(inputs.shape)
print(outputs.shape)
print(validationInputs.shape)
print(validationLabels.shape)


model.fit_generator(datagen.flow(inputs, outputs, batch_size = 100), epochs = 40, shuffle = True, validation_data = (validationInputs, validationLabels))

model.save('CifarSavedModel/cifarCnnModel.h5')

'''model = keras.models.load_model('CifarSavedModel/cifarCnnModel.h5')



predictions = model.predict(testImages)

correct = 0

for i in range(len(predictions)):
	if np.argmax(predictions[i]) == testLabels[i]:
		correct += 1

print("Testing accuracy:", correct / len(testLabels)) '''
#print(model.summary())

#tfjs.converters.save_keras_model(model, '/Users/makarandsubhashlahane/Desktop/Projects/JavaScript/Tensorflow.jsProjects/CifarClassifire/kerasCifarModel')

#labels_info = unpickle('cifar-10-batches-py/batches.meta')

#print(labels_info)


