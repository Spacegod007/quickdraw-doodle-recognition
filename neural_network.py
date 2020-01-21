import numpy as np
import pandas as pd

from keras.layers import Dense, Reshape, Flatten, Dropout, Conv2D, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, Adagrad
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

INPUT_SHAPE = (32 * 32,)

TEST_SIZE = 0.20
LEARNING_RATE = 0.01
LOSS_FUNCTION = "categorical_crossentropy"
BATCH_SIZE = 100
EPOCHS = 4
SAVE_MODEL_PATH = "model.h5"
WORD_COLUMN = "word"


def one_hot_encoding_label(labels):
	code = np.array(labels)
	encoder = LabelEncoder()
	return encoder.fit_transform(code)


model = Sequential([
	Reshape(input_shape=INPUT_SHAPE, target_shape=(32, 32, 1)),

	Conv2D(kernel_size=3, filters=24, use_bias=False, padding="same"),
	BatchNormalization(center=True, scale=False),
	Activation("relu"),

	Conv2D(kernel_size=6, filters=48, use_bias=False, padding="same", strides=2),
	BatchNormalization(center=True, scale=False),
	Activation("relu"),

	Conv2D(kernel_size=6, filters=64, use_bias=False, padding="same", strides=2),
	BatchNormalization(center=True, scale=False),
	Activation("relu"),

	Flatten(),

	Dense(680, activation="relu", use_bias=True),
	BatchNormalization(center=True, scale=False),
	Activation("relu"),

	Dropout(0.2),
	Dense(680, activation="relu", use_bias=True),
	BatchNormalization(center=True, scale=False),
	Activation("relu"),

	Dropout(0.2),
	Dense(340, activation="softmax")
])

# model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=LOSS_FUNCTION, metrics=['accuracy'])
model.compile(optimizer=Adagrad(lr=LEARNING_RATE), loss=LOSS_FUNCTION, metrics=['accuracy'])
print("Preparation sequence completed")

for i in range(2, 10):
	data_file = f"data\\data_{i}.csv"

	df = pd.read_csv(data_file)

	print("Data reading completed")

	words = df[WORD_COLUMN].unique()
	y = df[WORD_COLUMN].to_numpy()
	x = df.drop(columns=[WORD_COLUMN]).to_numpy()
	del df
	train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=TEST_SIZE)

	# train_data = train_data / 255 # todo check this

	print("Data preparation sequence completed")

	train_labels = one_hot_encoding_label(train_labels)
	test_labels = one_hot_encoding_label(test_labels)

	train_labels = to_categorical(train_labels, len(words))
	test_labels = to_categorical(test_labels, len(words))

	print("Label preparation sequence completed")

	print(f"Training data_{i}")

	model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_data, test_labels), verbose=2)

model.save(SAVE_MODEL_PATH)
model.summary()
