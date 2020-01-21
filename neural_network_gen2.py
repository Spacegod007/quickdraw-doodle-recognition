from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, Reshape, Flatten, Dropout, Conv2D, Activation, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

INPUT_SHAPE = (32, 32,)

TEST_SIZE = 0.20
LEARNING_RATE = 0.01
LOSS_FUNCTION = "categorical_crossentropy"
BATCH_SIZE = 512
EPOCHS = 3
CHECKPOINT_SAVE_MODEL_PATH = r"models\model_test_checkpoint.h5"
FINAL_SAVE_MODEL_PATH = r"models\model_test.h5"
DATA_PATH = r"D:\Users\Jordi\image data"
# DATA_PATH = r"E:\Users\jordi\Documents\PythonProjects\Data Challenge 2\train_simplified_images\image data"
CLASS_MODE = "categorical"

# 'rescale=' if image is inserted at wrong size
data_generator = ImageDataGenerator(validation_split=TEST_SIZE, horizontal_flip=True, rescale=1/255.)

train_generator = data_generator.flow_from_directory(DATA_PATH,
                                                     class_mode=CLASS_MODE,
                                                     batch_size=BATCH_SIZE,
                                                     subset="training",
                                                     color_mode="grayscale",
                                                     target_size=(32, 32))

validation_generator = data_generator.flow_from_directory(DATA_PATH,
                                                          class_mode=CLASS_MODE,
                                                          batch_size=BATCH_SIZE,
                                                          subset="validation",
                                                          color_mode="grayscale",
                                                          target_size=(32, 32))

model = Sequential([
	Conv2D(input_shape=(32, 32, 1), kernel_size=3, filters=32, use_bias=False, padding='same'),
	BatchNormalization(center=True, scale=False),
	Activation('relu'),
	MaxPooling2D(pool_size=(2, 2), strides=1, padding="same"),

	Conv2D(kernel_size=6, filters=64, use_bias=False, padding='same', strides=2, activation="relu"),
	BatchNormalization(center=True, scale=False),
	Activation('relu'),
	MaxPooling2D(pool_size=(2, 2), strides=1, padding="same"),

	Conv2D(kernel_size=6, filters=96, use_bias=False, padding='same', strides=2, activation="relu"),
	BatchNormalization(center=True, scale=False),
	Activation('relu'),
	MaxPooling2D(pool_size=(2, 2), strides=1, padding="same"),

	Flatten(),

	Dense(680, use_bias=False),
	BatchNormalization(center=True, scale=False),
	Activation('relu'),
	Dropout(0.3),

	Dense(680, use_bias=False),
	BatchNormalization(center=True, scale=False),
	Activation('relu'),
	Dropout(0.3),

	Dense(340, activation='softmax')
])

model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=LOSS_FUNCTION, metrics=['accuracy'])

model.fit_generator(train_generator,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    shuffle=True,
                    callbacks=[ModelCheckpoint(CHECKPOINT_SAVE_MODEL_PATH)],
                    verbose=1)

model.save(FINAL_SAVE_MODEL_PATH)
model.summary()
