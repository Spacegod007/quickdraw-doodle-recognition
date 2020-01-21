from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

INPUT_SHAPE = (32, 32,)

TEST_SIZE = 0.20
LEARNING_RATE = 0.01
LOSS_FUNCTION = "categorical_crossentropy"
BATCH_SIZE = 512
EPOCHS = 5
OLD_MODEL_PATH = r"models\model.h5"
SAVE_MODEL_PATH = r"models\model_2.h5"
DATA_PATH = r"D:\Users\Jordi\image data"
CLASS_MODE = "categorical"

# 'rescale=' if image is inserted at wrong size
data_generator = ImageDataGenerator(validation_split=TEST_SIZE, horizontal_flip=True)

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

model = load_model(OLD_MODEL_PATH)

model.fit_generator(train_generator,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    shuffle=True,
                    callbacks=[ModelCheckpoint(SAVE_MODEL_PATH)],
                    verbose=1,
                    initial_epoch=1,
                    # steps_per_epoch=250,
                    # max_queue_size=5,
                    )

model.save(SAVE_MODEL_PATH)
model.summary()
