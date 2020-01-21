from keras.models import load_model

MODEL_PATH = f"models/model.h5"

model = load_model(MODEL_PATH)
model.summary()
