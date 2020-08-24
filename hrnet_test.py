from tensorflow import keras
from hrnet import HRNetBody

if __name__ == "__main__":
    model = HRNetBody(name="HRNetBody")
    model(keras.Input((256, 256, 256)))
    model.summary()
