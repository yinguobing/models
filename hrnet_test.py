from tensorflow import keras
from hrnet import HRNetBody

if __name__ == "__main__":
    inputs = keras.Input((256, 256, 3))
    body= HRNetBody(name="HRNetBody")
    outputs = body(inputs)

    model = keras.Model(inputs, outputs)
    model.summary()

    model.save("./saved_model/hrnet_body")
