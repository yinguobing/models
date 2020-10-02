import tensorflow as tf
from resnet import make_resnet18, ResNet

devices = tf.config.get_visible_devices("CPU")
tf.config.set_visible_devices(devices)

if __name__ == "__main__":
    # Functional API.
    resnet18_func = make_resnet18((224, 224, 3), 1000)
    resnet18_func.summary()
    resnet18_func.save("./saved_model/res18_f")

    # A sub classed ResNet-18.
    res18_layer_config = [2, 2, 2, 2]
    resnet18_subclassed = ResNet(res18_layer_config, bottleneck=False,
                                 output_size=1000, name="resnet_18")

    resnet18_subclassed.prepare_summary((224, 224, 3))
    resnet18_subclassed.summary()

    resnet18_subclassed.predict(tf.zeros((1, 224, 224, 3)))
    resnet18_subclassed.save("./saved_model/res18")

    # A sub classed ResNet-50.
    res50_layer_config = [3, 4, 6, 3]
    resnet50_subclassed = ResNet(res50_layer_config, bottleneck=True,
                                 output_size=1000, name="resnet_50")

    resnet50_subclassed.prepare_summary((224, 224, 3))
    resnet50_subclassed.summary()

    resnet50_subclassed.predict(tf.zeros((1, 224, 224, 3)))
    resnet50_subclassed.save("./saved_model/res50")
