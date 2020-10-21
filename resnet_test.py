import tensorflow as tf
from resnet import make_resnet, ResNet

devices = tf.config.get_visible_devices("CPU")
tf.config.set_visible_devices(devices)

if __name__ == "__main__":

    # RESNET 18
    res18_layer_config = [2, 2, 2, 2]
    input_shape = (256, 256, 3)

    # Functional API.
    resnet18_func = make_resnet(res18_layer_config,
                                bottleneck=False,
                                input_shape=input_shape,
                                output_size=1000,
                                name="resnet_18_func")
    resnet18_func.summary()
    resnet18_func.save("./saved_model/res18_func")

    # Sub classed.
    resnet18_subclassed = ResNet(res18_layer_config,
                                 bottleneck=False,
                                 output_size=1000,
                                 name="resnet_18_subc")
    resnet18_subclassed.prepare_summary(input_shape)
    resnet18_subclassed.summary()
    resnet18_subclassed.predict(tf.zeros((1, 224, 224, 3)))
    resnet18_subclassed.save("./saved_model/res18_subc")

    # ResNet-50
    res50_layer_config = [3, 4, 6, 3]

    # Functional API
    resnet50_func = make_resnet(res50_layer_config,
                                bottleneck=True,
                                input_shape=input_shape,
                                output_size=1000,
                                name="resnet_50_func")
    resnet50_func.summary()
    resnet50_func.save("./saved_model/res50_func")

    # Sub classed
    resnet50_subclassed = ResNet(res50_layer_config,
                                 bottleneck=True,
                                 output_size=1000,
                                 name="resnet_50_subc")

    resnet50_subclassed.prepare_summary(input_shape)
    resnet50_subclassed.summary()
    resnet50_subclassed.predict(tf.zeros((1, 224, 224, 3)))
    resnet50_subclassed.save("./saved_model/res50_subc")
