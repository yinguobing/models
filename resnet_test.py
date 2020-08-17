import tensorflow as tf
from resnet import make_resnet18, ResNet18

if __name__ == "__main__":
    # Functional API.
    resnet18_func = make_resnet18((224, 224, 3), 1000)
    resnet18_func.summary()

    # Sub classed.
    resnet18_subclassed = ResNet18(1000)
    resnet18_subclassed(tf.zeros((1, 224, 224, 3)))
    resnet18_subclassed.summary()
