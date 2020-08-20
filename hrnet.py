"""A human friendly implimentation of High-Resolution Net."""

from tensorflow import keras
from tensorflow.keras import layers
from resnet import ResidualBlock, BottleneckBlock


class HRNBlock(layers.Layer):
    def __init__(self, filters=64, activation='relu'):
        super(HRNBlock, self).__init__()

        self.activation_fun = activation

        # There are 4 residual blocks in each modularized block.
        self.residual_block_1 = ResidualBlock(filters, activation)
        self.residual_block_2 = ResidualBlock(filters, activation)
        self.residual_block_3 = ResidualBlock(filters, activation)
        self.residual_block_4 = ResidualBlock(filters, activation)

    def call(self, inputs):
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)

        return x


class HRNBlocks(layers.Layer):
    def __init__(self, repeat=1, filters=64, activation='relu'):
        super(HRNBlocks, self).__init__()

        self.activation_fun = activation

        self.blocks = [ResidualBlock(filters, activation)
                       for _ in range(repeat)]

    def call(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)

        return inputs


class FusionLayer(layers.Layer):
    """A fusion layer actually do two things: resize the maps, match the channels"""

    def __init__(self, filters, upsample=False, activation='relu'):
        self.upsample = upsample
        self.downsample = layers.Conv2D(filters=filters,
                                        kernel_size=(3, 3),
                                        strides=(2, 2),
                                        padding='same')
        self.upsample = layers.UpSampling2D(size=(2, 2),
                                            interpolation='bilinear')
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

    def call(self, inputs):
        resample = self.upsample if self.upsample is True else self.downsample
        x = resample(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class FusionBlock(layers.Layer):
    """A fusion block will fuse multi-resolution inputs."""

    def __init__(self):
        # TODO: implementation this method.

    def call(self, inputs):
        # TODO: implementation this method.
        return inputs


class HRNetBody(keras.Model):
    def __init__(self, filters=64):
        super(HRNetV1, self).__init__()

        # Stage 1
        self.s1_bottleneck_1 = BottleneckBlock(64)
        self.s1_bottleneck_2 = BottleneckBlock(64)
        self.s1_bottleneck_3 = BottleneckBlock(64)
        self.s1_bottleneck_4 = BottleneckBlock(64)

        self.s1_fusion_12 = FusionLayer(filters*2)

        # Stage 2
        self.s2_b1_block = HRNBlock(filters)
        self.s2_b2_block = HRNBlock(filters*2)

        self.s2_fusion_12 = FusionLayer(filters*2)
        self.s2_fusion_13 = FusionLayer(filters*4)

        self.s2_fusion_21 = FusionLayer(filters, upsample=True)
        self.s2_fusion_23 = FusionLayer(filters*4)

        # Stage 3
        self.s3_b1_blocks = HRNBlocks(4, filters)
        self.s3_b2_blocks = HRNBlocks(4, filters*2)
        self.s3_b3_blocks = HRNBlocks(4, filters*4)

        # TODO: replace these fusion layers with fusion blocks.
        self.s3_fusion_12 = FusionLayer(filters*2)
        self.s3_fusion_23 = FusionLayer(filters*4)
        self.s3_fusion_34 = FusionLayer(filters*8)
        self.s3_fusion_21 = FusionLayer(filters, upsample=True)
        self.s3_fusion_32 = FusionLayer(filters*2, upsample=True)

        # Stage 4
        self.s4_b1_blocks = HRNBlocks(3, filters)
        self.s4_b2_blocks = HRNBlocks(3, filters*2)
        self.s4_b3_blocks = HRNBlocks(3, filters*4)
        self.s4_b4_blocks = HRNBlockS(3, filters*8)

    def call(self, inputs):
        # TODO: implimentation call method.
        return inputs
