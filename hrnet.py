"""A human friendly implimentation of High-Resolution Net."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from resnet import BottleneckBlock, ResidualBlock


class HRNBlock(layers.Layer):
    def __init__(self, filters=64, activation='relu'):
        super(HRNBlock, self).__init__()

        # There are 4 residual blocks in each modularized block.
        self.residual_block_1 = ResidualBlock(filters, activation)
        self.residual_block_2 = ResidualBlock(filters, activation)
        self.residual_block_3 = ResidualBlock(filters, activation)
        self.residual_block_4 = ResidualBlock(filters, activation)

    def call(self, inputs):
        x = self.residual_block_1(inputs)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)

        return x


class HRNBlocks(layers.Layer):
    def __init__(self, repeat=1, filters=64, activation='relu'):
        super(HRNBlocks, self).__init__()
        self.blocks = [ResidualBlock(filters, activation)
                       for _ in range(repeat)]

    def call(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)

        return inputs


class FusionLayer(layers.Layer):
    """A fusion layer actually do two things: resize the maps, match the channels"""

    def __init__(self, filters, upsample=False, activation='relu'):
        super(FusionLayer, self).__init__()
        self.upsample = upsample
        self.downsample_layer = layers.Conv2D(filters=filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='same')
        self.upsample_layer = layers.UpSampling2D(size=(2, 2),
                                                  interpolation='bilinear')
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

    def call(self, inputs):
        resample = self.upsample_layer if self.upsample is True else self.downsample_layer
        x = resample(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class Identity(layers.Layer):
    """A identity layer do NOT modify the tensors."""

    def __init__(self):
        super(Identity, self).__init__()

    def call(self, inputs):
        return tf.identity(inputs)


class FusionBlock(layers.Layer):
    """A fusion block will fuse multi-resolution inputs.

    A typical fusion block looks like a square box with cells. For example at
    stage 3, the fusion block consists 12 cells. Each cell represents a fusion
    layer. Every cell whose row < column is a down sampling cell, whose row ==
    column is a identity cell, and the rest are up sampling cells.

    |----------|----------|----------|
    | identity |    up    |    up    |
    |----------|----------|----------|
    |   down   | identity |    up    |
    |----------|----------|----------|
    |   down   |   down   | identity |
    |----------|----------|----------|
    |   down   |   down   |   down   |
    |----------|----------|----------|

    """

    def __init__(self, filters, stage=1, activation='relu'):
        super(FusionBlock, self).__init__()
        # Construct the fusion grid.
        columns = stage
        rows = stage + 1
        self._fusion_function_grid = []

        for row in range(rows):
            fusion_group = []
            for column in range(columns):
                if column == row:
                    fusion_group.append(Identity())
                elif column < row:
                    # Down sampling.
                    fusion_group.append(FusionLayer(filters * pow(2, row),
                                                    activation))
                else:
                    # Up sampling.
                    fusion_group.append(FusionLayer(filters * pow(2, row),
                                                    True, activation))
            self._fusion_function_grid.append(fusion_group)

        self._add_layers_group = [layers.Add() for _ in range(rows)]

    def call(self, inputs):
        """Fuse the last layer's outputs. The inputs should be a list of the
        last layers output tensors in order of branches."""
        rows = len(self._fusion_function_grid)
        columns = rows - 1

        # Every cell in the fusion grid has an output value.
        fusion_values = [[None for _ in range(columns)] for _ in range(rows)]

        for row in range(rows):
            for column in range(columns):
                # The input will be different for different cells.
                if column == row:
                    # The input is the branch output.
                    _inputs = inputs[row]
                elif column < row:
                    # Down sampling. The input is the fusion value of upper cell.
                    _inputs = fusion_values[row - 1][column]
                elif column > row:
                    # Up sampling. The input is the fusion value of the lower cell.
                    _inputs = fusion_values[row + 1][column]

                fusion_values[row][column] = self._fusion_function_grid[row][column](
                    _inputs)

        # The fused value for each branch.
        if columns == 1:
            outputs = fusion_values
        else:
            outputs = []
            for index, fusion_group in enumerate(fusion_values):
                outputs.append(self._add_layers_group[index](fusion_group))

        return outputs


class HRNetBody(keras.Model):
    def __init__(self, filters=64):
        super(HRNetBody, self).__init__()

        # Stage 1
        self.s1_bottleneck_1 = BottleneckBlock(64)
        self.s1_bottleneck_2 = BottleneckBlock(64)
        self.s1_bottleneck_3 = BottleneckBlock(64)
        self.s1_bottleneck_4 = BottleneckBlock(64)
        self.s1_conv3x3 = layers.Conv2D(filters=filters,
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding='same')
        self.s1_batch_norm = layers.BatchNormalization()

        self.s1_fusion = FusionBlock(filters, 1)

        # Stage 2
        self.s2_b1_block = HRNBlock(filters)
        self.s2_b2_block = HRNBlock(filters*2)

        self.s2_fusion = FusionBlock(filters, 2)

        # Stage 3
        self.s3_b1_blocks = HRNBlocks(4, filters)
        self.s3_b2_blocks = HRNBlocks(4, filters*2)
        self.s3_b3_blocks = HRNBlocks(4, filters*4)

        self.s3_fusion = FusionBlock(filters, 3)

        # Stage 4
        self.s4_b1_blocks = HRNBlocks(3, filters)
        self.s4_b2_blocks = HRNBlocks(3, filters*2)
        self.s4_b3_blocks = HRNBlocks(3, filters*4)
        self.s4_b4_blocks = HRNBlocks(3, filters*8)

    def call(self, inputs):
        # Stage 1
        x = self.s1_bottleneck_1(inputs)
        x = self.s1_bottleneck_2(x)
        x = self.s1_bottleneck_3(x)
        x = self.s1_bottleneck_4(x)
        x = self.s1_conv3x3(x)
        x = self.s1_batch_norm(x)
        x = self.s1_fusion([x])

        # Stage 2
        x_1 = self.s2_b1_block(x[0])
        x_2 = self.s2_b2_block(x[1])
        x = self.s2_fusion([x_1, x_2])

        # Stage 3
        x_1 = self.s3_b1_blocks(x[0])
        x_2 = self.s3_b2_blocks(x[1])
        x_3 = self.s3_b3_blocks(x[2])
        x = self.s3_fusion([x_1, x_2, x_3])

        # Stage 4
        x_1 = self.s4_b1_blocks(x[0])
        x_2 = self.s4_b2_blocks(x[1])
        x_3 = self.s4_b3_blocks(x[2])
        x_4 = self.s4_b4_blocks(x[3])

        return [x_1, x_2, x_3, x_4]


if __name__ == "__main__":
    model = HRNetBody()
    model(tf.zeros((1, 224, 224, 3)))
    model.summary()
