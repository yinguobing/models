"""A human friendly implimentation of High-Resolution Net."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from resnet import BottleneckBlock, ResidualBlock


class HRNBlock(layers.Layer):
    def __init__(self, filters=64, activation='relu'):
        super(HRNBlock, self).__init__()

        # There are 4 residual blocks in each modularized block.
        self.residual_block_1 = ResidualBlock(filters, False, activation)
        self.residual_block_2 = ResidualBlock(filters, False, activation)
        self.residual_block_3 = ResidualBlock(filters, False, activation)
        self.residual_block_4 = ResidualBlock(filters, False, activation)

    def call(self, inputs):
        x = self.residual_block_1(inputs)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)

        return x


class HRNBlocks(layers.Layer):
    def __init__(self, repeat=1, filters=64, activation='relu'):
        super(HRNBlocks, self).__init__()
        self.blocks = [ResidualBlock(filters, False, activation)
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
        if upsample:
            self.upsample_layer = layers.UpSampling2D(size=(2, 2),
                                                      interpolation='bilinear')
            self.match_channels = layers.Conv2D(filters=filters,
                                                kernel_size=(1, 1),
                                                strides=(1, 1),
                                                padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

    def call(self, inputs):
        if self.upsample:
            x = self.match_channels(inputs)
            x = self.upsample_layer(x)
        else:
            x = self.downsample_layer(inputs)
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

             B1         B2         B3         B4
        |----------|----------|----------|----------|
    B1  | identity |    ->    |    ->    |    ->    |
        |----------|----------|----------|----------|
    B2  |    <-    | identity |    ->    |    ->    |
        |----------|----------|----------|----------|
    B3  |    <-    |    <-    | identity |    ->    |
        |----------|----------|----------|----------|
    """

    def __init__(self, filters, branches_in, branches_out, activation='relu'):
        super(FusionBlock, self).__init__()
        # Construct the fusion layers.
        self._fusion_grid = []

        for row in range(branches_in):
            fusion_layers = []
            for column in range(branches_out):
                if column == row:
                    fusion_layers.append(Identity())
                elif column > row:
                    # Down sampling.
                    fusion_layers.append(FusionLayer(filters * pow(2, column),
                                                     False, activation))
                else:
                    # Up sampling.
                    fusion_layers.append(FusionLayer(filters * pow(2, column),
                                                     True, activation))

            self._fusion_grid.append(fusion_layers)

        self._add_layers_group = [layers.Add() for _ in range(branches_out)]

    def call(self, inputs):
        """Fuse the last layer's outputs. The inputs should be a list of the last layers output tensors in order of branches."""
        rows = len(self._fusion_grid)
        columns = len(self._fusion_grid[0])

        # Every cell in the fusion grid has an output value.
        fusion_values = [[None for _ in range(columns)] for _ in range(rows)]

        for row in range(rows):
            # The down sampling operation excutes from left to right.
            for column in range(columns):
                # The input will be different for different cells.
                if column < row:
                    # Skip all up samping cells.
                    continue
                elif column == row:
                    # The input is the branch output.
                    x = inputs[row]
                elif column > row:
                    # Down sampling, the input is the fusion value of the left cell.
                    x = fusion_values[row][column - 1]
                fusion_values[row][column] = self._fusion_grid[row][column](x)

            # The upsampling operation excutes in the opposite direction.
            for column in reversed(range(columns)):
                if column >= row:
                    # Skip all down samping and identity cells.
                    continue
                x = fusion_values[row][column + 1]
                fusion_values[row][column] = self._fusion_grid[row][column](x)

        # The fused value for each branch.
        if rows == 1:
            outputs = [fusion_values[0][0], fusion_values[0][1]]
        else:
            outputs = []
            fusion_values = [list(v) for v in zip(*fusion_values)]

            for index, values in enumerate(fusion_values):
                outputs.append(self._add_layers_group[index](values))

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

        self.s1_fusion = FusionBlock(filters, branches_in=1, branches_out=2)

        # Stage 2
        self.s2_b1_block = HRNBlock(filters)
        self.s2_b2_block = HRNBlock(filters*2)

        self.s2_fusion = FusionBlock(filters, branches_in=2, branches_out=3)

        # Stage 3
        self.s3_b1_blocks = HRNBlocks(4, filters)
        self.s3_b2_blocks = HRNBlocks(4, filters*2)
        self.s3_b3_blocks = HRNBlocks(4, filters*4)

        self.s3_fusion = FusionBlock(filters, branches_in=3, branches_out=4)

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
