import math

from tensorflow import keras as K


class Discriminator:
    MIN_SIZE = 8

    @classmethod
    def build_model(cls,
                    in_size: int = 32,
                    in_channels: int = 3,
                    size_fact: int = 1,
                    dropout: float = 0.3) -> K.Model:
        """
        Builds a DCGAN discriminator model
        :param in_size: Image input size to the model
        :param in_channels: Number of channels in the input image
        :param size_fact: A multiplier for the size of the network... above
                          1 increases performance at the expense of speed, and
                          below 1 increases speed at the expense of performance
        :param dropout: Overfitting dropout rate for each convolution layer
        :return: A Keras model representing the discriminator
        """
        cls._valid_in_size_or_error(in_size)
        model = K.Sequential(name='discriminator')

        input_shape = (in_size, in_size, in_channels)
        kernel_size = 5
        strides = 2

        # Find final size before densely-connected NN layer & number of layers
        final_size = cls._calc_final_size(in_size)
        num_layers = int(math.log2(in_size // final_size))

        # LAYER 1 to `num_layers` --- Convolutional Layers

        for i in range(num_layers):
            num_filters = int(final_size * 2 ** (4 + i) * size_fact)

            if i > 0:
                input_shape = model.output_shape[-3:]

            model.add(
                K.Sequential([
                    K.layers.Conv2D(filters=num_filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding='same',
                                    input_shape=input_shape),
                    K.layers.LeakyReLU(),
                    K.layers.Dropout(dropout)
                ], name='discriminator_conv_{}'.format(i + 1))
            )

        # FINAL LAYER --- Map to 0 or 1

        model.add(
            K.Sequential([
                K.layers.Flatten(),
                K.layers.Dense(1)
            ], name='final')
        )

        # # #

        return model

    @classmethod
    def _valid_in_size_or_error(cls, in_size: int) -> None:
        """
        Raises an error if an input size is invalid for a discriminator
        :param in_size: Input size
        """
        if in_size < cls.MIN_SIZE:
            raise DiscriminatorSizeError('Minimum discriminator size is {}'
                                         ''.format(cls.MIN_SIZE))

    @staticmethod
    def _calc_final_size(in_size: int) -> int:
        """
        Calculates the final size of an input size before the end
        densely-connected NN layer.
        :param in_size: Input size
        :return: The final size
        """
        final_size = in_size
        while True:
            if final_size // 2 == final_size / 2:  # Is it an int?
                # final_size / 2 is an int ... save and proceed
                final_size = final_size // 2
            else:
                # final size / 2 is a float... use the final size or 4 if 2
                return final_size if final_size > 2 else 4


class DiscriminatorSizeError(Exception):
    pass
