import math
import typing

from tensorflow import keras as K


class Generator:
    _MIN_INITIAL_PROJECTION = 3
    _MAX_INITIAL_PROJECTION = 7

    @classmethod
    def build_model(cls,
                    out_size: int = 32,
                    out_channels: int = 3,
                    size_fact: int = 1,
                    z_dim: int = 100) -> K.Model:
        """
        Builds a DCGAN generator model
        :param out_size: Size of images generated by the model
        :param out_channels: Channels of images generated by the model
        :param size_fact: A multiplier for the size of the network... above
                          1 increases performance at the expense of speed, and
                          below 1 increases speed at the expense of performance
        :param z_dim: Dimensionality of input latent samples
        :return: A Keras model representing the generator
        """
        cls._valid_out_size_or_error(out_size)
        model = K.Sequential(name='generator')

        input_shape = (z_dim,)
        kernel_size = 5
        strides = 2
        use_bias = False

        # Calculate the initial projection size
        projection_size = cls._calc_projection_size(out_size)

        # Number of layers needed in the network to reach output size
        num_layers = int(math.log2(out_size // projection_size))

        # Number of output filters in the convolution #1 vs #n
        num_filters_layer_1 = int(projection_size * 2 ** 5 * size_fact)
        num_filters_layer_n = int(num_filters_layer_1 * 2 ** (num_layers - 1))

        # LAYER 1 --- Latent Sample Projection

        layer1_size = projection_size ** 2 * num_filters_layer_n
        layer1_shape = (projection_size, projection_size, num_filters_layer_n)

        model.add(
            K.Sequential([
                K.layers.Dense(layer1_size,
                               use_bias=use_bias,
                               input_shape=input_shape),
                K.layers.BatchNormalization(),
                K.layers.LeakyReLU(),
                K.layers.Reshape(layer1_shape)
            ], name='projection')
        )

        # LAYERS 2 to `num_layers` --- Convolutional Layers

        i = 1
        for i in range(1, num_layers):
            num_filters = num_filters_layer_1 * 2 ** (num_layers - i - 1)
            model.add(
                K.Sequential([
                    K.layers.Conv2DTranspose(num_filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding='same',
                                             use_bias=use_bias),
                    K.layers.BatchNormalization(),
                    K.layers.LeakyReLU()
                ], name='generator_conv_{}'.format(i))
            )

        # FINAL LAYER --- Convolution & Map RGB/L Channels

        model.add(
            K.Sequential([
                K.layers.Conv2DTranspose(filters=out_channels,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding='same',
                                         activation=K.activations.tanh)
            ], name='generator_conv_{}'.format(i + 1))
        )

        # # #

        return model

    @classmethod
    def prev_valid_size(cls, size: int) -> int:
        """
        Finds the next lowest valid generator output size for a specified
        output size. If the size is valid, the same value is returned
        :param size: Size to search from
        :return: Next lowest valid size, or `size` if it is valid
        """
        min_out_size = list(Generator.get_valid_sizes(limit=20))[0]
        if size < min_out_size:
            return min_out_size

        return list(cls.get_valid_sizes(limit=size))[-1]

    @classmethod
    def get_valid_sizes(cls, limit=100) -> typing.Generator[int, None, None]:
        """
        :return: A list of valid output sizes for a generator up to a limit
        """
        size = cls._MAX_INITIAL_PROJECTION
        while size < limit:
            size += 1
            fact = size
            while fact // 2 == fact / 2 and fact != 0:
                fact /= 2

            if fact <= cls._MAX_INITIAL_PROJECTION:
                yield size

    @classmethod
    def _calc_projection_size(cls,
                              out_size: int,
                              min_p: int = _MIN_INITIAL_PROJECTION,
                              max_p: int = _MAX_INITIAL_PROJECTION) -> int:
        """
        Given a desired output size for a generator network, calculate what the
        initial projection size of the latent samples must be. Throw an
        exception if the answer is outside of a reasonable range. The ideal
        projection size is 4 but that leaves only powers of 2 starting at 4 to
        be allowed output sizes. Broadening the allowable range allows for more
        possible output sizes, but the effects on image quality are unknown.
        :param out_size: Desired output size of a generator network
        :param min_p: Minimum allowed projection size
        :param max_p: Maximum allowed projection size
        :return: Projection size
        """
        # find prime factors of the output size
        prime_factors = cls._primes(out_size)

        if 2 not in prime_factors:
            # The output size does not have 2 as a factor...
            msg = 'out_size `{}` does not have 2 as a factor'.format(out_size)
            raise GeneratorProjectionError(msg)
        else:
            while 2 in prime_factors:
                prime_factors.remove(2)

        prime_factors = list(prime_factors)
        if len(prime_factors) == 0:
            # This means the only prime factor was 2 so base should be 4
            initial_size = 4
        else:
            # Multiply all the remaining primes to get initial size
            initial_size = cls._mul(prime_factors)

        if min_p <= initial_size <= max_p:
            return initial_size
        else:
            msg = '`{0}` is an unsupported out_size because its required ' \
                  'initial projection size of `{1}` is outside of the range ' \
                  '[{2}, {3}] that is allowed.'.format(out_size,
                                                       initial_size,
                                                       min_p,
                                                       max_p)
            raise GeneratorProjectionError(msg)

    @staticmethod
    def _valid_out_size_or_error(out_size: int) -> None:
        """
        Raises an error if an output size is invalid for a generator
        :param out_size: Output size
        """
        min_out_size = list(Generator.get_valid_sizes(limit=20))[0]
        if out_size < min_out_size:
            raise GeneratorProjectionError()

    @staticmethod
    def _primes(n: int) -> list:
        """
        Finds the prime factors of an integer
        :param n: integer to find prime factors for
        :return: a list of the integer's prime factors
        """
        prime_factors = []
        d = 2
        while d * d <= n:
            while (n % d) == 0:
                prime_factors.append(d)
                n //= d
            d += 1
        if n > 1:
            prime_factors.append(n)
        return prime_factors

    @staticmethod
    def _mul(arr: list) -> int:
        """
        Finds the product of all elements in a list
        :param arr: The list to multiply
        :return: Product of all elements
        """
        if len(arr) == 0:
            raise ValueError('Cannot multiply an empty list')

        product = 1
        for i in arr:
            product *= i

        return product


class GeneratorProjectionError(Exception):
    """
    Occurs when an output size requires an out of range initial projection size
    """
    pass