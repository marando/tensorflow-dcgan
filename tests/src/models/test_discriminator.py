from unittest import TestCase

from src.models import Discriminator
from src.models.discriminator import DiscriminatorSizeError


class TestDiscriminator(TestCase):

    def test_layers_24(self):
        for c in (3, 1):
            output_shapes = self._make_discriminator_get_output_shapes(24, c)
            self.assertEqual(output_shapes, [(None, 12, 12, 48),
                                             (None, 6, 6, 96),
                                             (None, 3, 3, 192),
                                             (None, 1)])

    def test_layers_8(self):
        for c in (3, 1):
            output_shapes = self._make_discriminator_get_output_shapes(8, c)
            self.assertEqual(output_shapes, [(None, 4, 4, 64),
                                             (None, 1)])

    def test_layers_32(self):
        for c in (3, 1):
            output_shapes = self._make_discriminator_get_output_shapes(32, c)
            self.assertEqual(output_shapes, [(None, 16, 16, 64),
                                             (None, 8, 8, 128),
                                             (None, 4, 4, 256),
                                             (None, 1)])

    def test_layers_64(self):
        for c in (3, 1):
            output_shapes = self._make_discriminator_get_output_shapes(64, c)
            self.assertEqual(output_shapes, [(None, 32, 32, 64),
                                             (None, 16, 16, 128),
                                             (None, 8, 8, 256),
                                             (None, 4, 4, 512),
                                             (None, 1)])

    def test_layers_20(self):
        for c in (3, 1):
            output_shapes = self._make_discriminator_get_output_shapes(20, c)
            self.assertEqual(output_shapes, [(None, 10, 10, 80),
                                             (None, 5, 5, 160),
                                             (None, 1)])

    def test_layers_28(self):
        for c in (3, 1):
            output_shapes = self._make_discriminator_get_output_shapes(28, c)
            self.assertEqual(output_shapes, [(None, 14, 14, 112),
                                             (None, 7, 7, 224),
                                             (None, 1)])

    def test_too_small_size(self):
        """
        Tests that building a discriminator with too small of a size raises a
        DiscriminatorSizeError
        """
        for size in range(0, 8):
            with self.assertRaises(DiscriminatorSizeError):
                Discriminator.build_model(in_size=size, in_channels=3)

            with self.assertRaises(DiscriminatorSizeError):
                Discriminator.build_model(in_size=size, in_channels=1)

    @staticmethod
    def _make_discriminator_get_output_shapes(in_size: int,
                                              in_channels: int) -> list:
        """
        Makes a discriminator and returns a list of its layers' output shapes
        :param in_size: input size
        :param in_channels: input channels
        :return: a list of tuples for each layer's output shape
        """
        return [layer.output_shape for layer in
                Discriminator.build_model(in_size=in_size,
                                          in_channels=in_channels).layers]
