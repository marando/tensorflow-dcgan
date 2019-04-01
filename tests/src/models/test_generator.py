from unittest import TestCase

from src.models import Generator
from src.models.generator import GeneratorProjectionError


class TestGenerator(TestCase):

    def test_layers_24(self):
        for c in (3, 1):
            output_shapes = self._make_generator_get_output_shapes(24, c)
            self.assertEqual(output_shapes, [(None, 3, 3, 384),
                                             (None, 6, 6, 192),
                                             (None, 12, 12, 96),
                                             (None, 24, 24, c)])

    def test_layers_8(self):
        for c in (3, 1):
            output_shapes = self._make_generator_get_output_shapes(8, c)
            self.assertEqual(output_shapes, [(None, 4, 4, 128),
                                             (None, 8, 8, c)])

    def test_layers_32(self):
        for c in (3, 1):
            output_shapes = self._make_generator_get_output_shapes(32, c)
            self.assertEqual(output_shapes, [(None, 4, 4, 512),
                                             (None, 8, 8, 256),
                                             (None, 16, 16, 128),
                                             (None, 32, 32, c)])

    def test_layers_64(self):
        for c in (3, 1):
            output_shapes = self._make_generator_get_output_shapes(64, c)
            self.assertEqual(output_shapes, [(None, 4, 4, 1024),
                                             (None, 8, 8, 512),
                                             (None, 16, 16, 256),
                                             (None, 32, 32, 128),
                                             (None, 64, 64, c)])

    def test_layers_20(self):
        for c in (3, 1):
            output_shapes = self._make_generator_get_output_shapes(20, c)
            self.assertEqual(output_shapes, [(None, 5, 5, 320),
                                             (None, 10, 10, 160),
                                             (None, 20, 20, c)])

    def test_layers_28(self):
        for c in (3, 1):
            output_shapes = self._make_generator_get_output_shapes(28, c)
            self.assertEqual(output_shapes, [(None, 7, 7, 448),
                                             (None, 14, 14, 224),
                                             (None, 28, 28, c)])

    def test_invalid_sizes(self):
        """
        Tests building a generator with an invalid size raises an error
        """
        min_size = list(Generator.get_valid_sizes(limit=20))[0]
        for size in range(min_size, 50):
            if Generator.prev_valid_size(size) == size:
                continue

            with self.assertRaises(GeneratorProjectionError):
                Generator.build_model(out_size=size, out_channels=1)

            with self.assertRaises(GeneratorProjectionError):
                Generator.build_model(out_size=size, out_channels=3)

    @staticmethod
    def _make_generator_get_output_shapes(out_size: int,
                                          out_channels: int) -> list:
        """
        Makes a generator and returns a list of its layers' output shapes
        :param out_size: output size
        :param out_channels: output channels
        :return: a list of tuples for each layer's output shape
        """
        return [layer.output_shape for layer in
                Generator.build_model(out_size=out_size,
                                      out_channels=out_channels).layers]
