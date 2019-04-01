from unittest import TestCase

from src.dcgan import DCGAN
from src.models.generator import GeneratorProjectionError


class TestDCGAN(TestCase):

    def test_fails_size_less_8(self):
        for size in range(8):
            with self.assertRaises(GeneratorProjectionError):
                DCGAN(data_dir='.', out_dir='.', size=size)
