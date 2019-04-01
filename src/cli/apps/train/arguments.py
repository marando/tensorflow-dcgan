from argparse import ArgumentParser

from src.cli.defaults import *


def add_args(parser: ArgumentParser) -> None:
    """
    :param parser: Adds arguments to the training parser
    """
    training = parser.add_argument_group(title='Training Options')

    training.add_argument(
        '-e',
        '--epochs',
        default=EPOCHS,
        type=int,
        metavar='?',
        help='Number of epochs to train for.'
    )

    training.add_argument(
        '--epochs-to-save',
        default=EPOCHS_TO_SAVE,
        type=int,
        metavar='?',
        help='Number of epochs to pass before saving checkpoints, trained '
             'models, and generating GIFs. If no value is provided it will be'
             'calculated automatically based on dataset size.'
    )

    training.add_argument(
        '-b',
        '--batch-size',
        default=BATCH_SIZE,
        type=int,
        metavar='?',
        help='Size of training batches.'
    )

    training.add_argument(
        '-s',
        '--size',
        default=WIDTH,
        type=int,
        metavar='?',
        help='The width and height of images that are input to the '
             'discriminator as well as output from the generator.'
    )

    training.add_argument(
        '-L',
        '--greyscale',
        action='store_true',
        help='Interpret images as greyscale before training and during '
             'generation.'
    )

    data = parser.add_argument_group(title='Data and Output Options')

    data.add_argument(
        '-d',
        '--data-dir',
        required=True,
        type=str,
        metavar='?',
        help='A directory containing the training images. The directory can be '
             'an absolute path or relative to the `{0}` directory in the '
             'project\'s root.'.format(DATA_ROOT)
    )

    data.add_argument(
        '-o',
        '--out-dir',
        default=OUT_DIR,
        type=str,
        metavar='?',
        help='A directory to save all output from the training process. If '
             'none is specified `{0}` is used by default. This directory will '
             'hold logs, checkpoints and saved models.'.format(OUT_DIR)
    )

    data.add_argument(
        '-n',
        '--name',
        type=str,
        metavar='?',
        help='An optional name that can be used to identify the training run '
             'and will serve as the name of a sub folder under the output '
             'directory. If none is supplied a unique one will be generated at'
             'runtime.'
    )

    misc = parser.add_argument_group(title='Miscellaneous Options')
    misc.add_argument(
        '--no-gif',
        action='store_true',
        help='Disable generation of a training progress GIF each time '
             'checkpoints are saved.'
    )

    misc.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Do not automatically start TensorBoard.'
    )
