from argparse import ArgumentParser


def add_args(parser: ArgumentParser) -> None:
    """
    :param parser: Adds arguments to the generate parser
    """

    generate = parser.add_argument_group(title='Generator Options')

    generate.add_argument(
        '-c',
        '--count',
        default=1,
        type=int,
        metavar='?',
        help='Number of images to generate'
    )

    generate.add_argument(
        '-t',
        '--tile',
        action='store_true',
        help='Tile images in to the closest square according to count'
    )

    io = parser.add_argument_group(title='Input/Output Options')

    io.add_argument(
        '-m',
        '--model',
        required=True,
        metavar='?',
        help='Path to the trained `generator.h5` file.')

    io.add_argument(
        '-o',
        '--out-dir',
        type=str,
        metavar='?',
        help='A directory to save the generated images in. If nothing is '
             'supplied they will be saved in the same directory as the model.'
    )
