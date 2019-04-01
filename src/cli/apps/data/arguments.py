from argparse import ArgumentParser


def add_args(parser: ArgumentParser) -> None:
    """
    :param parser: Adds arguments to the training parser
    """
    training = parser.add_argument_group(title='Data Options')

    training.add_argument(
        '-d',
        '--download',
        choices=['cifar10', 'cifar100', 'fashion_mnist', 'mnist'],
        help='Download a dataset'
    )
