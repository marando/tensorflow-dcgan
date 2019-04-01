from argparse import ArgumentParser, Namespace

from . import apps


def get_args() -> (Namespace, ArgumentParser):
    """
    :return: Command line arguments
    """
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='command')

    # Add sub parsers for train and generate
    data = sub_parsers.add_parser('data', description=apps.data.desc)
    train = sub_parsers.add_parser('train', description=apps.train.desc)
    gen = sub_parsers.add_parser('generate', description=apps.generate.desc)

    # Add args to the subparsers
    apps.data.add_args(data)
    apps.train.add_args(train)
    apps.generate.add_args(gen)

    return parser.parse_args(), parser
