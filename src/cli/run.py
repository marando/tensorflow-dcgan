from .arguments import get_args
from . import apps


def run() -> None:
    """
    Runs the dcgan CLI tool
    """
    args, parser = get_args()

    # Data module
    if args.command == 'data':
        apps.data.run(args)

    # Training module
    elif args.command == 'train':
        apps.train.run(args)

    # Generate module
    elif args.command == 'generate':
        apps.generate.run(args)

    else:
        # Nothing specified, show help
        parser.print_help()
        exit(1)
