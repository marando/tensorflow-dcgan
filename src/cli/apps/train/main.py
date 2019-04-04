from argparse import Namespace

from src.models.generator import GeneratorProjectionError, Generator


def run(args: Namespace) -> None:
    """
    Runs the trainer module of the dcgan CLI tool
    :param args: Parsed command line arguments
    """
    # Only load TensorFlow when needed for speed
    from src.dcgan import DCGAN

    try:
        dcgan = DCGAN(out_dir=args.out_dir,
                      data_dir=args.data_dir,
                      name=args.name,
                      size=args.size,
                      channels=1 if args.greyscale else 3,
                      d_learn_rate=args.d_learn_rate,
                      g_learn_rate=args.g_learn_rate)

        dcgan.train(epochs=args.epochs,
                    epochs_to_save=args.epochs_to_save,
                    batch_size=args.batch_size,
                    start_tensorboard=not args.no_tensorboard,
                    gif=not args.no_gif)

    except GeneratorProjectionError:
        print('{} is an invalid size, try {}'
              ''.format(args.size, Generator.prev_valid_size(args.size)))
        exit(1)
