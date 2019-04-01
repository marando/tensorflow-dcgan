from datetime import datetime
import os
from argparse import Namespace
from src import util


def run(args: Namespace) -> None:
    """
    Runs the generator module of the dcgan CLI tool
    :param args: Parsed command line arguments
    """
    # Only load TensorFlow when needed for speed
    from src.dcgan import DCGAN

    # Generate the images
    images = DCGAN.generate(model_path=args.model, num_images=args.count)

    # Get or create an output directory
    if args.out_dir:
        out_dir = os.path.join(args.out_dir)
    else:
        out_dir = os.path.dirname(args.model)
        out_dir = os.path.join(out_dir, 'images')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # Unique unique_id for the images
    unique_id = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Generate an image tile
    if args.tile:
        tile_path = os.path.join(out_dir, '{}.jpg'.format(unique_id))
        image = util.generate_tile(images)
        image.save(tile_path)

    # Generate separate images (not tiled)
    else:
        # Make a sub folder when there's more than one image
        if len(images) > 1:
            out_dir = os.path.join(out_dir, unique_id)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        for i, image in enumerate(images):
            # Define the file name for the image
            if len(images) == 1:
                # Use the identifier
                filename = '{}.jpg'.format(unique_id)
            else:
                # Use the image number
                filename = '{:03d}.jpg'.format(i)

            # Save the image
            image_path = os.path.join(out_dir, filename)
            image.save(image_path)
