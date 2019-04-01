import atexit
import multiprocessing
import os
import random
import time
from datetime import datetime
from glob import glob

import imageio
import numpy as np
import tensorflow as T
from PIL import Image
from tensorflow import keras as K
from tensorflow.contrib import summary as S
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tqdm import tqdm

from src import util
from src.models import Discriminator, Generator

T.enable_eager_execution()
T.logging.set_verbosity(T.logging.ERROR)


class DCGAN:
    # Prefix for model storage
    _MODEL_PREFIX: str = ''

    # Prefix for checkpoint storage
    _CHECKPOINT_PREFIX: str = 'checkpoints'

    # Prefix for log storage
    _LOG_PREFIX: str = 'logs'

    def __init__(self,
                 data_dir: str,
                 out_dir: str,
                 size: int = 32,
                 channels: int = 1,
                 z_dim: int = 100,
                 d_learn_rate: float = 1e-4,
                 g_learn_rate: float = 1e-4,
                 label_smoothing: int = 0.1,
                 name: str = None) -> 'DCGAN':
        """
        Initializes a new DCGAN instance.
        :param data_dir: Directory of the input training data.
        :param out_dir: Directory root to store all output.
        :param size: Size to interpret input and generate output images.
        :param channels: Channels to interpret input images and generate output
                         images at, 1=L and 3=RGB.
        :param z_dim: Dimensionality of latent generator input samples.
        :param d_learn_rate: Discriminator learning rate.
        :param g_learn_rate: Generator learning rate
        :param label_smoothing: Amount to smooth real image labels by
        :param name: A name to identify the training run by that will serve as
                     a sub directory under `out_dir`.
        """
        # Check for valid data directory...
        if not data_dir or not os.path.exists(data_dir):
            raise NotADirectoryError()

        # Save parameters
        self._data_dir = data_dir
        self._out_dir = out_dir
        self._size = size
        self._channels = channels
        self._z_dim = z_dim
        self._d_learn_rate = d_learn_rate
        self._g_learn_rate = g_learn_rate
        self._label_smoothing = label_smoothing
        self._name = name
        self._global_step: T.Variable = None
        self._log_dir: str = None

        # Build discriminator and generator models
        self._generator = self._make_generator()
        self._discriminator = self._make_discriminator()

        # Define discriminator and generator Adam optimizers
        self._discriminator_optimizer = T.train.AdamOptimizer(d_learn_rate)
        self._generator_optimizer = T.train.AdamOptimizer(g_learn_rate)

    def train(self,
              epochs: int = 1000,
              batch_size: int = 32,
              epochs_to_save: int = 0,
              start_tensorboard: bool = False,
              gif: bool = False) -> None:
        """
        Train the DCGAN
        :param epochs: Number of epochs to train for.
        :param batch_size: Size of training batches.
        :param epochs_to_save: Save progress every `epochs_to_save` epochs. If
                               no value is supplied this will be calculated
                               automatically based on data set size. This also
                               controls how often progress GIFs are saved.
        :param start_tensorboard: True to automatically start TensorBoard
        :param gif: True to generate a training progress gif
        """
        # Initialize prerequisites of training
        dataset, epochs_to_save, epoch_kwargs, batch_kwargs = \
            self._init_training(batch_size=batch_size,
                                epochs_to_save=epochs_to_save,
                                start_tensorboard=start_tensorboard)

        for epoch in tqdm(range(epochs), **epoch_kwargs):
            for images in tqdm(dataset, **batch_kwargs):
                # Run the training step on the images
                self._training_step(images, batch_size)

            if (epoch + 1) % epochs_to_save == 0:
                # Save checkpoint and complete models
                self._checkpoint_manager.save()
                self._discriminator.save(self._discriminator_file)
                self._generator.save(self._generator_file)
                self._make_training_gif() if gif else None

            # Write sample images to TensorBoard logs and create training gif
            self._write_tensorboard_images()

    @classmethod
    def generate(cls,
                 model: K.Model = None,
                 model_path: str = None,
                 num_images: int = 1,
                 samples: T.Tensor = None) -> [Image.Image]:
        """
        Generates image(s) from a trained generator model.
        :param model: The generator model
        :param model_path: Path to h5 file containing the trained model
        :param num_images: Number of images to generated
        :param samples: Provide the samples used as input to the generator
        :return: A list of generated images
        """
        # Load the model from file if path was specified...
        if model_path is not None:
            generator = K.models.load_model(model_path)
        elif model is None:
            raise ValueError('Must provide either a model or path')
        else:
            generator = model

        # Determine z_dim and output channels
        z_dim = generator.input_shape[1]
        channels = generator.output_shape[3]

        # Generate samples if none were provided
        if samples is None:
            latent_samples = cls._latent_samples(num_images, z_dim)
        else:
            latent_samples = samples

        # Generate batch of images as a 4d numpy array
        #   (i, width, height, channels)
        np_images = generator(latent_samples, training=False).numpy()

        # If only one channel, remove the last dimension
        if channels == 1:
            np_images = np_images[:, :, :, 0]

        images = []
        for image_arr in np_images:
            # Normalize [-1, 1] to [0, 255]
            image_norm = np.array(image_arr * 127.5 + 127.5, dtype='uint8')

            # Create a PIL image
            image = Image.fromarray(image_norm)
            if channels == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')

            images.append(image)

        return images

    def _training_step(self, images: T.data.Dataset, batch_size: int) -> None:
        """
        Perform one individual training batch on a batch of images
        :param images: The images to train on
        :param batch_size: The size of the batch (since images is an iterator)
        """
        # Increment global step
        self._global_step.assign_add(1)

        # Generate new random samples
        latent_samples = self._latent_samples(batch_size, self._z_dim)

        g_tape = T.GradientTape()
        d_tape = T.GradientTape()

        with g_tape, d_tape:
            # Train generator by generating fake images with latent samples.
            fake_images = self._generator(latent_samples, training=True)

            # Train the discriminator on real and fake images.
            d_real = self._discriminator(images, training=True)
            d_fake = self._discriminator(fake_images, training=True)

            # Calculate losses
            g_loss = self._g_loss(d_fake)
            d_loss, d_loss_real, d_loss_fake = self._d_loss(d_real, d_fake)

            # Write losses to TensorBoard logs
            self._write_tensorboard_losses(d_loss,
                                           d_loss_real,
                                           d_loss_fake,
                                           g_loss)

        # Apply generator gradients
        g_grad = g_tape.gradient(g_loss, self._generator.variables)
        grad_var_g = zip(g_grad, self._generator.variables)
        self._generator_optimizer.apply_gradients(grad_var_g)

        # Apply discriminator gradients
        d_grad = d_tape.gradient(d_loss, self._discriminator.variables)
        grad_var_d = zip(d_grad, self._discriminator.variables)
        self._discriminator_optimizer.apply_gradients(grad_var_d)

    def _init_training(self,
                       batch_size: int,
                       epochs_to_save: int,
                       start_tensorboard: bool) -> (T.data.Dataset,
                                                    int,
                                                    dict,
                                                    dict):
        """
        Initialize training
        :param batch_size: Size of training batches
        :param epochs_to_save: Epoch threshold for progress saving
        :param start_tensorboard: True to start TensorBoard automatically
        :return:
        """
        if start_tensorboard:
            self._start_tensorboard()

        # Load data set and get buffer size
        dataset, buffer_size = self._load_dataset()

        # Cache, shuffle, batch and prefetch the dataset
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size)

        # If not provided, calculate epochs_to_save based on dataset size
        if epochs_to_save < 1:
            threshold = 1e4
            epochs_to_save = int(threshold // buffer_size)

            if epochs_to_save < 1:
                epochs_to_save = 1

            if epochs_to_save == 1:
                print('saving progress every epoch based on dataset size')
            else:
                print('saving progress every {:d} epochs based on dataset size'
                      ''.format(epochs_to_save))

        # Figure out how many sample images are needed based on image size
        num_sample_images = self._calc_num_sample_images()

        # Generate samples that will not change (for TensorBoard images / GIF)
        constant_samples = self._latent_samples(num_sample_images, self._z_dim)
        constant_samples = T.Variable(constant_samples)
        self._constant_samples = constant_samples

        # Calculate batch size (round to include partial batches)
        batch_total = round(buffer_size / batch_size)

        # tqdm progress bar arguments
        epoch_kwargs = {'unit': 'epoch', 'position': 0, 'leave': False}
        batch_kwargs = {'total': batch_total,
                        'unit': 'batch',
                        'position': 1,
                        'leave': False}

        # Initialize logging and restore latest checkpoint (if exists)
        self._init_logging()
        self._restore_checkpoint()

        # Log output directories
        tqdm.write('output root: {}'.format(self._out_fpath()))
        tqdm.write('logging to:  {}'.format(self.log_dir))

        # Compile into callable TensorFlow graph for performance. For more info
        # see: http://tensorflow.org/api_docs/python/tf/contrib/eager/defun
        self._training_step = T.contrib.eager.defun(self._training_step)

        return dataset, epochs_to_save, epoch_kwargs, batch_kwargs

    def _init_logging(self) -> None:
        """
        Initialize TensorBoard Logging
        """
        # Create summary writer and set as default
        self._summary_writer = S.create_file_writer(self.log_dir)
        self._summary_writer.set_as_default()
        self._global_step = T.train.get_or_create_global_step()

    def _restore_checkpoint(self) -> None:
        """
        Initialize checkpoint features, and restore if one exists.
        """
        # Objects to be saved in the checkpoints
        checkpoint_objects = {
            'generator': self._generator,
            'generator_optimizer': self._generator_optimizer,
            'discriminator': self._discriminator,
            'discriminator_optimizer': self._discriminator_optimizer,
            'constant_samples': self._constant_samples
        }

        # Create the checkpoint and checkpoint manager
        checkpoint = T.train.Checkpoint(**checkpoint_objects)
        self._checkpoint_manager = \
            CheckpointManager(checkpoint=checkpoint,
                              directory=self._checkpoint_dir,
                              max_to_keep=5)

        # Load the latest checkpoint if it exists
        last_checkpoint = self._checkpoint_manager.latest_checkpoint
        status = checkpoint.restore(last_checkpoint)

        try:
            # Check if the checkpoint was loaded...
            status.assert_existing_objects_matched()
            tqdm.write('Loaded checkpoint: {}'.format(last_checkpoint))
        except AssertionError as e:
            # Checkpoint was not loaded...
            self._log_params()

    def _log_params(self):
        with open(self._out_fpath('summary.txt'), 'w') as f:
            data_dir = os.path.abspath(self._data_dir)
            out_dir = os.path.abspath(self._out_dir)

            f.write('TensorFlow:      {}\n'.format(T.__version__))
            f.write('data_dir:        {}\n'.format(data_dir))
            f.write('out_dir:         {}\n'.format(out_dir))
            f.write('size:            {}\n'.format(self._size))
            f.write('channels:        {}\n'.format(self._channels))
            f.write('z_dim:           {}\n'.format(self._z_dim))
            f.write('d_learn_rate:    {}\n'.format(self._d_learn_rate))
            f.write('g_learn_rate:    {}\n'.format(self._g_learn_rate))
            f.write('label_smoothing: {}\n'.format(self._label_smoothing))
            f.write('name:            {}\n'.format(self._name))

            f.write('\n\nDISCRIMINATOR\n')
            f.write('input shape: {}\n'.format(self._discriminator.input_shape))
            self._discriminator.summary(line_length=80,
                                        print_fn=lambda l: f.write(l + '\n'))

            f.write('\n\nGENERATOR\n')
            f.write('input shape: {}\n'.format(self._generator.input_shape))
            self._generator.summary(line_length=80,
                                    print_fn=lambda l: f.write(l + '\n'))

    def _preprocess_image(self, fpath: T.Tensor) -> T.Tensor:
        """
        Preprocess an image
        :param fpath: Path of the image
        :return:
        """
        # Read the file, resize and convert to appropriate channels
        image = T.read_file(fpath)
        image = T.image.decode_jpeg(image, self._channels)
        image = T.image.resize_images(image, self._dimensions)

        # Normalize to the range [-1, 1]
        return (image - 127.5) / 127.5

    def _load_dataset(self, data_limit: int = 0) -> (T.data.Dataset, int):
        """
        Load the dataset
        :param data_limit: Optional limit to number of items in the dataset
        :return: Dataset of images for this instance
        """
        cpus = multiprocessing.cpu_count()
        data_root = os.path.join(self._data_dir, '**/*.jpg')
        image_paths = list(glob(data_root, recursive=True))
        random.shuffle(image_paths)  # Make sure they are randomized

        # Limit the data if relevant
        if data_limit > 0:
            image_paths = image_paths[:data_limit]

        # Pre-process the images
        path_ds = T.data.Dataset.from_tensor_slices(image_paths)
        image_ds = path_ds.map(self._preprocess_image, num_parallel_calls=cpus)

        # Return the dataset and it's length (since it's an iterable)
        return image_ds, len(image_paths)

    def _make_training_gif(self, fps: int = 6) -> None:
        """
        Generate a training progress gif
        :param fps: Number of frames per second in the gif
        """
        # Generate some file paths
        timestamp = int(time.time())
        gif_dir = 'gif_frames'
        frame_file = self._out_fpath(gif_dir, '{}.jpg'.format(timestamp))
        frame_dir = os.path.dirname(frame_file)
        frame_jpegs = os.path.join(frame_dir, '*.jpg')
        gif_file = self._out_fpath('training.gif')

        # Generate *constant* sample images and tile all of them
        images = self.generate(self._generator, samples=self._constant_samples)
        tiled_composite = util.generate_tile(images)
        tiled_composite.save(frame_file)

        # Load all the images and make a gif
        images = [imageio.imread(jpg) for jpg in glob(frame_jpegs)]
        imageio.mimsave(gif_file, images, format='GIF', duration=1 / fps)

    def _write_tensorboard_images(self, max_images: int = 3) -> None:
        """
        Write sample images to TensorBoard logs
        :param max_images: Maximum number of images per group to write
        """
        # Generate images with the same seed each epoch (to track progress)
        constant_samples = self._constant_samples
        constant_images = self._generator(constant_samples, training=False)

        # Generate images with a random seed each epoch (for variability)
        image_count = constant_samples.shape[0]  # mimic constant image_count
        latent_samples = self._latent_samples(image_count, self._z_dim)
        random_images = self._generator(latent_samples, training=False)

        # Write the images to the logs
        with S.always_record_summaries():
            S.image('constant', constant_images, max_images=max_images)
            S.image('random', random_images, max_images=max_images)

    def _start_tensorboard(self) -> None:
        """
        Start TensorBoard using this instance's log dir as the --logdir arg
        """
        # Register exit handler, and start process silently and save PID file
        atexit.register(self._kill_tensorboard)
        os.system('tensorboard --logdir={0}/.. &> {1} & '
                  'echo $! >{2}'.format(self.log_dir,
                                        self._out_fpath('.tensorboard.log'),
                                        self._tensorboard_pid))

    def _kill_tensorboard(self) -> None:
        """
        Kill the TensorBoard process
        """
        try:
            pid = open(self._tensorboard_pid).read()
            os.system('kill {}'.format(pid))
            os.remove(self._tensorboard_pid)
        except:
            # Most likely the process does not exist, which doesn't matter
            pass

    def _out_fpath(self, *args: str) -> str:
        """
        Get's the path to a specified file relative to this instance's
        configured output location
        :param args: file name or directory names or both
        :return: string representation of the path
        """
        if not self._name:
            # Generate a new name if one was not provided
            self._name = self._generate_name()

        # Get directory path of the file or directory relative to the out dir
        file_or_dir_path = os.path.join(self._out_dir, self._name, *args)
        dpath = os.path.dirname(file_or_dir_path)

        # Make the intermediate dirs if they do not exist
        if not os.path.exists(dpath):
            os.makedirs(dpath)

        # Return the path
        return file_or_dir_path

    def _generate_name(self) -> str:
        """
        :return: A generated unique(ish) name for this instance to group runs
                 by parameters.
        """
        data_dir = os.path.basename(os.path.normpath(self._data_dir))
        data_dir = data_dir.replace(' ', '_')

        channel_name = 'RGB' if self._channels == 3 else 'L'

        return '{data_name}_{size}_{channels}'.format(data_name=data_dir,
                                                      size=self._size,
                                                      channels=channel_name)

    def _make_discriminator(self) -> K.Model:
        """
        :return: Make a discriminator model for this instance
        """
        return Discriminator.build_model(in_size=self._size,
                                         in_channels=self._channels)

    def _make_generator(self) -> K.Model:
        """
        :return: Make a generator model for this instance
        """
        return Generator.build_model(out_size=self._size,
                                     out_channels=self._channels,
                                     z_dim=self._z_dim)

    def _calc_num_sample_images(self, max_tiled_size: int = 320) -> int:
        """
        Calculates the number of sample images needed to make a square tile of
        progress images at a specified width. Adjusts the results if the
        resulting answer is out of a reasonable range.
        :param max_tiled_size: Maximum desired size of the square tile
        :return: Number of required images
        """
        num_sample_images = (max_tiled_size // self._size) ** 2

        # Adjust for outliers
        if num_sample_images < 4:
            num_sample_images = 4
        elif num_sample_images > 400:
            num_sample_images = 400

        return num_sample_images

    def _d_loss(self,
                real_images: T.Tensor,
                fake_images: T.Tensor) -> (T.Tensor, T.Tensor, T.Tensor):
        """
        Calculate discriminator losses
        :param real_images: A batch of real images
        :param fake_images: A batch of fake images
        :return: Calculated losses
        """
        # Alias loss function
        loss_fn = T.losses.sigmoid_cross_entropy
        s = self._label_smoothing

        # Generate labels
        labels_real = T.subtract(T.ones_like(real_images, dtype=T.float32), s)
        labels_fake = T.zeros_like(fake_images)

        # Calculate the losses on real and fake images and the total loss
        loss_real = loss_fn(multi_class_labels=labels_real, logits=real_images)
        loss_fake = loss_fn(multi_class_labels=labels_fake, logits=fake_images)
        loss_total = loss_real + loss_fake

        # Return all of the losses
        return loss_total, loss_real, loss_fake

    @staticmethod
    def _g_loss(fake_images: T.Tensor) -> T.Tensor:
        """
        Calculate generator loss
        :param fake_images: Images generated by the generator
        :return: Calculated loss
        """
        # Generate labels
        labels = T.ones_like(fake_images)

        # Return the calculated loss
        return T.losses.sigmoid_cross_entropy(labels, fake_images)

    @staticmethod
    def _latent_samples(count: int = 1, n: int = 100) -> T.Tensor:
        """
        Generate `count` number of latent samples of size `n`.
        :param count: Number of samples to generate
        :param n: Size of each sample
        :return: A Tensor containing all of the samples
        """
        return T.random_normal((count, n))

    @staticmethod
    def _write_tensorboard_losses(d_loss: T.Tensor,
                                  d_loss_real: T.Tensor,
                                  d_loss_fake: T.Tensor,
                                  g_loss: T.Tensor) -> None:
        """
        Write losses to TensorBoard logs
        :param d_loss: Discriminator total loss
        :param d_loss_real: Discriminator loss on real images
        :param d_loss_fake: Discriminator loss on fake images
        :param g_loss: Generator loss
        """
        with S.always_record_summaries():
            # First family has total losses
            S.scalar('d_loss', d_loss, family='aggregate')
            S.scalar('g_loss', g_loss, family='aggregate')

            # Second family has a breakdown of discriminator losses
            S.scalar('d_loss_fake', d_loss_fake, family='discriminator')
            S.scalar('d_loss_real', d_loss_real, family='discriminator')

    @property
    def _generator_file(self) -> str:
        """
        :return: Path to the hdf5 file of the full generator model along with
        its trained weights
        """
        return self._out_fpath(self._MODEL_PREFIX, 'generator.h5')

    @property
    def _discriminator_file(self) -> str:
        """
        :return: Path to the hdf5 file of the full discriminator model along
        with its trained weights
        """
        return self._out_fpath(self._MODEL_PREFIX, 'discriminator.h5')

    @property
    def _checkpoint_dir(self) -> str:
        """
        :return: Path to the checkpoint directory
        """
        return self._out_fpath(self._CHECKPOINT_PREFIX)

    @property
    def log_dir(self) -> str:
        """
        :return: Path to the TensorBoard log directory
        """
        if self._log_dir is None:
            # Generate a new dir for each call if it has not yet been made, and
            # use the format: YYYYMMDD-HHMMSS
            log_id = datetime.now().strftime('%Y%m%d-%H%M%S')
            self._log_dir = self._out_fpath(self._LOG_PREFIX, log_id)

        return self._log_dir

    @property
    def _tensorboard_pid(self) -> str:
        """
        :return: Path to the Tensorboard PID file
        """
        return self._out_fpath('.tensorboard.pid')

    @property
    def _shape(self) -> (int, int, int):
        """
        :return: Shape of images as a list of (width, height, channels
        """
        return self._size, self._size, self._channels

    @property
    def _dimensions(self) -> (int, int):
        """
        :return: Size of images as a list of (width, height)
        """
        return self._size, self._size
