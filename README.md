# TensorFlow DCGAN
*A TensorFlow deep convoluted generative adversarial network implementation*

![](https://user-images.githubusercontent.com/4701701/55361420-29d9fe00-54a5-11e9-899c-6f72e6532c77.gif)  

## Initial Setup

1. Clone and enter the repository:
    ```bash
    git clone git@github.com:marando/tensorflow-dcgan.git
    cd tensorflow-dcgan
    ```

2. Install a Python 3 virtual environment, upgrade PIP, and install project 
   dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements
    ```

    >***Note:** `build-venv.sh` contains all the above commands. If you source 
    the script you won't have to manually activate the virtual environment 
    afterwards. Otherwise, run `source bin/activate` to activate the virtual 
    environment.*
3. At this point if you have an optimized TensorFlow wheel you could install it
   for better performance:
    ```bash
    pip install /path/to/tensorflow.whl
    ```
       

## Datasets

### Download a Dataset

Common datasets can be downlaoded to the `data` directory. To download a 
dataset, use the following command:
```
./dcgan data --download <dataset name>
```
>***Note:** Available datasets are: `cifar10`, `cifar100`, `fashion_mnist`, and 
`mnist`.*


## Training the DCGAN

For training, `--data-dir` is the only required argument and can be a relative 
or absolute path to where your dataset resides. 
```
./dcgan train --data-dir data/mnist
```
>***Note:** The directory you provide will be scanned recursively for all JPEG 
images.*

### Training Output
By default all output is stored in the `out` directory at the project root in a
sub directory that is auto generated based on the dataset name and image 
parameters. You can override the output root with `--out-dir` and change the 
sub directory name with `--name`.

### Training Checkpoints
Checkpoints and trained models will be saved at an automatically calculated 
epoch threshold based on dataset size. This is to prevent small datasets from 
saving progress at an unreasonable frequency. However, if you want to customize
the threshold, you can pass the number of epochs that must pass before saving 
progress with `--epochs-to-save`.

### Learning Rate
The learning rate for the discriminator and generator can be changed by using
the `--d-learn-rate` and `--g-learn-rate` parameters, otherwise 0.0001 is the 
default value for both.

### Training Progress GIFs
An animated GIF image of the training progress will be saved in the output 
directory each time a checkpoint is saved. Individual frames are stored in a
folder called `gif_frames`. You can disable GIF generation by passing the 
`--no-gif` argument.
    
### TensorBoard
Updates are regularly written to the TensorBoard log files in the `log` 
directory. A TensorBoard server is started when training begins that 
automatically loads the correct log root. Go to 
[http://localhost:6006](http://localhost:6006) to view progress.

>***Note:** If you want to disable TensorBoard from automatically starting, use
`--no-tensorboard`. Logs will still be saved but the server will not be started
automatically. This is useful if you wish to manually run TensorBoard.*

### Trained Models
The weights and architecture for the trained models are stored each time 
progress is saved in HDF5 files (`.h5` extension) in the output directory. To 
generate images, `generator.h5` is required.


## Generating Images
To generate an image, specify the path to the trained `generator.h5` file using
the `--model` argument:
```
./dcgan generate --model ~/dcgan_out/my_session/generator.h5
```

If you want to generate more than one image, use `--count` and specify the 
number of images you wish to generate. 

### Composite Tiles
You can also generate a composite of multiple images by passing the `--tile` 
option. If you use this option the number of images generated will be truncated
to the next lowest perfect square and a square image will be generated. For 
example, the below command will generate a tile of 64 images arranged in an 8×8 
grid:
```
./dcgan generate -m /path/to/generator.h5 --count 64 --tile
```

>***Note:** Generated images are stored in the output root under a folder called
`images` for single images and image tiles. If you generate more than 
one image at once they will be placed in a unique sub directory.*

## Examples

### MNIST

Here is an example of running the trainer on the MNIST dataset. The commands to
replicate are as follows:

```
./dcgan data --download mnist
./dcgan train --data-dir data/mnist --size 28 --greyscale
```

Here is a GIF of the progress from epoch 1 to 46:

![](https://user-images.githubusercontent.com/4701701/55361420-29d9fe00-54a5-11e9-899c-6f72e6532c77.gif) 

After the first epoch:

![](https://user-images.githubusercontent.com/4701701/55361469-3a8a7400-54a5-11e9-95aa-9d1bdd404dad.jpg)

After 10 epochs:

![](https://user-images.githubusercontent.com/4701701/55361475-41b18200-54a5-11e9-87b4-680b124f7d44.jpg)

After 46 epochs:

![](https://user-images.githubusercontent.com/4701701/55361484-48d89000-54a5-11e9-9373-2a00bffc5cee.jpg)
