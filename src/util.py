from PIL import Image


def generate_tile(images: [Image.Image]) -> Image.Image:
    """
    Takes a list of images and tiles them all into a single square image with
    the closest perfect square number of images
    :param images: List of images to tile as PIL Image object
    :return: Composite of the image
    """
    # Infer width and height from the images
    width = images[0].size[0]
    height = images[0].size[1]

    # Convert to closest perfect square
    count = int(len(images) ** (1 / 2))

    # Generate the images
    result = Image.new("RGB", (width * count, height * count))
    for index, img in enumerate(images):
        x = index // count * width
        y = index % count * height
        w, h = img.size
        result.paste(img, (x, y, x + w, y + h))

    return result
