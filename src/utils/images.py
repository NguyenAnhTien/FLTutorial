from PIL import Image

def imread(
        image_file: str
    ) -> Image:
    """
    """
    image = Image.open(image_file)
    return image
