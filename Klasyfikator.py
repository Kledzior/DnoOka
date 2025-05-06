from DnoOka import westepnePrzetworzenie
from PIL import Image
from skimage.filters import frangi
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import exposure
from skimage.measure import label, regionprops
from skimage import morphology




def main():
        filename = f'images/{1:02d}_h.jpg'  # Formatowanie np. 01_h.jpg, 02_h.jpg, ...
        img = Image.open(filename)
        img.show()


if __name__ == "__main__":
    main()
