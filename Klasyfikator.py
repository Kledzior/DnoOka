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
from sklearn.ensemble import GradientBoostingClassifier




def main():
        filename = f'images/{1:02d}_h.jpg'  # Formatowanie np. 01_h.jpg, 02_h.jpg, ...
        img = Image.open(filename)
        print(img.size)
        img.show()

        img = westepnePrzetworzenie(img)
        # plt.imshow(img,cmap='gray',aspect='auto')
        # plt.show()


if __name__ == "__main__":
    main()
