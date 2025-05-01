from PIL import Image
from skimage.filters import frangi
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import exposure
from skimage.measure import label, regionprops
from skimage import morphology


def classifyVessel(img):
    # min_val = np.min(img)
    # max_val = np.max(img)
    
    # # Ustawienie progu na połowę wartości między min a max
    # threshold = (min_val + max_val) / 2.0
    threshold = 0.85
    binaryMask = np.zeros_like(img,dtype=np.uint8)
    binaryMask[img > threshold] = 255

    return binaryMask

def getGreenArray(img):
    img_np = np.array(img)
    
    green_channel = img_np[:, :, 1]
    return green_channel
    

def main():
    images = []  # Lista do przechowywania obrazów

    for i in range(1, 16):  # Zakres od 1 do 15
        filename = f'images/{i:02d}_h.jpg'  # Formatowanie np. 01_h.jpg, 02_h.jpg, ...
        img = Image.open(filename)
        images.append(img)
        
        

    img = images[0]
    size = img.size
    print(size)
    

    ######### Wstępne przetworzenie obrazu #########
    imgGreen = getGreenArray(img)
    imgGreenNorm = exposure.equalize_hist(imgGreen)       # poprawa kontrastu
    imgGreenBlur = gaussian(imgGreenNorm, sigma=1)        # redukcja szumu
    # plt.imshow(imgGreen,cmap='gray',aspect='auto')
    # plt.show()
    ######### Wlasciwe przetworzenie obrazu #########
    frangiGreen = frangi(imgGreenBlur)


    ######### Końcowe przetwarzanie obrazu #########
    frangiGreenEqualized = exposure.equalize_hist(frangiGreen)
    print(f"Min value: {np.min(frangiGreenEqualized)}, Max value: {np.max(frangiGreenEqualized)}")
    binaryMask = classifyVessel(frangiGreenEqualized)
    # Wyświetlanie rozciągniętego obrazu


    groundTruths = []  # Lista do przechowywania ground truthów

    for i in range(1, 16):
        filename = f'groundTruth/{i:02d}_h.tif'
        img = Image.open(filename)
        groundTruths.append(img)
    groundTruthNp = np.array(groundTruths[0])


    fieldOfView = Image.open('fieldOfView/01_h_mask.tif')
    grayFieldOfView = fieldOfView.convert('L')
    fieldOfView = np.array(grayFieldOfView)

    height = groundTruthNp.shape[0]
    width = groundTruthNp.shape[1]

    countTrueZeros = 0
    countTrueOnes = 0
    countHitZeros = 0
    countHitOnes = 0
    totalNumberOfPixels = 0
    for i in range(height):
        for j in range(width):
            if(fieldOfView[i,j]==255):
                totalNumberOfPixels += 1
                if(groundTruthNp[i,j] == 255):
                    countTrueOnes += 1
                    if(binaryMask[i,j] == 255):
                        countHitOnes += 1

                if(groundTruthNp[i,j] == 0):
                    countTrueZeros += 1
                    if(binaryMask[i,j] == 0):
                        countHitZeros += 1    
                

    print(f"Total number of pixels {totalNumberOfPixels}")
    print("True zeros - > ",countTrueZeros)
    print("True ones - > ",countTrueOnes)
    print("Hit zeros - > ",countHitZeros)
    print("Hit ones - > ",countHitOnes)
    print(f"Percentile zeros - > {countHitZeros/countTrueZeros*100}%")
    print(f"Percentile ones - > {countHitOnes/countTrueOnes*100}%")




if __name__ == "__main__":
    main()