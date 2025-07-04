from PIL import Image
from skimage.filters import frangi
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import exposure
import cv2


def classifyVessel(img):
    min_val = np.min(img)
    max_val = np.max(img)
    test=0
    #test
    # # Ustawienie progu na połowę wartości między min a max
    # threshold = (min_val + max_val) / 2.0 daje avg dla zeros 89.45% i ones 47.91% (threshold == +- 0.85)
    # threshold = 0.73 #Daje avg dla zeros 74.27% i ones 66.1%
    #threshold = 0.72 #Daje avg dla zeros 72.54% i ones 68%
    threshold = min_val #Daje avg dla zeros 71.47% i ones 69%
    # if (threshold < min_val):
    #     threshold = min_val + 0.01

    binaryMask = np.zeros_like(img,dtype=np.uint8)
    binaryMask[img > threshold] = 255

    return binaryMask

def getGreenArray(img):
    img_np = np.array(img)
    
    green_channel = img_np[:, :, 1]
    return green_channel

def westepnePrzetworzenie(img, show=True):
    ######### Wstępne przetworzenie obrazu #########
    imgGreen = getGreenArray(img)

    # Upewnij się, że dane są typu uint8, bo tego wymaga CLAHE
    if imgGreen.max() <= 1.0:
        imgGreen_uint8 = (imgGreen * 255).astype(np.uint8)
    else:
        imgGreen_uint8 = imgGreen.astype(np.uint8)

    # Zastosowanie CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgGreenCLAHE = clahe.apply(imgGreen_uint8)

    # print("Przykładowy piksel przed CLAHE:", imgGreen[0][25])
    # print("Po CLAHE:", imgGreenCLAHE[0][25])

    # Normalizacja do zakresu [0,1] przed filtrem Gaussa
    imgGreenCLAHE = imgGreenCLAHE.astype(np.float32) / 255.0

    # Redukcja szumu
    imgGreenBlur = gaussian(imgGreenCLAHE, sigma=1)

    if show:
        plt.imshow(imgGreenBlur, cmap='gray', aspect='auto')
        plt.title("Obraz po wstępnym przetworzeniu (CLAHE)")
        plt.axis('off')
        plt.show()

    return imgGreenBlur

def compute(n=5):#powinno generowac n obrazow przetworzonych
    images = [] # Lista do przechowywania obrazów
    binaryMasks = [] #Lista do przechowywania obrazów przetworzonych
    for i in range(1, n+1):  
        filename = f'images/{i:02d}_h.jpg'  # Formatowanie np. 01_h.jpg, 02_h.jpg, ...
        img = Image.open(filename)
        images.append(img)
    
        # img = images[i-1]
        # size = img.size

        imgGreenBlur = westepnePrzetworzenie(img)

        ######### Wlasciwe przetworzenie obrazu #########
        frangiGreen = frangi(imgGreenBlur)
        plt.imshow(frangiGreen,cmap='gray',aspect='auto')
        plt.title("Obraz po zastosowaniu filtra frangiego")
        plt.show()


        ######### Końcowe przetwarzanie obrazu #########
        frangiGreenEqualized = exposure.equalize_hist(frangiGreen)
        plt.imshow(frangiGreenEqualized,cmap='gray',aspect='auto')
        plt.title("Koncowe przetwarzanie obrazu")
        plt.show()


        print(f"Min value: {np.min(frangiGreenEqualized)}, Max value: {np.max(frangiGreenEqualized)}")
        binaryMask = classifyVessel(frangiGreenEqualized)
        binaryMasks.append(binaryMask)
        # Wyświetlanie obrazu
        plt.imshow(binaryMask,cmap='gray',aspect='auto')
        plt.title("Binarna maska odpowiedzi algorytmu")
        plt.show()

    return binaryMasks



def verifyEffectiveness(binaryMasks):#verify effectiveness potrzebuje otrzymac co najmniej 5 obrazow juz przetworzonych oraz ground truth
    groundTruths = []
    srednieZero = []
    srednieJeden=[]
    for i in range(1,len(binaryMasks)+1):
        binaryMask = binaryMasks[i-1]
        filename = f'groundTruth/{i:02d}_h.tif'
        img = Image.open(filename)
        groundTruths.append(img)
        groundTruthNp = np.array(groundTruths[i-1])


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
        for j in range(height):
            for k in range(width):
                if(fieldOfView[j,k]==255):
                    totalNumberOfPixels += 1
                    if(groundTruthNp[j,k] == 255):
                        countTrueOnes += 1
                        # print(binaryMask[j,k])
                        if(binaryMask[j,k] == 255):
                            countHitOnes += 1

                    if(groundTruthNp[j,k] == 0):
                        countTrueZeros += 1
                        if(binaryMask[j,k] == 0):
                            countHitZeros += 1    
        print(f"Image 'images/{i:02d}_h.jpg'")
        print(f"Total number of pixels {totalNumberOfPixels}")
        print("True zeros - > ",countTrueZeros)
        print("True ones - > ",countTrueOnes)
        print("Hit zeros - > ",countHitZeros)
        print("Hit ones - > ",countHitOnes)
        print(f"Percentile zeros - > {countHitZeros/countTrueZeros*100}%")
        print(f"Percentile ones - > {countHitOnes/countTrueOnes*100}%")
        srednieZero.append(countHitZeros/countTrueZeros*100)
        srednieJeden.append(countHitOnes/countTrueOnes*100)
    print(sum(srednieZero)/len(binaryMasks))
    print(sum(srednieJeden)/len(binaryMasks))

def computeAndVerify(n=5):
    binaryMasks = compute(n)#Default n = 5
    verifyEffectiveness(binaryMasks)

def main():
    computeAndVerify()
  




if __name__ == "__main__":
    main()