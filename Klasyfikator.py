from DnoOka import westepnePrzetworzenie
from DnoOka import verifyEffectiveness
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import Parallel, delayed
from time import time
from sklearn.ensemble import RandomForestClassifier

def wytnijFragmentyIEtykiery(img,groundTruth,fieldOfView,windowSize = 5, procent = 0.01):
    height, width = img.shape

    X = []
    y = []

    #Punkty w ktorych moze sie znajdowac lewy gorny róg window i bedzie okej
    valid_coords = []
    for i in range(height - windowSize + 1):
        for j in range(width - windowSize + 1):
            if (fieldOfView[i, j] == 255):
                valid_coords.append((i, j))
    

    #Losowe próbkowanie
    np.random.seed(None)
    sample_size = int(len(valid_coords) * procent)
    sampled_coords = np.random.choice(len(valid_coords), size=sample_size, replace=False)#size = ile koordynatow chcemy, len(valid_coords) -> rozmiar listy z koordynatami 
    #np dla sampled_indices = np.random.choice(4, size=2, replace=False)  # np. [1, 3] dostaniemy 2 losowe pary indeksow z zakresu od 0 do 3 wlacznie
    print(sampled_coords[0])
    print(valid_coords[0])

    for index in sampled_coords:
        i,j = valid_coords[index]
        window = img[i:i+windowSize,j:j+windowSize]
        X.append(window.flatten())
        label = groundTruth[int(i+windowSize/2),int(j+windowSize/2)]
        y.append(label)


    return np.array(X), np.array(y)
    
def process_window(i, j, img, fieldOfView, clf, windowSize):
    if fieldOfView[i, j] == 255:
        window = img[i:i+windowSize, j:j+windowSize]
        feature = window.flatten().reshape(1, -1)
        pred = clf.predict(feature)[0]
        return (i, j, pred)
    return None

def probaPierwsza(groundTruthNp,img,fieldOfView,clf,windowSize):#czas mielenia na i7 -> 281 sekund
    height, width = img.shape
    prediction_mask = np.zeros_like(groundTruthNp, dtype=np.uint8)

    results = Parallel(n_jobs=-1)(delayed(process_window)(i, j, img, fieldOfView, clf, windowSize)
                                for i in range(height - windowSize + 1)
                                for j in range(width - windowSize + 1))
    
        
    for result in results:
        if result:
            i, j, pred = result
            prediction_mask[i, j] = 255 if pred == 255 else 0
    return prediction_mask
def main():
    filename = f'images/{1:02d}_h.jpg'  # Formatowanie np. 01_h.jpg, 02_h.jpg, ...
    img = Image.open(filename)
    print(img.size)
    # img.show()

    img = westepnePrzetworzenie(img)

    groundTruth = Image.open("groundTruth/01_h.tif")
    groundTruthNp = np.array(groundTruth)
    groundTruth.show()

    fieldOfView = Image.open("fieldOfView/01_h_mask.tif")
    grayFieldOfView = fieldOfView.convert('L')
    fieldOfView = np.array(grayFieldOfView)

    start_time = time()
    X,y = wytnijFragmentyIEtykiery(img,groundTruthNp,fieldOfView)#opcjonalnie mozna dac parametr windowSize

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)# mozna dodac parametr random_state=42 wtedy za kazdym razem dane beda dzielone tak samo
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

    clf.fit(X_train,y_train)


    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    prediction_mask = np.zeros_like(groundTruthNp, dtype=np.uint8)

    windowSize = 5

    height, width = img.shape
    
    prediction_mask = probaPierwsza(groundTruthNp,img,fieldOfView,clf,windowSize)
    
    verifyEffectiveness([prediction_mask])


    end_time = time()
    execution_time = end_time - start_time
    print(f"Czas wykonania: {execution_time:.2f} sekundy")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(groundTruthNp, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predykcja modelu")
    plt.imshow(prediction_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
