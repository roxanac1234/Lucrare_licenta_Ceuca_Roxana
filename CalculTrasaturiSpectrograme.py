from ExtragereDate import *
import matplotlib.pyplot as plt
import cv2
import matplotlib
from skimage import feature
import mahotas
import statistics
matplotlib.use('TKAgg', force=True)
# frecventa de esantionare
frecventaEsantionare =128
def calculTrasaturiImagini(subiect, experiment):
    trasaturiImagini = []
    fisierTrasaturiCSV = open('Spectrograme/Trasaturi_spectrograme.csv', 'a')
    fisierTrasaturiCSV.write("\n")

    for canal in CANALE_SELECTATE:
        denumire_imagine ='Spectrograme/Spectrograma_semnal_Subiect_'+str(subiect)+'_experiment_'+str(experiment)+'_canal_'+canal
        # Se genereala spectrograma pentru subiect, experiment, canal
        genereazaSpectrograma(subiect, experiment, canal)
        # Se citeste spectrograma
        img = cv2.imread(denumire_imagine+'.png')
        print('Se calculeaza HSV Histogram')
        # trasaturiHue, trasaturiSat, trasaturiVal = colorHistogramOfHSVImageMax(img)
        trasaturiHue, trasaturiSat, trasaturiVal = colorHistogramOfHSVImageMean(img)
        trasaturiImagini.append(trasaturiHue)
        trasaturiImagini.append(trasaturiSat)
        trasaturiImagini.append(trasaturiVal)
        print('Se calculeaza trasaturi HOG')
        trasaturiHOG = histogramOfOrientedGradients(img)
        trasaturiImagini.append(trasaturiHOG)
        print('Se calculeaza HARALICK texture')
        trasaturiTexturaHaralick = HaralickTexture(img)
        trasaturiImagini.append(trasaturiTexturaHaralick)
        print("Se calculeaza trasaturi detector sift")
        trasaturiSIFT = siftDetector(img)
        trasaturiImagini.append(trasaturiSIFT)
        fisierTrasaturiCSV.write(str(subiect)+"  "+str(experiment)+"  "+canal+"  "+str(trasaturiHue)+"  "+str(trasaturiSat)+"  "+str(trasaturiVal)+"  "+str(trasaturiHOG)+"  "+str(trasaturiTexturaHaralick)+"  "+str(trasaturiSIFT)+"\n")
    fisierTrasaturiCSV.close()
    return trasaturiImagini, len(trasaturiImagini)

# Metoda care genereaza spectrograma semnalului
def genereazaSpectrograma (subiect, experiment, canal):
    print("Se genereaza spectrograma pentru subiect ", subiect, ", experiment ", experiment, "canal ", canal)
    denumire_imagine = 'Spectrograme/Spectrograma_semnal_Subiect_' + str(subiect) + '_experiment_' + str(experiment) + '_canal_' + canal
    # Se extrag datele pentru canalul, subiectul si experimentul dat ca parametru
    dateCanal = extrageDateCanale(subiect = subiect, experiment=experiment)[canal]
    fig = plt.figure()
    ax = plt.subplot(111)
    # Se genereaza spectrograma semnalului
    ax.specgram(dateCanal, Fs=128)
    # Se salveaza ca imagine fara a se retine axele de coordonate
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(denumire_imagine+'.png', bbox_inches=extent)
    # plt.show()
    plt.close(fig)

# Metoda care calculeaza histograma imaginii semnalului in spatiul HSV, luand media pentru trasaturile calculate
def colorHistogramOfHSVImageMean(img):
    trasaturiHue = []
    trasaturiSat = []
    trasaturiVal = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    val_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    for h in hue_hist:
        trasaturiHue.append(h[0])
    for s in sat_hist:
        trasaturiSat.append(s[0])
    for v in val_hist:
        trasaturiVal.append(v[0])
    print("Trasaturi HSV ", statistics.mean(trasaturiHue), " ", statistics.mean(trasaturiSat)," ", statistics.mean(trasaturiVal))
    return statistics.mean(trasaturiHue), statistics.mean(trasaturiSat), statistics.mean(trasaturiVal)

# Metoda care calculeaza histograma imaginii semnalului in spatiul HSV, luand valoarea maxima a trasaturilor calculate
def colorHistogramOfHSVImageMax(img):
    trasaturiHue = []
    trasaturiSat = []
    trasaturiVal = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    val_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    for h in hue_hist:
        trasaturiHue.append(h[0])
    for s in sat_hist:
        trasaturiSat.append(s[0])
    for v in val_hist:
        trasaturiVal.append(v[0])
    return max(trasaturiHue), max(trasaturiSat), max(trasaturiVal)
# Metoda care determina histograma gradientilor orientati (HOG)
def histogramOfOrientedGradients(img):
    img = cv2.resize(img, (128, 256))
    (hog, hog_image) = feature.hog(img, orientations=9,
                                   pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                   block_norm='L2-Hys', visualize=True, transform_sqrt=True)
    print("HOG ", statistics.mean(hog))
    return statistics.mean(hog)


# Metoda care calculeaza valorile pentru textura folosind metoda Haralick
def HaralickTexture(img):

    img_resized = cv2.resize(img, (300, 300))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    image_har = mahotas.features.haralick(img_gray).mean(axis=0)
    print("Haralick texture ", statistics.mean(image_har))
    return statistics.mean(image_har)

# Metoda care calculeaza descriptorii SIFT
def siftDetector(img):
    descriptorValues = []
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(400)
    keypoints_sift = sift.detect(grayImage, None)
    descriptors_sift = sift.compute(grayImage, keypoints_sift)
    img = cv2.drawKeypoints(grayImage, keypoints_sift, img)
    for d in descriptors_sift[1]:
        descriptorValues.append(statistics.mean(d))
    print("SIFT ", statistics.mean(descriptorValues))
    return statistics.mean(descriptorValues)

if __name__ == '__main__':
     # genereazaSpectrograma(subiect=3, experiment=36, canal='F3')
     calculTrasaturiImagini(subiect=2, experiment=24)



