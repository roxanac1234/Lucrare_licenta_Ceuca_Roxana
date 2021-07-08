import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from  ExtragereDate import filtrareDupaEtichete, binarizeazaDate, SUBIECTI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from CalculTrasaturiSpectrograme import calculTrasaturiImagini

def clasificare_spectrograme():
    trasaturi = []
    dateBinarizate = []
    for subiect in range(1, 32):
        # Se realizeaza o filtrare a etichetelor functie de marginea inferioara (3.5) si cea superioara (5.5)
        # Se determina experimentele relevante
        valenta, excitare, experimente =filtrareDupaEtichete(subiect)
        #Se calculeaza trasaturile
        for exp in experimente:
            print("Experiment "+ exp.__str__())
            tr, l=calculTrasaturiImagini(subiect, exp)
            trasaturi.append(tr)
            # Se binarizeaza valorile etichetelor
            valoarebinara = binarizeazaDate(subiect, 3.5, 5.5, exp)
            dateBinarizate.append(valoarebinara)
    print("Lungime date binarizate ", len(dateBinarizate))
    print('Lungime trasaturi ', len(trasaturi))
    # Lungimea datelor binarizate este egala cu suma numarului de experimente relevante pentru utilizatori
    lungime  = len(dateBinarizate)
    trasaturi = np.array(trasaturi).reshape(lungime, 84)
    # Iesirile vor fi reprezentate de doua clase:
    # 0-indica o stare negativa
    # 1-indica o stare pozitiva
    dateBinarizate=np.array(dateBinarizate)
    x_antrenare, x_testare, y_antrenare, y_testare = train_test_split(trasaturi, dateBinarizate, test_size=0.20, random_state=0)
    # Se aplica algoritmul SVM cu o functie liniara
    clasificatorSVM = SVC(kernel='linear')
    clasificatorSVM.fit(x_antrenare, y_antrenare)
    y_predictii = clasificatorSVM.predict(x_testare)
    # Se afiseaza metricile
    print("Folosind SVM, s-a obtinut o  acuratete  de ", accuracy_score(y_testare, y_predictii) , ".")
    print("F1 score: ", f1_score(y_testare, y_predictii))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_testare, y_predictii)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("Area under curve: ", roc_auc)
    # # Se aplica Random Forest
    # clasificatorRandomForest = RandomForestClassifier(n_estimators=8, random_state=0,
    #                                     max_features='sqrt', criterion='entropy')
    # # Antrenarea datelor
    # clasificatorRandomForest.fit(x_antrenare, y_antrenare)
    # # Predicțiile rezultate
    # y_predictii = clasificatorRandomForest.predict(x_testare)
    # # Metricile de acuratețe
    # print("Folosind Random Forest s-a obtinut o acuratete de : ", accuracy_score(y_testare, y_predictii), ".")
    # print("F1 score: ", f1_score(y_testare, y_predictii))
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_testare, y_predictii)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # print("Area under curve: ", roc_auc)
if __name__ == '__main__':
    clasificare_spectrograme()
