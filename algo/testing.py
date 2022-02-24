import warnings
warnings.filterwarnings("ignore")
import os
import keras
import cv2
import numpy as np


def algo2():
    path=os.path.join(os.getcwd(),'C:/Users/HP/Downloads/Doctors Prescriptions Final Edit/final/Final Edit/Algo','C:/Users/HP/Downloads/Doctors Prescriptions Final Edit/final/Final Edit/Algo/crop')
    med_list = []
    for i in os.listdir(path):
        print(i)

        kernel = np.ones((2,1),np.uint8)

        imag=cv2.imread(os.path.join(path,i),0)

        imag=cv2.resize(imag,(227,227))



        ou = cv2.adaptiveThreshold(imag,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)

        ou = cv2.morphologyEx(ou, cv2.MORPH_OPEN, kernel)
        ou= ou.reshape(227,227)
        cv2.imshow('medicine',ou)
        cv2.waitKey(1)
        ou=255-ou
        ou=np.multiply(imag,ou)
        cv2.imshow('medicine',ou)
        cv2.destroyAllWindows
        cv2.waitKey(100)





        model = keras.models.load_model('modelcnn1_num.h5')

        OUT=model.predict(ou.reshape(-1,227,227,1))

        X=np.argmax(OUT)
        d={0:'CAP.RAZOD',1:'TAB.CEFIXIME',2:'TAB.DICLOFLEX',3:'TAB.NORFLEX 400MG',4:'TAB.ROSTAR',5:'xylometazoline nasal drops'}
        print(d[X])
        med_list.append(d[X])
    return med_list
