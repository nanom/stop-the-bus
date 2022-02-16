# --- CONFIGURACION DE UBICACION PARA USO DE MODULOS ---

import sys

# Agrego al sys.path el directorio padre para importar modulos
sys.path.append("..")



# -- IMPORTS ---

import numpy as np
import time, os, cv2

# Mis modulos
from object_detection.darknetpy import darknet as dknet
from utils import processInputPath, isEmpty



# --- CONSTANTES ---

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CROPS_FOLDER = os.path.join(CURRENT_PATH,"../images/buses_crops/")



# --- FUNCIONES PARA EL PROCESAMIENTO DE DETECCIONES ---

def filterByRatio(bboxs):
    """
    Elimino bounding box segun relacion de aspecto.
    """

    ratio = .92#.97#0.98
    delta = .12*2#.21#0.19

    condition = lambda w,h: (float(w/h) >= (ratio-delta)) and (float(w/h) <= (ratio+delta))
    bboxs = [[l,c,x1,y1,x2,y2] for l,c,x1,y1,x2,y2 in bboxs if condition((x2-x1),(y2-y1))]

    return bboxs


def originsizeHeuristic(img):
    # Recibo imagen de dimensiones cuadradas
    H,W = img.shape[:2]

    # firts approach
    # mean_dim = 624.33
    # delta_dim = 266.66/2

    # second apprach
    mean_dim = 346.42
    delta_dim = 152.35/2

    max_threshoold = mean_dim + delta_dim
    min_threshoold = mean_dim - delta_dim

    if H < min_threshoold or H > max_threshoold:
        resized_img = cv2.resize(img, (int(mean_dim), int(mean_dim)))
    else:
        resized_img = img

    return resized_img


def getCrops(img_orig, bboxs, top_square=True, first_cuadrant=False, originsize_heursitc=False, show=False, save=False):

    h,w = img_orig.shape[:2]
    crops_list = []

    for box in bboxs:
        x1 = box[2] # upleft_x
        y1 = box[3] # upleft_y
        x2 = box[4] # bottomright_x
        y2 = box[5] # bottomright_y

        width = (x2 - x1)

        if top_square:
            # --- HEURISTICA FIRST_APPROACH ---
            # Recupero el cuadrado superior.
            y2 = np.minimum(y1 + width ,h)

            if first_cuadrant:
                # --- HEURISTICA SECOND_APPROACH ---
                # De cuadrado superior tomo cuadrante segun medias y desviaciones otorgadas por modulo 'first_cuadrant_estimate.py'
                mean_min_x = 0.316
                std_min_x = 0.103
                mean_max_x = 0.481
                std_max_x = 0.092

                # Escalo c/u de los vertices
                x2 = x1 + int(width*mean_max_x + width*2*std_max_x)
                x1 = x1 + int(width*mean_min_x - width*2*std_min_x)
                y2 = y1 + (x2 -  x1)


        img_crop = img_orig[y1:y2, x1:x2, :]

        if originsize_heursitc:
            # --- HEURISTICA ORIGINSIZE VS DETECCIONES POSITIVAS ---
            img_crop = originsizeHeuristic(img_crop)

        if show:
            cv2.namedWindow('Crops',cv2.WINDOW_NORMAL)
            cv2.imshow("Crops", img_crop)
            cv2.waitKey(500) #500

        if save:
            name = (time.asctime().strip().replace(" ", "").replace(":", ""))
            cv2.imwrite(CROPS_FOLDER+name+".png",img_crop)
            print(name+".png")
            time.sleep(1.1)

        # Agrego crop actual a la lista
        crops_list.append(img_crop)

    return crops_list



# --- FUNCION PARA VISUALIZACION DE DETECCIONES SOBRE IMAGEN ORIGINAL ---

def drawBboxs(img_orig, bboxs):

    new_img = img_orig.copy()
    for box in bboxs:
        x1 = box[2]
        y1 = box[3]
        x2 = box[4]
        y2 = box[5]

        # color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
        color = (0,255,0)
        cv2.rectangle(new_img,(x1,y1),(x2,y2), color, 6)

    cv2.namedWindow('Imagen',cv2.WINDOW_NORMAL)
    cv2.imshow("Imagen", new_img)
    cv2.waitKey(1)#50

    # name = (time.asctime().strip().replace(" ", "").replace(":", ""))
    # cv2.imwrite(CROPS_FOLDER+name+".png",new_img)
    # print(name+".png")
    # time.sleep(1.1)
