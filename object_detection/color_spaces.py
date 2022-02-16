# --- CONFIGURACION DE UBICACION PARA USO DE MODULOS --- 

import sys

# Agrego al sys.path el directorio padre para importar modulos
sys.path.append("..")



# --- IMPORTS ---

from distutils.util import strtobool
from itertools import permutations as permut
import os, cv2, argparse, time
import numpy as np

# Mis modulos
from utils import processInputPath, loadImage



# --- CONSTANTES ---

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(CURRENT_PATH,"../images/color_spaces/")

# Mejores id de transformaciones (resultado de procces_dataset.py)

# First
BEST_TRANSFORMATIONS = [8,15,48,49,7,43,9,35,40,28,5,31,32,21,30] # top15
# BEST_TRANSFORMATIONS = [8,15,48,49,7,43,35,9,28,31,40,5,21] # top13
# BEST_TRANSFORMATIONS = [15,48,43,8,7,49,35,28,31,5] # top10
# BEST_TRANSFORMATIONS = [48,43,35,15,8,5,49] # top7
# BEST_TRANSFORMATIONS = [48,35,43,15,5] # top5
# BEST_TRANSFORMATIONS = [48,43,5] # top3


# Second
# BEST_TRANSFORMATIONS = [8,30,43,48,49,33,7,28,45,15,31,1,17,9,21] # top15
# BEST_TRANSFORMATIONS = [30,33,43,48,49,7,45,8,28,31,9,21,1] # top13
# BEST_TRANSFORMATIONS = [30,43,48,33,49,28,45,31,8,7]  # top10
# BEST_TRANSFORMATIONS = [30,33,43,48,28,49,31] # top7
# BEST_TRANSFORMATIONS = [33,48,43,30,28] # top5
# BEST_TRANSFORMATIONS = [48,33,43] #top3

# COLOR_MANAGERS:: (Space_color, num_de_Canal, Invertir, Nombre)
COLOR_MANAGERS = {  0:("RGB",None,False,"rgb"),1:("RGB",None,True,"-rgb"),
                    2:("RGB",0,False,"r-rgb"),3:("RGB",0,True,"-r-rgb"),
                    4:("RGB",1,False,"g-rgb"),5:("RGB",1,True,"-g-rgb"),
                    6:("RGB",2,False,"b-rgb"),7:("RGB",2,True,"-b-rgb"),

                    8:(cv2.COLOR_RGB2HLS,None,False,"hls"),9:(cv2.COLOR_RGB2HLS,None,True,"-hls"),
                    10:(cv2.COLOR_RGB2HLS,0,False,"h-hls"),11:(cv2.COLOR_RGB2HLS,0,True,"-h-hls"),
                    12:(cv2.COLOR_RGB2HLS,1,False,"l-hls"),13:(cv2.COLOR_RGB2HLS,1,True,"-l-hls"),
                    14:(cv2.COLOR_RGB2HLS,2,False,"s-hls"),15:(cv2.COLOR_RGB2HLS,2,True,"-s-hls"),

                    16:(cv2.COLOR_RGB2HSV,None,False,"hsv"),17:(cv2.COLOR_RGB2HSV,None,True,"-hsv"),
                    18:(cv2.COLOR_RGB2HSV,1,False,"s-hsv"),19:(cv2.COLOR_RGB2HSV,1,True,"-s-hsv"),
                    20:(cv2.COLOR_RGB2HSV,2,False,"v-hsv"),21:(cv2.COLOR_RGB2HSV,2,True,"-v-hsv"),

                    22:(cv2.COLOR_RGB2LAB,None,False,"lab"),23:(cv2.COLOR_RGB2LAB,None,True,"-lab"),
                    24:(cv2.COLOR_RGB2LAB,0,False,"l-lab"),25:(cv2.COLOR_RGB2LAB,0,True,"-l-lab"),
                    26:(cv2.COLOR_RGB2LAB,1,False,"a-lab"),27:(cv2.COLOR_RGB2LAB,1,True,"-a-lab"),
                    28:(cv2.COLOR_RGB2LAB,2,False,"b-lab"),29:(cv2.COLOR_RGB2LAB,2,True,"-b-lab"),

                    30:(cv2.COLOR_RGB2YUV,None,False,"yuv"),31:(cv2.COLOR_RGB2YUV,None,True,"-yuv"),
                    32:(cv2.COLOR_RGB2YUV,0,False,"y-yuv"),33:(cv2.COLOR_RGB2YUV,0,True,"-y-yuv"),
                    34:(cv2.COLOR_RGB2YUV,1,False,"u-yuv"),35:(cv2.COLOR_RGB2YUV,1,True,"-u-yuv"),
                    36:(cv2.COLOR_RGB2YUV,2,False,"v-yuv"),37:(cv2.COLOR_RGB2YUV,2,True,"-v-yuv"),

                    38:(cv2.COLOR_RGB2YCrCb,None,False,"YCrCb"),39:(cv2.COLOR_RGB2YCrCb,None,True,"-YCrCb"),
                    40:(cv2.COLOR_RGB2YCrCb,1,False,"Cr-YCrCb"),41:(cv2.COLOR_RGB2YCrCb,1,True,"-Cr-YCrCb"),
                    42:(cv2.COLOR_RGB2YCrCb,2,False,"Cb-YCrCb"),43:(cv2.COLOR_RGB2YCrCb,2,True,"-Cb-YCrCb"),
                    
                    44:(cv2.COLOR_RGB2Luv,None,False,"luv"),45:(cv2.COLOR_RGB2Luv,None,True,"-luv"),
                    46:(cv2.COLOR_RGB2Luv,1,False,"u-luv"),47:(cv2.COLOR_RGB2Luv,1,True,"-u-luv"),
                    48:(cv2.COLOR_RGB2Luv,2,False,"v-luv"),49:(cv2.COLOR_RGB2Luv,2,True,"-v-luv"),

                    50:("EDGES",None,False,"edges"),51:("EDGES",None,True,"-edges")
                }



# --- FUNCIONES PARA MANEJO DE NFORMACION DE TRANSFORMACIONES ---

def getTransfName(id):
    return COLOR_MANAGERS[id][3]


def getNumOfTransf():
    return len(COLOR_MANAGERS)



# --- FUNCIONES GENERADORAS DE TRANSFORMACIONES DE IMAGENES ---

def invertImg(img):
    # Nota: Esto lo puedo generalizar de la siguiente forma debido a que todas las conversiones 
    #       de espacios de colores que realiza opencv, mapea a un rango de valores de 0-255.
    return (255-img)


def imgToEdges(img):
    """
    Deteccion de bordes horizontales y verticales en <img>
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel_x = np.array( (  [-0.5, 0, 0.5],
                            [-0.5, 0, 0.5],
                            [-0.5, 0, 0.5],
                            ), dtype="float")

    kernel_y = np.array( (  [-0.5,-0.5,-0.5],
                            [0, 0 ,0],
                            [0.5,0.5,0.5]), dtype="float")
    
    grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

    gx = cv2.convertScaleAbs(grad_x)
    gy = cv2.convertScaleAbs(grad_y)

    magnitude = cv2.cvtColor((gx | gy), cv2.COLOR_GRAY2BGR)
    
    return magnitude


def imgToTransf(img, id_transf, one_channel=False):
    """
    Input:
        img = Imagen a transformar
        id_transf = id de transformacion de interes

    Output:
        new_img: Nueva imagen producto de aplicarle la transformacion numero <id_transf> a <img>

    NOTA:   Si flag <one_channel> == True, <new_img> sera solo una imagen de un canal (ie. una dim)
            Si flag <one_channel> == False, <new_img> repetira la trasformacion en los 3 canales. (ie. tres dim).
    """

    color_space = COLOR_MANAGERS[id_transf][0]
    ch = COLOR_MANAGERS[id_transf][1]
    invert = COLOR_MANAGERS[id_transf][2]

    new_img = img.copy()

    if color_space != "EDGES" and color_space != "RGB":
        if ch != None:
                new_img = cv2.cvtColor(new_img,color_space)[:,:,ch]
                if not one_channel:
                    new_img = cv2.cvtColor(new_img,cv2.COLOR_GRAY2RGB)
        else:
            new_img = cv2.cvtColor(new_img,color_space)
    
    elif color_space == "EDGES":
        new_img = imgToEdges(new_img)

    elif color_space == "RGB":
        if ch != None:
            new_img = new_img[:,:,ch]
            if not one_channel:
                new_img = cv2.cvtColor(new_img,cv2.COLOR_GRAY2RGB)


    if invert:
        new_img = invertImg(new_img)

    return new_img



# --- FUNCION PARA MANEJO Y GENERACION DE FUSIONES DE CANALES ---

def getFuseAndPermut3Channels(img_list, list_of_ids, show=False, save=False):
    """
    Esta funcion devuelve todas las imagenes formada por todas las permutaciones posibles entre 
    los tres canales de <list_of_ids>.
    
    Input:
        list_of_ids: Lista de los 3 id's de los canales (ie. transformaciones) deseadas.
    """

    assert(type(img_list) == list)
    
    transf_list = []
    
    for img in img_list:
        H, W = img.shape[:2]    

        # Genero lista de permutaciones de canales. (Nota: permut es iterador, por eso debo instanciarlo cada vez.)
        for id_permut, p in enumerate(permut(list_of_ids)):
            permut_name = ""
            new_fuse = np.zeros((H,W,3), dtype=np.uint8)

            for ith, id in enumerate(p):
                ch = imgToTransf(img, id, one_channel=True)
                permut_name += getTransfName(id)+" "
                try:
                    dim = ch.shape[2]
                except:
                    pass
                else:
                    sys.exit("Error al fusionar canales: El id:{} ingresado no pertence a un canal!".format(id))
                
                new_fuse[:,:,ith] = ch

            if show:
                cv2.namedWindow('Fuse',cv2.WINDOW_NORMAL)
                cv2.imshow("Fuse", new_fuse)
                print(permut_name)
                cv2.waitKey(0)

            if save:    
                name = (time.asctime().strip().replace(" ", "").replace(":", ""))
                name = name+"_"+permut_name+".png"
                cv2.imwrite(os.path.join(SAVE_FOLDER,name),new_fuse)
                print(name)
                time.sleep(1)

            transf_list.append((id_permut, new_fuse))

    return transf_list



# --- FUNCION PARA MANEJO Y OBTENCION DE TRANSFORMACIONES ---

def getTransformations(img_list, show=False, save=False):
    
    assert(type(img_list) == list)
    
    transf_list = []
    
    for img in img_list:

        # Lista de ids de seleccion de transformaciones de interes
        transf_ids = BEST_TRANSFORMATIONS

        # Lista de todas las transformaciones
        # transf_ids = list(range(getNumOfTransf()))

        for id in transf_ids:
            t = imgToTransf(img, id, one_channel=False)
            transf_list.append((id,t))

            if show:
                cv2.namedWindow('Trans',cv2.WINDOW_NORMAL)
                cv2.imshow("Trans", t)
                cv2.waitKey(0)

            if save:    
                name = (time.asctime().strip().replace(" ", "").replace(":", ""))
                name = name+"_"+getTransfName(id)+".png"
                cv2.imwrite(os.path.join(SAVE_FOLDER,name),t)
                print(name)
                time.sleep(1)

    return transf_list



# --- FUNCIONES AUXILIARES ---

def getArguments():
    ap = argparse.ArgumentParser(description='Transformaciones de imagen')
    ap.add_argument("-i","--image_path", required=True, type=str, help="Path de imagen o directorio")
    ap.add_argument("-save", "--save", default="False", type=str, help="Guardar salida en directorio <../image/color_spaces>. (Default: False)")
    ap.add_argument("-show", "--show", default="False", type=str, help="Muestro cada transformacion individual. (Default: False)")
    return vars(ap.parse_args())



# --- FUNCION PRINCIPAL MAIN ---

if __name__ == '__main__':

    args = getArguments()


    # Creo lista de imagenes para porcesar
    img_list = [loadImage(img_path) for img_path in processInputPath(args["image_path"], target_list=[".jpg", ".JPG", ".png"])]
    
    # Obtengo transformaciones
    getTransformations(img_list, show=strtobool(args["show"]), save=strtobool(args["save"]))

    # Obtengo todas las fusiones que me otorgan los canales de interes (en este caso [13,26,36])
    #getFuseAndPermut3Channels(img_list, [13,26,36], show = strtobool(args["show"]), save = strtobool(args["save"]))