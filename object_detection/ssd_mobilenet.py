# The model we’ll be using is a Caffe version of the original TensorFlow implementation by Howard et al.
# (https://github.com/Zehaos/MobileNet) and was trained by chuanqi305 (https://github.com/chuanqi305/MobileNet-SSD).

# The MobileNet SSD was first trained on the COCO dataset (Common Objects in Context) and was then fine-tuned 
# on PASCAL VOC reaching 72.7% mAP (mean average precision).


# --- CONFIGURACION DE UBUCACION PARA USO DE MODULOS --- 

import sys

# Agrego al sys.path el directorio padre para importar modulos
sys.path.append("..")



# -- IMPORTS ---

from distutils.util import strtobool
import numpy as np
import argparse, time, cv2, os

# Mis modulos
from object_detection.tools import getCrops, drawBboxs, filterByRatio
from utils import processInputPath, isEmpty



# --- CONSTANTES ---

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

CFG_FILE = os.path.join(CURRENT_PATH,"cfg/ssd-mobilenet.prototxt")
WEIGHTS_FILE = os.path.join(CURRENT_PATH,"weights/ssd-mobilenet.caffemodel")
CLASSES_FILE = os.path.join(CURRENT_PATH,"data/ssd-mobilenet.names")

CROPS_FOLDER = os.path.join(CURRENT_PATH,"../images/buses_crops/")


class SsdMobileNet():
    def __init__(self, cfg_file, weights_file, classes_file, conf_threshold=.5):

        self.cfg_file = cfg_file
        self.weights_file = weights_file
        self.classes_names = self.__readClasses(classes_file)
        self.conf_threshold = conf_threshold

        self.net = cv2.dnn.readNetFromCaffe(self.cfg_file, self.weights_file)

    def predict(self, image, class_name="all", filter_by_ratio=False):
        """
        Dado una imagen (o su path) <image>, retorna todos los bounding boxs encontrados 
        para cada <class_name>.
        """
        
        if isinstance(image,str):
            # Se paso path de imagen
            self.img_orig = cv2.imread(image)
            image = self.img_orig.copy()
        
        elif isinstance(image,np.ndarray):
            # Se paso imagen como numpy.ndarray
            self.img_orig = image.copy()
        
        # Obtengo lista de detecciones
        bboxs = self.forward(image)            

        if not isEmpty(bboxs):

            # Filtro por indice de confidencia, controlo limites y decodifico cada deteccion
            bboxs = self.__decode(bboxs)

            # Filtro todas las clases que no sean <class_name>
            if class_name != "all":
                bboxs = [box for box in bboxs if box[0] == class_name]

                if filter_by_ratio:
                    if not isEmpty(bboxs):
                        bboxs = filterByRatio(bboxs)


        # [[label, conf, upleft_x, upleft_y, bottomright_x, bootonright_y], ...]
        return bboxs

    def forward(self, image):
        
        # Redimensiono imagen a tamaño de entrada de CNN.
        # Nota: 300 fue el parametro usado por el autor para el entrenamiento de la red.

        new_dim = 300
        img_resized = cv2.resize(image, (new_dim, new_dim))

        # Construyo el blob desde la imagen para poder realizar el forwarding en la red
        # Nota: Tanto la escala = 0.007843 como la media = 127.5 para la normalizacion fueron
        #       los parametros usados por el autor para el entrenamiento de la red.

        blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (new_dim, new_dim), 127.5)

        self.net.setInput(blob)
        
        detections = self.net.forward()

        # Elimino primeras dos dimensiones no utilizadas
        bboxs = detections[0,0,:]

        return bboxs

    def __decode(self, bboxs):
        """
        Input
            bboxs: [ [0, id_label, conf, upleft_x, upleft_y, bootonright_x, bootonright_y], ...]
            con  valores proporcionales en rango [0,1].
           
        Output:
            bboxs: [ [label, conf, upleft_x, upleft_y, bottomright_x, bottomright_y ], ...]
            con valores proporcionales a dimension original de imagen de entrada.
        """

        H, W = self.img_orig.shape[:2]

        # Filtro detecciones con indices de confidencia < self.conf_threshold
        mask = (bboxs[:,2] >= self.conf_threshold)
        bboxs = bboxs[mask]

        # Redimensiono coordenadas de cada bounding box y realizo control de limites
        bboxs = [[  self.classes_names[int(e[1])],
                    e[2],
                    max( int(e[3] * W), 0),  # upleft_x
                    max( int(e[4] * H), 0),  # upleft_y
                    min( int(e[5] * W), W),  # bootonright_x
                    min( int(e[6] * H), H)   # bootonright_y
                ]
                for e in bboxs]

        return bboxs

    def __readClasses(self, path):
        """
        Leo archivo de clases y genero una lista
        """
        with open(path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        
        return class_names



# --- FUNCIONES AUXILIARES ---

def getArguments():
    ap = argparse.ArgumentParser(description='SDD MobileNet')
    ap.add_argument("-i","--image_path", required=True, type=str , help="Path de imagen o carpeta")
    ap.add_argument("-cname", "--class_name", type=str, default='bus', help="Nombre de la clase de interes. (Default = 'bus')")
    ap.add_argument("-conf", "--conf_threshold", default=0.5, type=float, help="Umbral de confidencia. (Default = 0.5)")
    ap.add_argument("-save", "--save", default="False", type=str, help="Guardar salida en carpeta buses_crops. (Default: False)")
    ap.add_argument("-show", "--show", default="False", type=str, help="Muestro cada deteccion individual. (Default: False)")
    return vars(ap.parse_args()) 


# --- FUNCION PRINCIPAL MAIN ---

if __name__ == '__main__':
    
    args = getArguments()

    # Instancio la clase Darknet para iniciar
    net = SsdMobileNet(CFG_FILE, WEIGHTS_FILE, CLASSES_FILE, args["conf_threshold"])
    
    for image_path in processInputPath(args["image_path"], target_list=[".jpg",".JPG",".png"]):
        
        # Recupero bounding box para cada class_name detectada
        bboxs = net.predict(image_path, args["class_name"], filter_by_ratio=True)

        if not isEmpty(bboxs):
            print("Detectado! en {}".format(image_path.split("/")[-1]))

            # Imprimo detecciones
            print(*bboxs, sep= "\n")

            # Muestro imagen con detecciones
            drawBboxs(net.img_orig, bboxs)

            # Recupero crops de cada deteccion en caso de que las haya
            crops = getCrops(net.img_orig, bboxs, show=strtobool(args["show"]), save=strtobool(args["save"]))
            
        else:
            print("\nNo se ha detectado en {}".format(image_path.split("/")[-1]))

    if args["show"]:
        cv2.destroyAllWindows()
