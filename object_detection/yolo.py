# --- CONFIGURACION DE UBICACION PARA USO DE MODULOS --- 

import sys, wget

# Agrego al sys.path el directorio padre para importar modulos
sys.path.append("..")


# -- IMPORTS ---

from distutils.util import strtobool
import numpy as np
import os, cv2, argparse, time

# Mis modulos
from object_detection.darknetpy import darknet as dknet
from object_detection.tools import getCrops, drawBboxs, filterByRatio
from utils import processInputPath, isEmpty



# --- CONSTANTES ---

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))



# --- CLASES ---

class Darknet():
    def __init__(self, version, conf_threshold=.5, iou_threshold=.45):
        
        # Check and download weights for each model
        self.__downloadWeights(version)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.net = dknet.load_net(self.cfg_file, self.weights_file, 0)
        self.meta = dknet.load_meta(self.classes_file)

    def predict(self, image, class_name="all", filter_by_ratio=False):
        """
        Dado una imagen (o su path) <image>, retorna todos los bounding boxs encontrados 
        para cada <class_name>.
        """

        if isinstance(image,str):
            # Se paso path de imagen
            self.img_orig = cv2.imread(image)
            image = image.encode('utf-8')
            direct_image = False
        
        elif isinstance(image,np.ndarray):
            # Se paso imagen como numpy.ndarray
            self.img_orig = image.copy()
            direct_image = True

        bboxs = dknet.detect(   self.net, 
                                self.meta,
                                image,
                                conf_threshold=self.conf_threshold,
                                iou_threshold=self.iou_threshold, 
                                direct_image=direct_image)

        if not isEmpty(bboxs):

            # Controlo limites y decodifico cada deteccion
            bboxs = self.__decode(bboxs)

            # Elimino todas las clases que no sean igual a class_name
            if class_name != "all":
                bboxs = [box for box in bboxs if box[0] == class_name]

                if filter_by_ratio:
                    if not isEmpty(bboxs):
                        bboxs = filterByRatio(bboxs)

        # [[label, conf, upleft_x, upleft_y, bottomright_x, bootonright_y], ...]
        return bboxs

    def __decode(self, bboxs):
        """
        Input
           bboxs: [[b'label', conf, [x_center, y_center, width, height]], ...]
        
        Output
           bboxs: [[label, conf, topleft_x, topleft_y, bottomright_x, bottomright_y], ...]
        """

        h,w = self.img_orig.shape[:2]

        bboxs = [[  e[0].decode('UTF-8'),
                    e[1],
                    max( int(e[2][0] - (e[2][2] / 2.0)), 0),    #topleft_x
                    max( int(e[2][1] - (e[2][3] / 2.0)), 0),    #topleft_y
                    min( int(e[2][0] + (e[2][2] / 2.0)), w),    #bottomright_x
                    min( int(e[2][1] + (e[2][3] / 2.0)), h)     #bottomright_y
                ]
                for e in bboxs]

        return bboxs
    
    def __downloadWeights(self, version):    
        # Set data file
        classes_file = os.path.join(CURRENT_PATH, "data/coco.data")
        # Set config file
        cfg_file = os.path.join(CURRENT_PATH, "cfg/"+version+".cfg")

        # Set and check weights files
        if version == 'yolov2':
            url = "https://pjreddie.com/media/files/yolov2.weights"
        elif version == 'yolov3':
            url = "https://pjreddie.com/media/files/yolov3.weights"
        elif version == 'yolov3t':
            url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
        
        weights_folder = os.path.join(CURRENT_PATH, "weights/")
        if not os.path.isdir(weights_folder):
            print(f"Creating 'weights' folder...")
            os.makedirs(weights_folder)

        weights_file = os.path.join(weights_folder,version+".weights")
        if not os.path.isfile(weights_file):
            print(f"Downloading '{version}' weights for the only time. Please wait ...")
            wget.download(url, out=weights_file)

        # Codifico string para compatibilidad con lengaje C
        self.cfg_file = cfg_file.encode('utf-8')
        self.classes_file = classes_file.encode('utf-8')
        self.weights_file = weights_file.encode('utf-8')


# --- FUNCIONES AUXILIARES ---

def getArguments():
    ap = argparse.ArgumentParser(description='YOLO')
    ap.add_argument("-i","--image_path", required=True, type=str , help="Path de imagen o carpeta")
    ap.add_argument("-cname", "--class_name", type=str, default='bus', help="Nombre de la clase de interes. (Default = 'bus')")
    ap.add_argument("-conf", "--conf_threshold", default=0.5, type=float, help="Umbral de confidencia. (Default = 0.5)")
    ap.add_argument("-iou", "--iou_threshold", default=0.45, type=float, help="Indice de iou. (Default: 0.45)")
    ap.add_argument("-save", "--save", default="False", type=str, help="Guardar salida en carpeta buses_crops. (Default: False)")
    ap.add_argument("-show", "--show", default="False", type=str, help="Muestro cada deteccion individual. (Default: False)")
    return vars(ap.parse_args())    



# --- FUNCION PRINCIPAL MAIN ---

if __name__ == '__main__':

    args = getArguments()

    # Instancio la clase Darknet para iniciar
    net = Darknet('yolo2', args["conf_threshold"], args["iou_threshold"])
    
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