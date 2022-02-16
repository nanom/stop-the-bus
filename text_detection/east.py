# --- CONFIGURACION DE UBUCACION PARA USO DE MODULOS --- 

import sys

# Agrego al sys.path el directorio padre para importar modulos
sys.path.append("..")



# -- IMPORTS ---

import numpy as np
import argparse, time, cv2, os
from distutils.util import strtobool

# Mis modulos
from utils import processInputPath



# --- CONSTANTES ---

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PB_FILE = os.path.join(CURRENT_PATH,"cfg/frozen_east_text_detection.pb")
CROPS_FOLDER = os.path.join(CURRENT_PATH,"../images/text_crops/")



# --- CLASES ---

class TextDetection():
    def __init__(self, pb_file, conf_threshold=0.001, iou_threshold=0.1):
        
        self.pb_file = pb_file
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.net = cv2.dnn.readNet(self.pb_file)

    def predict(self, img_orig, newsize, padding):
        
        # Pre procesamiento de la imagen de entrada. Creo blob para insertar en la red
        img_resized, rH, rW = imagePreprocess(img_orig, newsize, newsize)
        
        # Obtengo lista de indices de confidencia y geometrias de bboxs para todas las detecciones que se encontraron
        (scores, geometry) = self.forward(img_resized)
        
        # Filtro por indice de confidencia y decodifico cada bboxs a (x1,y1,x2,y2,conf,angulo,x_offset,y_offset)
        bboxs = decode(scores, geometry, self.conf_threshold, rH, rW)

        # Aplico non max suppression para eliminar bboxs redundantes
        bboxs = nonMaxSuppression(bboxs, self.iou_threshold)

        # Control de limites y agregado de padeo
        bboxs = addPadding(bboxs, img_orig, padding)

        # [[upleft_x, upleft_y, bottomright_x, bootonright_y, score,angle, x_offset, y_offset], ...]
        return bboxs

    def forward(self, img):
        # Obtengo alto y ancho de imagen
        H, W = img.shape[:2]      
        
        # Creo lista de layers de las cuales voy a necesitar extraer informacion
        layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
        
        # Construyo el blob desde la imagen para poder realizar el forwarding en la red
        # blob = cv2.dnn.blobFromImage(img, 1.0, (H, W), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(img, 1.0, (H, W), swapRB=True, crop=False)

        self.net.setInput(blob)
        
        scores, geometry = self.net.forward(layerNames)

        return (scores, geometry)
        

# --- FUNCIONES PRINCIPALES PARA LA OBTENCION Y DECODIFICACION DE BBOXS DE AREAS DE TEXTO ---

def borderlinesControl(x1, y1, x2, y2, img_orig):
    """
    (Nota: EAST a veces se comporta erroneamente en valores borderlines 
    otorgando valores negativos o que exeden limites de imagen original)
        
    Input:
        box = x1,y1,x2,y2. coordenadas correspondientes a esq_sup_izq y esq_inf_der.
    """

    maxH,maxW = img_orig.shape[:2]
    
    x1 = max(x1,0)      # x1
    y1 = max(y1,0)      # y1
    x2 = min(x2,maxW)   # x2
    y2 = min(y2,maxH)   # y2

    return x1, y1, x2, y2

    
def addPadding(bboxs, img_orig, padding):
    """
    Funcion para agregado de padding a detecciones y control de bordes 
    (Nota: EAST a veces se comporta erroneamente en valores borderlines 
    otorgando valores negativos o que exedentes limites de imagen original)
    """

    assert (padding >= 0 and padding <= 100)

    maxH,maxW = img_orig.shape[:2]
    num_bboxs = len(bboxs)

    for i in range(num_bboxs):

        # Control de valores borderline
        x1, y1, x2, y2 = borderlinesControl(bboxs[i,0], bboxs[i,1], bboxs[i,2], bboxs[i,3], img_orig)

        bboxs[i,0] = x1
        bboxs[i,1] = y1
        bboxs[i,2] = x2
        bboxs[i,3] = y2

        # Agrego (padding %) a cada lado        
        if padding > 0:

            deltaX = (x2 - x1) * padding / 100.0
            deltaY = (y2 - y1) * padding / 100.0

            bboxs[i,0] = int(max(0, x1 - deltaX))       # x1
            bboxs[i,1] = int(max(0, y1 - deltaY))       # y1
            bboxs[i,2] = int(min(maxW, x2 + 2*deltaX))  # x2
            bboxs[i,3] = int(min(maxH, y2 + 2*deltaY))  # y2

    return bboxs


def imagePreprocess(img, new_height, new_width):
    
    # Compruebo que tanto el alto y el ancho sean multiplos de 32 (necesario para entrada a la red)
    assert(new_height % 32 == 0)
    assert(new_width % 32 == 0)
    
    H, W = img.shape[:2]

    resized = cv2.resize(img, (new_height, new_width))

    # Determino la relacion de aspecto entre la imagen original y la redimensionada
    rH = H / float(new_height)
    rW = W / float(new_width)

    
    return resized, rH, rW


def decode(scores, geometry, conf_threshold, ratioHeight, ratioWidth):
    """
    Recibo las estructuras de salida <scores> y <geometry> del metodo forward donde:
        scores = array de indices de confidencia de c/u del las celdas del feature maps de la salida de la red EAST.
        geometry = array de d0,d1,d2,d3 y angulo de rotacion de c/u de las celdas del feature maps de la salida de la red EAST.

    Y retorno array escalado a la imagen real de:
        X1,Y1: esquina superior izq por cada bounding box
        X2,Y2: esquina inferior der por cada bounding box
        scores: array de confidencias por cada bounding box
        angles: angulo de rotacion por cada bounding box
        x_offset
        y_offset
    """
   
    # Elimino primer dimension para trabajar con datos mas comodos (1,1,inputH/4,inputW/4) > (1,inputH/4,inputW/4)
    scores = scores[0]
    geometry = geometry[0]

    # Me quedo con las celdas que posea un score > conf_threshold
    mask = (scores > conf_threshold)

    scoresData = scores[mask]
    d0 = (geometry[0])[mask[0]]
    d1 = (geometry[1])[mask[0]]
    d2 = (geometry[2])[mask[0]]
    d3 = (geometry[3])[mask[0]]
    angles = (geometry[4])[mask[0]]


    # Recupero indices de valores True de mask y computo ofsset en relacion a la dimension de la imagenn de 
    # entrada a la red, ya que nuestro features maps es 4x veces menor.
    x_offset = (np.where(mask)[2]) * 4.0
    y_offset = (np.where(mask)[1]) * 4.0

    # Calculo esquinas de bounding box (en relacion a imagen de entrada a la red ya que los offset estan escalados)
    ur = np.array((d1 + x_offset, y_offset - d0)).T
    dr = np.array((d1 + x_offset, y_offset + d2)).T
    ul = np.array((x_offset - d3, y_offset - d0)).T
    dl = np.array((x_offset - d3, y_offset + d2)).T

    # Redimensiono esquinas en relacion al tamano de la imagen original antes de ser redimensionada a un multiplo de 32
    # para entrar a la red.
    ur = ur * (ratioWidth,ratioHeight)
    dl = dl * (ratioWidth,ratioHeight)
    ul = ul * (ratioWidth,ratioHeight)
    dr = dr * (ratioWidth,ratioHeight)
    x_offset = x_offset * ratioWidth
    y_offset = y_offset * ratioHeight

    X1 = ul[:,0]
    Y1 = ul[:,1]
    X2 = dr[:,0]
    Y2 = dr[:,1]


    bboxs = np.array([X1,Y1,X2,Y2,scoresData,angles,x_offset,y_offset]).T

    return bboxs

    
def getIou(box1, box2):
    """
    Input
    box1.shape == (1,8)
    box2.shape == (:,8)
    """
    assert(box1.shape == (1,8))
    assert(box2.shape[1] == (8))

    x1, y1, x2, y2 = box1[:,0], box1[:,1], box1[:,2] , box1[:,3]
    xs1, ys1, xs2, ys2 = box2[:,0], box2[:,1], box2[:,2] , box2[:,3]
    
    # calculo interseccicion
    rect_x1 = np.maximum(x1,xs1)
    rect_y1 = np.maximum(y1,ys1)
    rect_x2 = np.minimum(x2,xs2)
    rect_y2 = np.minimum(y2,ys2)

    inter_area = np.maximum(rect_x2 - rect_x1 ,0 ) * np.maximum(rect_y2 - rect_y1 ,0)

    # Calculo areas de cada bbox
    bbox_area = (x2 - x1 ) * (y2 - y1)
    bbox_s_area = (xs2 - xs1) * (ys2 - ys1)

    iou = inter_area / (bbox_area + bbox_s_area - inter_area)

    return iou


def nonMaxSuppression(bboxs, iou_threshold):

    # Ordeno detecciones de forma descendiente en relacion a indice de confidencia que posee indice 4 en este caso.
    # Recupero indices
    idx = np.argsort(bboxs[:,4])

    # idx orden ascendene -> orden descendente
    idx = idx[::-1]
    sort_bboxs = bboxs[idx]

    num_detections = bboxs.shape[0]

    # 1. tomo la deteccion con mayor confidencia
    # 2. calculo IoU entre la deteccion de 1. y todas las demas
    #     - Elimino aquellas que tengan un IoU > nms_threshold
    # 3. vuelvo al paso 1. hasta que no queden mas detecciones
    for i in range(num_detections):

        try:
            # TODO: Lo inserto dentro de un try ya que debido a las eliminaciones
            # puede que el indice caiga fuera de rango
            b1 = bboxs[i].reshape(1,-1)
            b2 = bboxs[i+1:]
            # Obtengo lista de ious de b1 con respecto a b2 (o en su defecto la lista de bbox de b2)
            ious = getIou(b1,b2)
        except:
            break

        # Elimino todas las detecciones con un iou > io_threshold
        mask = (ious < iou_threshold)
        id_mask = np.where(mask)
        bboxs = np.concatenate((bboxs[:i+1],b2[id_mask]))

    return bboxs



# --- FUNCIONES PARA LA OBTENCION DE RECORTES DE DETECCIONES ---

def getCrops(img_orig, bboxs, show=False, save=False):
    """
    img_orig : Imagen original de la cual se sacaran los distintos recortes
    bboxs : Lista de bounding box de c/u de las zonas con texto detectado
    padding : Porcentaje de pixeles que extenderan los limites detetados en cada dimension
    show : Flag para mostrar la imagen recortada
    save : Flag que permite guardar una copia de la imagen recortada en el directorio text_crops
    """

    maxH , maxW = img_orig.shape[:2]
    crops = []


    for box in bboxs:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        img_crop = img_orig[y1:y2, x1:x2, :]

        if show:    
            cv2.namedWindow('Crops',cv2.WINDOW_NORMAL)
            cv2.imshow("Crops", img_crop)
            cv2.waitKey(0)

        if save:
            name = (time.asctime().strip().replace(" ", "").replace(":", ""))
            cv2.imwrite(CROPS_FOLDER+name+".png",img_crop)
            print(name+".png")
            time.sleep(1)

        crops.append(img_crop)

    if show:
        cv2.destroyAllWindows()

    
    return crops



# --- FUNCIONES PARA LA OBTENCION DE RECORTES DE DETECCIONES EN RELACION AL ANGULO DE DETECCION ---

def getCropsUsingPolygons(img_orig, bboxs, show=False, save=False):
    """
    img_orig : Imagen original de la cual se sacaran los distintos recortes
    bboxs : Lista de bounding box de c/u de las zonas con texto detectado
    show : Flag para mostrar la imagen recortada
    save : Flag que permite guardar una copia de la imagen recortada en el directorio text_crops
    """

    crops = []
    maxH , maxW = img_orig.shape[:2]

    # Recupero puntos de poligono
    polygons = createPolygons(bboxs)
    num_pol = len(polygons)
    
    for i in range(num_pol):
    
        ul_x = polygons[i,0]
        ul_y = polygons[i,1]
        
        ur_x = polygons[i,2]
        ur_y = polygons[i,3]

        dr_x = polygons[i,4]
        dr_y = polygons[i,5]

        dl_x = polygons[i,6]
        dl_y = polygons[i,7]

        # Convierdo a grados
        theta = bboxs[i,5]*180/np.pi

        # Selecciono las esquinas izq superior y derecha inferior que mejor capturen la totalidad
        # de la imagen rotada
        if theta < 0:
            x1 = dl_x
            y1 = ul_y
            x2 = ur_x
            y2 = dr_y

        else:
            x1 = ul_x
            y1 = ur_y
            x2 = dr_x
            y2 = dl_y

        # Control de extremos. 
        #   Los vertices de los poligonos, al tener un angulo, pueden salirse de la imagen
        #   otorgando valores negativos y/o superiores a limites provocando un error en 
        #   el posterior recorte de imagen.

        x1, y1, x2, y2 = borderlinesControl(x1, y1, x2, y2, img_orig)

        # Obtengo crop de deteccion
        img_crop = img_orig[y1:y2, x1:x2, :]

        # Roto imagen en relacion al punto central de la misma un angulo de -theta para volver a su lugar
        M = cv2.getRotationMatrix2D( ((x2 - x1)/2, (y2 - y1)/2), -theta, 1.0)
        img_crop_rot = cv2.warpAffine(img_crop, M, (x2 - x1, y2 - y1))
        

        if show:
            cv2.namedWindow('Crops',cv2.WINDOW_NORMAL)
            cv2.imshow("Crops", img_crop_rot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save:
            name = (time.asctime().strip().replace(" ", "").replace(":", ""))
            cv2.imwrite(CROPS_FOLDER+name+".png",img_crop_rot)
            print(name+".png")
            time.sleep(1)
        
        # Muestro posiciones de texto DEBUG
        # print("\t['line_number?',{},{},{},{}]".format(x1,y1,x2,y2))

        crops.append(img_crop_rot)
        

    if show:
        cv2.destroyAllWindows()
    
    return crops
    

def createPolygons(bboxs):
    """
    Recibo bboxs = [X1,Y1,X2,Y2,scores,angles,x_offsets,y_offsets] donde:
        X1,Y1: esquina superior izq por cada bounding box
        X2,Y2: esquina inferior der por cada bounding box
        scores: array de confidencias por cada bounding box
        angles: angulo de rotacion por cada bounding box
        x_offset
        y_offset

    Y retorno array de coordenadas de las cuatro esquinas del bbox en relacion al angulo de rotacion:
        [upper_left,upper_right,down_right,down_left]
    """
    
    x1 = bboxs[:,0]
    y1 = bboxs[:,1]
    x2 = bboxs[:,2]
    y2 = bboxs[:,3]
    angles = bboxs[:,5]
    x_offset = bboxs[:,6]
    y_offset = bboxs[:,7]

    ul_x = x1
    ul_y = y1

    ur_x = x2
    ur_y = y1

    dl_x = x1
    dl_y = y2

    dr_x = x2
    dr_y = y2

    ul = rotatePoints(ul_x,ul_y,angles,x_offset,y_offset)
    ur = rotatePoints(ur_x,ur_y,angles,x_offset,y_offset)
    dl = rotatePoints(dl_x,dl_y,angles,x_offset,y_offset)
    dr = rotatePoints(dr_x,dr_y,angles,x_offset,y_offset)

    # Return un arreglo del tipo [ul_x,ul_y,ur_x,ur_y,dr_x,dr_y,dl_x,dl_y]
    return np.concatenate((ul,ur,dr,dl), axis=1)


def rotatePoints(x_point,y_point, theta, x_offset, y_offset):
    """
    Retorna el vector resultante de rotar el vector ("x_point","y_point") un angulo "theta" en 
    relacion al punto central ("x_offset","y_offset")
    """

    # Para poner el origen en el comienzo del vector (x_point,y_point)
    x_adjusted = (x_point - x_offset)
    y_adjusted = (y_point - y_offset)

    cos = np.cos(theta)
    sin = np.sin(theta)
    
    # Formula para calcular el (xVector,yVector) en relacion a un punto que no es (0,0)
    new_x_point = x_offset + (cos * x_adjusted) + (sin * y_adjusted)
    new_y_point = y_offset + (cos * y_adjusted) - (sin * x_adjusted)

    return np.array([new_x_point, new_y_point],np.int32).T



# --- FUNCIONES PARA VISUALIZACION DE DETECCIONES SOBRE IMAGEN ORIGINAL ---

def drawBboxs(img_orig, bboxs):
    img_with_bbox = img_orig.copy()
    for box in bboxs:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        

        # color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
        color = (0,0,255)
        cv2.rectangle(img_with_bbox,(x1,y1),(x2,y2), color,2)

    cv2.namedWindow('Imagen',cv2.WINDOW_NORMAL)
    cv2.imshow("Imagen", img_with_bbox)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()


def drawPolygons(img_orig, bboxs):
    img_with_bbox = img_orig.copy()

    for box in createPolygons(bboxs):

        points = box.reshape(-1,1,2)

        #color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
        color = (0,255,0)
        cv2.polylines(img_with_bbox, [points], True, color, 2)

    cv2.namedWindow('Imagen',cv2.WINDOW_NORMAL)
    cv2.imshow("Imagen", img_with_bbox)
    cv2.waitKey(50)
    # cv2.destroyAllWindows()



# --- FUNCIONES AUXILIARES ---

def getArguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_path", required=True, type=str, help="Path de la imagen o carpeta")
    ap.add_argument("-conf", "--conf_threshold", default=0.001, type=float, help="Indice de confidencia. (Default: 0.001)")
    ap.add_argument("-iou", "--iou_threshold", default=0.1, type=float, help="Indice de iou. (Default: 0.1)")
    ap.add_argument("-pad", "--padding", default=0, type=float, help="Recorta un -pad porciento menos de pixeles en relacion al ancho y alto de la deteccion en EAST. (Default: 0)")
    ap.add_argument("-newsize", "--newsize", default=128, type=int, help="Redimensionar a imagen cuadrada para entrada en EAST. (Default: 128)")
    ap.add_argument("-save", "--save", default="False", type=str, help="Guardar salida en carpeta text_crops. (Default: False)")
    ap.add_argument("-show", "--show", default="False", type=str, help="Muestro cada deteccion individual. (Default: False)")
    return vars(ap.parse_args())



# --- FUNCION PRINCIPAL MAIN ---

if __name__ == '__main__':
    
    args = getArguments()

    # Instancio la clase TextDetection para iniciar
    net = TextDetection(PB_FILE, conf_threshold=args["conf_threshold"], iou_threshold=args["iou_threshold"])

    # Leo imagen/es de entrada
    for image_path in processInputPath(args["image_path"], target_list=[".jpg",".JPG",".png"]):
        
        # img_orig = cv2.imread(args["image"])
        img_orig = cv2.imread(image_path)

        # Recupero bounding box para cada agrupacion de caracteres detectados
        bboxs = net.predict(img_orig, args["newsize"], args["padding"])
        
        if len(bboxs) > 0:
            # Muestro imagen con detecciones
            #drawBboxs(img_orig, bboxs)
            drawPolygons(img_orig, bboxs)
            
            #Obtengo recortes de bounding boxes sin rotar 
            # crops = getCrops(img_orig, bboxs, show = strtobool(args["show"]), save = strtobool(args["save"]))
            
            #Obtengo recortes de poligonos rotados a 0 grados (derechos)
            crops = getCropsUsingPolygons(img_orig, bboxs, show=strtobool(args["show"]), save=strtobool(args["save"]))