# --- IMPORTS ---


from object_detection import yolo, tools
from object_detection import ssd_mobilenet as mobilenet
from object_detection import color_spaces as c_spaces

from text_detection import east
from text_recognition import ocr

from utils import isEmpty
import argparse, sys



# --- FUNCIONES PARA UTILIZACION DE MODULOS PRINCIPALES ---

def getBuses(net, image, with_transf=True, channels_to_fuse=[],  verbose=False):
    """
    Uso de modelo OBJECT DETECTION.
    Retorna una lista de imagenes <buses_list>, de todos los autobuses detectados
    en imagen de entrada <image>. 
    
    Input
        net: Instancia de clase de red OBJET DETECTION a utilizar. (Yolo, SddMobileNet, etc)
        image: Imagen o path de imagen.
        with_transf: FLag para indicar si se aplican, o no, transformaciones de imagen a cada deteccion.
        channels_to_fuse: Lista de 3 Id's de canales para generar transformaciones en base a sus combinaciones.
        verbose: Flag para depuracion.
    """

    buses_list = []

    # Obtengo bounding box de cada deteccion en <image>
    bboxs = net.predict(image, class_name="bus", filter_by_ratio=True)


    if verbose:  
        # Muestro imagen con detecciones
        tools.drawBboxs(net.img_orig, bboxs)

    if not isEmpty(bboxs):

        if verbose:
            # print("\nDetectado!")
        
            # Imprimo detecciones
            print(*bboxs, sep = "\n")
        
        # Obtengo los recortes de cada deteccion
        buses_list = tools.getCrops(    net.img_orig, 
                                        bboxs, 
                                        top_square=True,
                                        first_cuadrant=False,
                                        originsize_heursitc=False,
                                        show=False, 
                                        save=False
                                    )
        
        if with_transf:
            
            # Genero lista de transformaciones a cada autobus detetctado
            # Nota: Ahora <buses_list> sera una lista de tuplas [(id_transf, transformacion),..]
            
            if isEmpty(channels_to_fuse):
                buses_list = c_spaces.getTransformations(buses_list)

            else:
                assert(len(channels_to_fuse) == 3)
                buses_list = c_spaces.getFuseAndPermut3Channels(buses_list, channels_to_fuse)

    else:
        if verbose:
            pass#print("\nNingun autobus detectado!")


    return buses_list


def getTextAreas(net, image, newsize, padding, verbose=False):
    """
    Uso de EAST.
    Retorna una lista de imagenes <texts_areas> de todas aquellas areas que se hayan detectado como
    caracteres en la imagen de entrada <image>, mediante el uso del algoritmo EAST.
    
    Input
        net: Instancia de clase EAST
        image: Imagen con areas de texto a detectar
        newsize: Dimension final de la imagen al que sera llevada c/u de las detecciones.
        padding: Porcentaje de 'padeo' de cada una de las detecciones.
        verbose: Flag para depuracion.
    """

    texts_areas = []

    # Obtengo bounding box de cada area de texto detectado en <image>
    bboxs = net.predict(image, newsize=newsize, padding=padding)

    # Solo me quedo con bboxs de interes. (2/3 de la imagen hacia arriba)
    # Nota: bboxs = [[upleft_x, upleft_y, bottonright_x, bottonright_y, score, angle, offset_x, offset_y],...]
    
    # height = image.shape[0]
    # threshold = height - (height / 3.0)
    # mask = (bboxs[:,3] <= threshold)
    # bboxs = bboxs[mask]

    if verbose:
        # Muestro imagen con detecciones    
        east.drawPolygons(image, bboxs)
        

    if not isEmpty(bboxs):
        # Obtengo los recortes de cada deteccion
        texts_areas = east.getCropsUsingPolygons(image, bboxs, show=False, save=False)

    return texts_areas


def getNumberFromTextArea(net, image):
    """
    Uso de modulo OCR.
    Dada una imagen <image>, que contenga alguna secuencia de caracteres, recupera y transforma
    a un entero <prediction>, los numeros reconocidos en esta. None cc. 

    Input
        net: Instancia de clase OCR
        image: Imagen con secuencia de caracteres.
    """
    
    prediction = None

    # Convierto caracteres reconocidos en <image> a un string
    full_string = net.imageToString(image)

    # Limpio <full_string> dejando solo caracteres numericos
    only_number = net.getOnlyNumbers(full_string)

    if only_number != "":
        prediction = int(only_number)
  
    return prediction



# --- FUNCIONES PARA INICIALIZACION DE CLASES ---

def initMobileNet(conf_threshold=.5):
    
    # Instancio clase SddMobilnet para iniciar Mobilnet
    return mobilenet.SsdMobileNet(  mobilenet.CFG_FILE, 
                                    mobilenet.WEIGHTS_FILE, 
                                    mobilenet.CLASSES_FILE, 
                                    conf_threshold=conf_threshold
                                )


def initYolo(version='yolo2', conf_threshold=.5, iou_threshold=.45):

    # Instancio clase Darknet para iniciar YOLO
    return yolo.Darknet(    version=version,
                            conf_threshold=conf_threshold, 
                            iou_threshold=iou_threshold
                        )


def initEast(conf_threshold=.001, iou_threshold=.1):

    # Instancio clase TextDetections para iniciar EAST
    return east.TextDetection(      east.PB_FILE, 
                                    conf_threshold=conf_threshold, 
                                    iou_threshold=iou_threshold
                                )


def initOcr():
    
    # Instancio clase TextRecognition para iniciar OCR
    return ocr.TextRecognition(     language=ocr.DEFAULT_LANGUAJE, 
                                    oem=ocr.DEFAULT_OEM, 
                                    psm=ocr.DEFAULT_PSM
                            )


# --- FUNCION AUXILIARES ---

def choiceModel(opc, conf_threshold=.5, iou_threshold=.45):
    """
    Retorna clase instanciada de modelo de deteccion elegido
    """

    models_availables = ['mobilenet','yolov2','yolov3','yolov3t']

    if opc not in models_availables:
        sys.exit(f"Error!. Only {', '.join(models_availables)} models availables.")

    net = None

    if opc == "mobilenet":
        net = initMobileNet(conf_threshold)
    else:
        net = initYolo(opc, conf_threshold, iou_threshold)
    
    return net


def isBusComming(bus_numbers, expected_number):
    """
    Recibe una lista de numeros de autobuses <bus_numbers> y reporta los mejores candidatos en orden
    de frecuencia descendente de deteccion, retornando True sii <expected_number> esta incluido 
    en <bus_numbers>.
    """

    bus_founded = False

    # Calculo probabilidades de numeros obtenidos 
    cte = 1.0 / len(bus_numbers)
    bus_numbers_dic = {e:round(bus_numbers.count(e)*cte,2) for e in bus_numbers}

    # Ordeno segun mayores probabilidades
    bus_numbers_sorted = sorted(bus_numbers_dic.items(), key=lambda kv: kv[1], reverse=True)
    # os.system("clear")
    
    print("\n --- Potencial candidato ---")
    show = lambda tups_list: ["\t{}) LINEA:{} -> {}%".format(jth+1,tup[0],round(tup[1]*100,3)) for jth,tup in enumerate(tups_list)]
    print(*show(bus_numbers_sorted[:5]), sep="\n")
    
    if expected_number in bus_numbers:
        bus_founded = True

    return bus_founded


def worker(net, ith, queue, output, verbose=False):
    """
    Input
        net: Instancia de red OCR.
        ith: Identificador de thread (Opcional).
        queue: Cola de imagenes de areas de texto detectadas.
        output: Lista donde se guardara las predicciones.
        verbose: Flag para depuracion.
    """
    
    while not queue.empty():
        text_area = queue.get()
        prediction = getNumberFromTextArea(net, text_area)
        
        if prediction != None:
            output.append(prediction)
            if verbose:
                print("proc_{} > {}".format(ith,prediction))

def getArguments():
    ap = argparse.ArgumentParser(description='Pipeline')
    ap.add_argument("-i","--image_path", required=True, type=str, 
        help="Path de imagen o directorio")
    ap.add_argument("-m", "--model", default='mobilenet', type=str, 
        help="Modelo de detector a utilizar. Use -model ['yolov2'|'yolov3'|'yolov3t'|'mobilenet']. (Default: 'mobilenet')")
    ap.add_argument("-n","--expected_number", required=True, default=000000, type=int, 
        help="Numero de autobus esperado")
    ap.add_argument("-v","--verbose", default="True", type=str, 
        help="Flag para depuracion. Muestra salidas de modulos principales. (Defaul: False)")
    ap.add_argument("-oconf", "--od_conf_threshold", default=0.5, type=float, 
        help="Umbral de confidencia de detector. (Default: 0.5)")
    ap.add_argument("-oiou", "--od_iou_threshold", default=0.45, type=float, 
        help="Indice de iou de detector. (Default: 0.45)")
    ap.add_argument("-econf", "--e_conf_threshold", default=0.001, type=float, 
        help="Indice de confidencia de EAST. (Default: 0.001)")
    ap.add_argument("-eiou", "--e_iou_threshold", default=0.1, type=float, 
        help="Indice de iou de EAST. (Default: 0.1)")
    ap.add_argument("-newsize", "--newsize", default=128, type=int, 
        help="Redimensionar a imagen cuadrada para entrada en EAST. (Default: 128)")
    ap.add_argument("-pad", "--padding", default=0, type=float, 
        help="Recorta un -pad porciento menos de pixeles en relacion al ancho y alto de la deteccion en EAST. (Default: 0)")
    return vars(ap.parse_args())


def printSystemConfig(args):
    print("\nConfiguracion actual de sistema:")
    print("\t{}: conf:{} iou:{}".format(args["model"],args["od_conf_threshold"],args["od_iou_threshold"]))
    print("\tEAST: conf:{} iou:{} pad:{} size:{}".format(args["e_conf_threshold"],args["e_iou_threshold"],args["padding"],args["newsize"]))