# --- IMPORTS ---

from threading import Thread
from queue import Queue
from distutils.util import strtobool
import argparse

# Mis modulos
from api import getBuses, getTextAreas, initEast, initOcr, choiceModel, isBusComming, worker, printSystemConfig, getArguments
from utils import Timer, processInputPath, isEmpty



# --- CONSTANTES ---

NUMBER_OF_THREAD = 4



# --- FUNCION PRINCIPAL MAIN ---

if __name__ == '__main__':

    args = getArguments()

    # --- INICIALIZACION DE CLASES Y VARIABLES ---

    VERBOSE = strtobool(args["verbose"])
    crono = Timer()

    objdet_net = choiceModel(args["model"], args["od_conf_threshold"], args["od_iou_threshold"] )
    east_net = initEast(args["e_conf_threshold"], args["e_iou_threshold"] )
    ocr_net = initOcr()
    

    # -- ETAPA de BUSQUEDA DE AUTOBUSES y DETECCION DE TEXTO --

    # Inicio control de tiempo de analisis
    crono.start()

    # Inicializacion de cola para alojar zonas de texto detectadas en cada autobus
    text_areas = Queue(maxsize=0)

    path_list = processInputPath(args["image_path"], target_list=[".jpg",".JPG",".png"])

    for img_path in path_list:

        # Recupero imagenes de autobuses 
        buses_images = getBuses(objdet_net, img_path, verbose=VERBOSE)

        if not isEmpty(buses_images):
            for _ , bus in buses_images:

                # Obtengo lista de imagenes de zonas de texto encontradas en <bus>
                areas = getTextAreas(east_net, bus, args["newsize"], args["padding"], verbose=VERBOSE)
                
                if not isEmpty(areas):
                    for area in areas:
                        text_areas.put(area)


    
    # --- ETAPA de RECONOCIMIENTO DE TEXTO ---

    # Inicializo lista para numeros de autobuses detectados.
    bus_numbers = []
    
    if not text_areas.empty():
        
        # Creacion y lanzamiento de hilos para concurrencia.
        thread_list = []
        for ith in range(NUMBER_OF_THREAD):
            t = Thread(target=worker, args=(ocr_net, ith, text_areas, bus_numbers, VERBOSE, ))
            t.start()
            thread_list.append(t)

        # Espero a que terminen su trabajo
        for t in thread_list:
            t.join()


    
    # --- ETAPA DE ANALISIS DE CANDIDATOS Y BUSQUEDA DE NUMERO DESEADO ---

    if not isEmpty(bus_numbers):

        if isBusComming(bus_numbers, args["expected_number"]):
            print("\n\n----------------------")
            print("AUTOBUS EN CAMINO!")
            print("----------------------")

    
    # --- IMPRESION DE CONFIGURACION DE SISTEMA Y PARAMETRO VARIOS ---

    printSystemConfig(args)
    print("\t{} frames analizados".format(len(path_list)))
    crono.stop()