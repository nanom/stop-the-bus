from queue import Queue
from threading import Thread
from distutils.util import strtobool
import argparse, os, cv2

from api import initEast, initOcr, getBuses, getTextAreas, choiceModel, isBusComming, worker, getArguments
from utils import Timer, isEmpty


NUMBER_OF_THREAD = 4


cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FPS, 5)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
fps = cam.get(cv2.CAP_PROP_FPS)

net = choiceModel("mobilenet", 0.5, 0.45)
east_net = initEast(0.001, 0.1)
ocr_net = initOcr()
crono = Timer()


def getArguments():
    ap = argparse.ArgumentParser(description='Run camera detection')
    ap.add_argument("-n", "--num", required=True, type=int, help="Numero esperado")
    ap.add_argument("-v", "--verbose", default="False", type=str, help="Debug. default:False")
    return vars(ap.parse_args())


# --- FUNCION PRINCIPAL MAIN ---

if __name__ == '__main__':

    args = getArguments()
    
    os.system("clear")

    print("Sistema activado esperando LINEA:{} a {} fps...".format(args["num"],fps))
    
    while(True):

        buses_list = []
        buses_detected = 0

        while (buses_detected < 1):
            _ , frame = cam.read()
            buses = getBuses(net=net, image=frame, verbose=True)
            
            if not isEmpty(buses):
                buses_detected += 1
                
                for bus in buses:
                    buses_list.append(bus)
            # cv2.waitKey(5)

        if not isEmpty(buses_list):      
            crono.start()
            text_areas = Queue(maxsize=0)
            
            for _ , bus in buses_list:
                areas = getTextAreas(east_net, bus, 96, 5, verbose=strtobool(args["verbose"]))

                if not isEmpty(areas):
                    for area in areas:
                        text_areas.put(area)
            
            if not text_areas.empty():
                bus_numbers = []
                thread_list = []

                for ith in range(NUMBER_OF_THREAD):
                    t = Thread(target=worker, args=(ocr_net, ith, text_areas, bus_numbers, False,))
                    t.start()
                    thread_list.append(t)

                for t in thread_list:
                    t.join()

                if not isEmpty(bus_numbers):
                  if isBusComming(bus_numbers, args["num"]):
                    print("----------------------")
                    print("AUTOBUS EN CAMINO!")
                    print("----------------------\n")
                    crono.stop()
                    break   

    cv2.destroyAllWindows()
    cam.release()
