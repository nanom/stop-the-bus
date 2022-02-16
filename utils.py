import os, sys, time, cv2

# --- CLASES VARIAS ---

class Timer():
    def __init__(self):
        self.init = 0
        self.end = 0

    def start(self):
        self.init = time.time()
        print("Inicio de analisis ...")
    
    def stop(self):
        self.end = time.time() - self.init
        print("Finalizacion de analisis en {}".format(self.toString(self.end)))

    def toString(self, a_time):
        return str(round(a_time // 60,2))+"m"+str(round(a_time % 60,2))+"s"


def loadImage(path_name):
    return cv2.imread(path_name)


def isEmpty(l):
    return (len(l) == 0)


def processInputPath(path, target_list):
    """
    Comprueba si <path> hace referencia a un archivo o a un directorio y 
    devuelve una lista de sus respectivas direcciones absolutas.
    """
    path_list = []
    # target_list = [".jpg",".JPG",".png"]


    # Compruebo si es un archivo
    if os.path.isfile(path):
        path_list.append(os.path.join(os.getcwd(),path))
    
    # Compruebo si es un directorio
    elif os.path.isdir(path):
        
        # En caso de que se haya agregado / al final, lo retiro
        if path[-1] == "/":
            path = path[:-1]

        path_list = [   os.path.join(os.getcwd(),path+"/"+img) for img in os.listdir(path) 
                            if os.path.splitext(img)[1] in target_list
                    ]
    else:
        sys.exit("Error: '{}'\n\tEl path de la carpeta o archivo es inexistente. Verifique e intente nuevamente".format(path))

    try:
        # Si las imagenes poseen como nombre solo numeros, las ordeno segun orden entero. Corrige ordenamiento 1,100,101,2 -> 1,2,100,101
        path_list = sorted(path_list, key=lambda file: int(os.path.splitext(os.path.basename(file))[0]))
    except:
        # Si la/s no poseen un numero de nombre, las ordeno de manera alfabetica
        path_list = sorted(path_list)

    return  path_list