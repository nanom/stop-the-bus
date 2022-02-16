# --- CONFIGURACION DE UBUCACION PARA USO DE MODULOS --- 

import sys

# Agrego al sys.path el directorio padre para importar modulos
sys.path.append("..")



# --- IMPORTS ---

import numpy as np
import pytesseract, argparse, cv2, os

# Mis modulos
from utils import processInputPath, isEmpty



# -- CONSTANTES ---

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LANGUAJE = "eng"

DEFAULT_OEM = 1
    # OCR Engine modes:
    # 0    Legacy engine only.
    # 1    Neural nets LSTM engine only.
    # 2    Legacy + LSTM engines.
    # 3    Default, based on what is available.

DEFAULT_PSM = 7
    #Page segmentation modes:
    #  0    Orientation and script detection (OSD) only.
    #  1    Automatic page segmentation with OSD.
    #  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
    #  3    Fully automatic page segmentation, but no OSD. (Default)
    #  4    Assume a single column of text of variable sizes.
    #  5    Assume a single uniform block of vertically aligned text.
    #  6    Assume a single uniform block of text.
    #  7    Treat the image as a single text line.
    #  8    Treat the image as a single word.
    #  9    Treat the image as a single word in a circle.
    # 10    Treat the image as a single character.
    # 11    Sparse text. Find as much text as possible in no particular order.
    # 12    Sparse text with OSD.
    # 13    Raw line. Treat the image as a single text line,
    #       bypassing hacks that are Tesseract-specific.



# --- CLASES ---

class TextRecognition():
    def __init__(self, language, oem, psm):
        
        self.language = language
        self.oem = str(oem)
        self.psm = str(psm)
        self.config = ("--oem "+self.oem+" --psm "+self.psm)

        # self.config = ("-l "+self.language+" --oem "+self.oem+" --psm "+self.psm)
        # self.config = ("--oem "+self.oem+" --psm "+self.psm+ '--tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata"')
        # self.config = ("-l eng -c tessedit_char_whitelist=0123456789")

    def imageToString(self, crop):

        text = pytesseract.image_to_string(crop, lang=self.language, config=self.config)
        # text = pytesseract.image_to_string(crop, config="outputbase digits")
        # text = pytesseract.image_to_string(crop, config="-c tessedit_char_whitelist=0123456789")
        
        return text


    def getOnlyNumbers(self, text):
        # Descarto caracteres no numericos
        text = "".join(ch for ch in text if ord(ch) in range(48,58))

        return text



# --- FUNCIONES AUXILIARES ---
    
def getArguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image-path", required=True, type=str, help="Path tde imagen o carpeta")
    ap.add_argument("-l", "--language", default="eng", type=str, help="Idioma de imagen a detectar")
    ap.add_argument("-oem", "--oem", default=1, type=int, help="")
    ap.add_argument("-psm", "--psm", default=7, type=int, help="")
    return vars(ap.parse_args())



# --- FUNCION PRINCIPAL MAIN ---

if __name__ == '__main__':

    args = getArguments()

    # Instancio la clase TextRecognition
    net = TextRecognition(language=args["language"], oem=args["oem"], psm=args["psm"])

    # Leo imagen de entrada
    for image_path in processInputPath(args["image_path"], target_list=[".jpg",".JPG",".png"]):
        img_orig = cv2.imread(image_path)
        
        # #  SIN TESTEAR EN RESULTADOS FINALES -------------------------
        gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)        
        t, threshold1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        t, threshold2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        dst1 = cv2.GaussianBlur(threshold1,(3,3),cv2.BORDER_DEFAULT)
        dst2 = cv2.GaussianBlur(threshold2,(3,3),cv2.BORDER_DEFAULT)
        combinations = [img_orig, threshold1, dst1, threshold2, dst2]
        # -----------------------------------------------       

        # Convierto imagen en texto
        for img_orig in combinations:
            text = net.imageToString(img_orig)
            print("________________")
            print(">> {}".format(text))

            cv2.namedWindow('Preview',cv2.WINDOW_NORMAL)
            cv2.imshow("Preview", img_orig)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


