# Bus Number Detection

### 1. Install using Conda
> $ conda create -n myenv python=3.6

> $ conda activate myenv

> $ pip install -r requirements.txt 


### 2. Run demo
> $ python short_pipeline.py -i test_sets/set11/ -n 18

```bash
-i IMAGE_PATH, --image_path IMAGE_PATH
                    Path de imagen o directorio
-m MODEL, --model MODEL
                    Modelo de detector a utilizar. Use -model
                    ['yolo'|'mobilenet']. (Default: 'mobilenet')
-n EXPECTED_NUMBER, --expected_number EXPECTED_NUMBER
                    Numero de autobus esperado
-v VERBOSE, --verbose VERBOSE
                    Flag para depuracion. Muestra salidas de modulos
                    principales. (Defaul: False)
-oconf OD_CONF_THRESHOLD, --od_conf_threshold OD_CONF_THRESHOLD
                    Umbral de confidencia de detector. (Default: 0.5)
-oiou OD_IOU_THRESHOLD, --od_iou_threshold OD_IOU_THRESHOLD
                    Indice de iou de detector. (Default: 0.45)
-econf E_CONF_THRESHOLD, --e_conf_threshold E_CONF_THRESHOLD
                    Indice de confidencia de EAST. (Default: 0.001)
-eiou E_IOU_THRESHOLD, --e_iou_threshold E_IOU_THRESHOLD
                    Indice de iou de EAST. (Default: 0.1)
-newsize NEWSIZE, --newsize NEWSIZE
                    Redimensionar a imagen cuadrada para entrada en EAST.
                    (Default: 128)
-pad PADDING, --padding PADDING
                    Recorta un -pad porciento menos de pixeles en relacion
                    al ancho y alto de la deteccion en EAST. (Default: 0)
```

![ullpipelien gif](/readme/short_pipeline.gif)

### Stages of buses line numbers detection
1.  Buses detection
![bus detection gif](/readme/bus_detection.gif)

2.  Line numbers  detection
![lines detection gif](/readme/line_detection.gif)

3.  Line numbers recognition
![lines recognition gif](/readme/line_recognition.gif)