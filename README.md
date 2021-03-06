# stop-the-bus
[[Paper]](https://49jaiio.sadio.org.ar/pdfs/asai/ASAI-10.pdf) [Colab] [[Demo Video]](https://www.youtube.com/watch?v=DeLpJ9ud7p4)

<p align="center">
<img src="readme/arquitectura.png" alt="Overview" style="width:70%;"/>
</p>


The goal of this project is the exploitation of computer vision techniques and the analysis of images for the generation of a tool that potentially allows people with visual impairments to be assisted. To achieve this, a modular architecture based on object detectors and optical character recognition is presented and evaluated, mainly constituted by two stages: one for the detection buses, based on the SSD-MobileNet object detection model; and another, responsible for line number recognition, where the EAST and OCR-Tesseract text detection and recognition models are tested, respectively. With a maximum probability of recognition of 62\% in a simple image; over an image sequence, the final system was able to correctly recognize the bus line in 72\% of the cases.

For more information, please refer to our paper: [Stop the Bus!: Computer vision for automatic recognition of urban bus lines.](https://49jaiio.sadio.org.ar/pdfs/asai/ASAI-10.pdf)

## Install
```
git clone https://github.com/nanom/stop-the-bus.git
conda create -n myenv python=3.6
conda activate myenv
pip install -r requirements.txt
```

## Usage
It has three modules to be able to run the detection system. Detections can be done using three different models:  **SSD-MobileNet**, **YOLO v2** and **YOLO v3 tiny**.

* `short_pipeline.py:` Given a sequence of images, give notice of the arrival of the bus, when the first detection of the expected line number occurs.
* `long_pipeline.py:` Given a sequence of images, it gives notice of the arrival of the bus, if at the end of the processing of all the images of the entered sequence, the expected line number is detected.
* `real_camera.py`: Bus number lines recognition in real time through a camera device.


Run with:
```
python short_pipeline.py -i test_images/ -m mobilenet -n 66
```
More configurations with `--help` flag:
```
$ python short_pipeline.py --help
usage: short_pipeline.py [-h] -i IMAGE_PATH [-m MODEL] -n EXPECTED_NUMBER
                         [-v VERBOSE] [-oconf OD_CONF_THRESHOLD]
                         [-oiou OD_IOU_THRESHOLD] [-econf E_CONF_THRESHOLD]
                         [-eiou E_IOU_THRESHOLD] [-newsize NEWSIZE]
                         [-pad PADDING]

Stop The Bus!

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_PATH, --image_path IMAGE_PATH
                        Path of image or directory
  -m MODEL, --model MODEL
                        Detector model to use. Use --model
                        ['mobilenet'|'yolov2'|'yolov3t']. (Default:
                        'mobilenet')
  -n EXPECTED_NUMBER, --expected_number EXPECTED_NUMBER
                        Expected bus number
  -v VERBOSE, --verbose VERBOSE
                        Flag for debugging. Show output from main modules.
                        (Default: False)
  -oconf OD_CONF_THRESHOLD, --od_conf_threshold OD_CONF_THRESHOLD
                        Detector confidence threshold. (Default: 0.5)
  -oiou OD_IOU_THRESHOLD, --od_iou_threshold OD_IOU_THRESHOLD
                        Detector IoU index. (Default: 0.45)
  -econf E_CONF_THRESHOLD, --e_conf_threshold E_CONF_THRESHOLD
                        EAST confidence index. (Default: 0.001)
  -eiou E_IOU_THRESHOLD, --e_iou_threshold E_IOU_THRESHOLD
                        EAST IoU index. (Default: 0.1)
  -newsize NEWSIZE, --newsize NEWSIZE
                        Resize to square image for EAST input. (Default: 128)
  -pad PADDING, --padding PADDING
                        Trips one '--padding' percent less pixels relative to
                        the width and height of the detection in EAST.
                        (Default: 0)
```


## Stages of buses line numbers recognition
1.  Bus detection stage
<p align="center">
<img src="readme/bus_detection.gif" alt="drawing" style="width:50%;"/>
</p>


2.  Line numbers  detection stage
<p align="center">
<img src="readme/line_detection.gif" alt="drawing" style="width:50%;"/>
</p>

3.  Line numbers recognition stage
<p align="center">
<img src="readme/line_recognition.gif" alt="drawing" style="width:50%;"/>
</p>

## Reference
```
@inproceedings{maina2020stop,
  title={Stop the Bus!: computer vision for automatic recognition of urban bus lines},
  author={Maina, Hern{\'a}n J and S{\'a}nchez, Jorge A},
  booktitle={XXI Simposio Argentino de Inteligencia Artificial (ASAI 2020)-JAIIO 49 (Modalidad virtual)},
  year={2020}
}
```
