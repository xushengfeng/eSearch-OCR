
from paddleocr import PaddleOCR
import json
import base64
import cv2
import numpy as np

开启纠错 = False


def correct(v):
    开启纠错 = v
    if v:
        import pycorrector


def ocr(data):
    ocr = PaddleOCR(use_gpu=False, lang="ch")  # 首次执行会自动下载模型文件
    data = json.loads(str(data, encoding="utf-8"))
    image_string = data["image"]
    img_data = base64.b64decode(image_string)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = ocr.ocr(img_np)
    dic = {}
    dic["words_result_num"] = len(result)
    dic["words_result"] = []

    for index, line in enumerate(result):
        dic["words_result"].append({})
        if 开启纠错:
            correct_sent, err = pycorrector.correct(str(line[1][0]))
            dic["words_result"][index]["words"] = correct_sent
        else:
            dic["words_result"][index]["words"] = str(line[1][0])
        dic["words_result"][index]["location"] = xywh(line[0])
        dic["words_result"][index]["probability"] = float(line[1][1])
    return dic


def xywh(o_list):
    x1 = min([item[0] for item in o_list])
    x2 = max([item[0] for item in o_list])
    y1 = min([item[1] for item in o_list])
    y2 = max([item[1] for item in o_list])
    l = {}
    l["top"] = int(x1)
    l["left"] = int(y1)
    l["width"] = int(x2 - x1)
    l["height"] = int(y2 - y1)
    return l
