import argparse
import web
from paddleocr import PaddleOCR
import json
import base64
import cv2
import numpy as np

开启纠错 = False
端口 = 8080

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--check", action="store_true")
parser.add_argument("-p", "--port", type=int)
args = parser.parse_args()
if args.check:
    开启纠错 = True
    import pycorrector
if args.port:
    端口 = args.port


def ocr(data, lang):
    ocr = PaddleOCR(use_gpu=False, lang=lang)  # 首次执行会自动下载模型文件
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


urls = ("/", "index")


def c_or_e(x):
    letters = 0
    space = 0
    digit = 0
    others = 0
    for c in x:
        if ord(c) in range(65, 91) or ord(c) in range(97, 123):
            letters += 1
        elif c.isspace():
            space += 1
        elif c.isdigit():
            digit += 1
        else:
            others += 1
    if letters / (len(x) - space) > 0.5:
        return "e"
    else:
        return "c"


class index:
    def POST(self):
        data = web.data()
        # 判断是否处于检测服务状态
        if data == b"":
            return
        x = ""
        ocr_r = ocr(data, "ch")
        return json.dumps((ocr_r))


class MyApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ("0.0.0.0", port))


if __name__ == "__main__":
    app = MyApplication(urls, globals())
    app.run(port=端口)
