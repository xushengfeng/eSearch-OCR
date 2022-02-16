# encoding:utf-8

import requests
import json
import base64

"""
通用文字识别（高精度版）
"""

access_token = ''


def key(key):
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    ak = key.split(',')[0]
    sk = key.split(',')[1]
    host = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=" + \
        ak+"&client_secret="+sk
    response = requests.get(host)
    if response:
        global access_token
        access_token = response.json()["access_token"]


def ocr(data):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
    # 二进制方式打开图片文件
    data = json.loads(str(data, encoding="utf-8"))
    image_string = data["image"]
    img = image_string

    params = {"image": img, "detect_direction": "true", "paragraph": "true"}
    request_url = request_url + "?access_token=" + access_token
    headers = {"content-type": "application/x-www-form-urlencoded"}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        ocr_r = response.json()
        try:
            wr = []
            for idx, val in enumerate(ocr_r["paragraphs_result"]):
                wr.append({"words": ""})
            for idx, val in enumerate(ocr_r["paragraphs_result"]):
                for i in val["words_result_idx"]:
                    wr[idx]["words"] += ocr_r["words_result"][i]["words"]
            return {"words_result": wr}
        except:
            print(ocr_r)
