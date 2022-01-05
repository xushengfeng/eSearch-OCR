# encoding:utf-8

import requests
import web
import json
import base64

"""
通用文字识别（高精度版）
"""


def ocr(data, lang):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic" # accurate(_basic) general(_basic)
    # 二进制方式打开图片文件
    data = json.loads(str(data, encoding="utf-8"))
    image_string = data["image"]
    img = image_string
    # img = base64.b64decode(image_string)

    params = {"image": img, "detect_direction": "true", "paragraph": "true"}
    access_token = (
        "[你的百度OCR access_token]"
    )
    request_url = request_url + "?access_token=" + access_token
    headers = {"content-type": "application/x-www-form-urlencoded"}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        return response.json()


urls = ("/", "index")


class index:
    def POST(self):
        data = web.data()
        ocr_r = ocr(data, "ch")
        try:
            wr = []
            for idx, val in enumerate(ocr_r["paragraphs_result"]):
                wr.append({"words": ""})
            for idx, val in enumerate(ocr_r["paragraphs_result"]):
                for i in val["words_result_idx"]:
                    wr[idx]["words"] += ocr_r["words_result"][i]["words"]
            return json.dumps({"words_result": wr})
        except:
            print(ocr_r)


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
