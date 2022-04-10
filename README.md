# eSearch-OCR

本仓库是 [eSearch](https://github.com/xushengfeng/eSearch)的 OCR 服务依赖

目前支持本地 OCR（基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)）

[PaddleOCR License](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/LICENSE)

[paddle 预测库](https://paddle-inference.readthedocs.io/en/latest/user_guides/download_lib.html)

[编译文档](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.3/deploy/cpp_infer/)

## 旧版本卸载

### docker 镜像删除

```
docker rmi e-search-service:latest
```

PaddleOCR 的模型数据存在用户目录的隐藏文件夹 paddleocr 下

```
Windows:
    C:\Users\用户\paddleocr
Linux:
    ~/.paddleocr
macOS:
    ~/。paddleocr
```

Python 模块请执行用`pip uninstall [包名]`按需卸载
这个项目用到的包：

```
web.py
numpy
paddleocr
paddlepaddle
jieba
pypinyin
six
psutil
requests
shapely
kenlm
```
