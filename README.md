# eSearch-OCR

本仓库是 [eSearch](https://github.com/xushengfeng/eSearch)的OCR服务依赖

目前支持本地OCR（基于PaddleOCR）和百度在线OCR

百度OCR需要在官网申请。你可以直接修改里面的“[官网获取的AK]”“{官网获取的SK]”来使用百度OCR，截止2022年1月，百度OCR还可以[免费领取服务](https://cloud.baidu.com/doc/OCR/s/dk3iqnq51)

## 安装

```shell
python setup.py
```

**注意**：目前PaddleOCR只支持python<=3.9，如果你是ArchLinux用户，先用AUR安装 `python39`然后在虚拟环境下安装

## 运行

```shell
python serve.py
	-c --check 自动纠错（仅适用于本地识别，可能耗时2秒左右）
	-p --port  服务端口
	-s --set   设置ocr服务提供者（默认local_ocr(本地OCR) 可设置为baidu）
```

建议在[eSearch](https://github.com/xushengfeng/eSearch)设置-自动运行命令中填入

```
cd [你下载的目录] && python serve.py
```
