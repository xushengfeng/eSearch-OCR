# eSearch-OCR

本仓库是 [eSearch](https://github.com/xushengfeng/eSearch)的 OCR 服务依赖

目前支持本地 OCR（基于 PaddleOCR）和百度在线 OCR

百度 OCR 需要在官网申请。你可以直接修改`serve.py`里面的 key 变量值，并使用下面的[参数](#cli)来使用百度 OCR，截止 2022 年 1 月，百度 OCR 还可以[免费领取服务](https://cloud.baidu.com/doc/OCR/s/dk3iqnq51)

## 安装

### 源代码安装

下载目录并在里面运行

```shell
python setup.py
```

**注意**：目前 PaddleOCR 只支持 python<=3.9，如果你是 ArchLinux 用户，先用 AUR 安装 `python39`然后在虚拟环境下安装。如果你使用 Docker 运行，则不用配置环境。

### [Docker 安装](#Docker)

**提示**：点击跳转到下面的命令安装。在此之前需要准备 Docker，为了快速下载，可能需要更改仓库源，具体请自行搜索。

## 运行

### 源代码

```shell
python serve.py
```

建议在[eSearch](https://github.com/xushengfeng/eSearch)设置-自动运行命令中填入

```
cd [你下载的目录] && python serve.py
```

### Docker

```
docker run -it -p 8080:8080 xsf0root/e-search-service:latest
```

建议在[eSearch](https://github.com/xushengfeng/eSearch)设置-自动运行命令中填入

```
docker run --rm -it -p 8080:8080 xsf0root/e-search-service:latest

```

### cli

```
-c --check 自动纠错（仅适用于本地识别，可能耗时2秒左右）
-p --port  服务端口
-s --set   设置ocr服务提供者（默认local_ocr(本地OCR) 可设置为baidu）
-k --key   设置baidu ocr的ak和sk  [AK],[SK]  使用英文逗号连接
```

## 卸载

PaddleOcr 的模型数据存在用户目录的隐藏文件夹 paddleocr 下
配置文件夹里的service-installed请手动卸载
```
Windows:
    C:\Users\用户\eSearch\
Linux:
    ~/.config/eSearch/
macOS:
    ~/Library/Application Support/eSearch/
```