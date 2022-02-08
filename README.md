# eSearch-OCR

本仓库是 [eSearch](https://github.com/xushengfeng/eSearch)的OCR服务依赖

目前支持本地OCR（基于PaddleOCR）和百度在线OCR

## 安装

```shell
python setup.py
```

**注意**：目前PaddleOCR只支持python<=3.9，如果你是ArchLinux用户，先用AUR安装 `python39`然后在虚拟环境下安装

## 运行

```shell
python serve.py
```

建议在[eSearch](https://github.com/xushengfeng/eSearch)设置-自动运行命令中填入

```
cd [你下载的目录] && python serve.py
```
