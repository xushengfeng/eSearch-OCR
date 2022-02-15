import platform
import os


print('安装相关库')
os.system('pip install -r requirements.txt')
os.system('pip install paddlepaddle==2.2.2 -i https://mirror.baidu.com/pypi/simple')

print('初始化PaddleOCR')

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_gpu=False)  # 首次执行会自动下载模型文件

import requests
import shutil
kenlmpath = os.path.exists("kenlm-master")
if not kenlmpath:
    print('下载kenlm库中')
    file_url ="https://github.com/kpu/kenlm/archive/master.zip"
    res = requests.get(url=file_url)
    with open(r'kenlm-master.zip',mode='wb') as f:  # 需要用wb模式
        f.write(res.content)
    shutil.unpack_archive(filename=r'kenlm-master.zip',extract_dir=r'./',format='zip')


print("进入目录")
os.chdir("kenlm-master")
print('安装kenlm中')
os.system('python setup.py install')

pla = platform.system()
if pla == "Windows":
    path = os.path.expandvars(r"%APPDATA%\eSearch\service-installed")
elif pla == "Linux":
    path = os.path.expanduser("~") + "/.config/eSearch/service-installed"
elif pla == "Darrwin":
    path = (
        os.path.expanduser("~")
        + "/Library/Application Support/eSearch/service-installed"
    )
print(path)
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)