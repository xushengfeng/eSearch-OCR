import platform
import os

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
