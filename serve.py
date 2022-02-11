import local_ocr
import argparse
import web
import os
import psutil
import time
import sys
import json


def write_pid():
    pid = os.getpid()
    fp = open("pid.log", 'w')
    fp.write(str(pid))
    fp.close()


def read_pid():
    if os.path.exists("pid.log"):
        fp = open("pid.log", 'r')
        pid = fp.read()
        fp.close()
        return pid
    else:
        return False


def is_ran():
    pid = int(read_pid())
    if pid in psutil.pids():
        return True
    else:
        write_pid()
        return False


if __name__ == "__main__":
    if is_ran():
        sys.exit()


开启纠错 = False
端口 = 8080

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--check", action="store_true")
parser.add_argument("-p", "--port", type=int)
args = parser.parse_args()
if args.check:
    开启纠错 = True
    local_ocr.correct(True)
if args.port:
    端口 = args.port


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


urls = ("/", "index")


class index:
    def POST(self):
        data = web.data()
        # 判断是否处于检测服务状态
        if data == b"":
            return
        x = ""
        with HiddenPrints():
            ocr_r = local_ocr.ocr(data, "ch")
        return json.dumps((ocr_r))


class MyApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ("0.0.0.0", port))


if __name__ == "__main__":
    app = MyApplication(urls, globals())
    app.run(port=端口)
