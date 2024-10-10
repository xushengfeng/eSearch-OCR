const x = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");

start();

async function start() {
    const pro = document.createElement("progress");
    document.body.append(pro);
    await x.init({
        detPath: "./m/v4/ppocr_det.onnx",
        recPath: "./m/v4/ppocr_rec.onnx",
        dic: fs.readFileSync("../assets/ppocr_keys_v1.txt").toString(),
        layoutDic: "text\ntitle\nfigure\nfigure_caption\ntable\ntable_caption\nheader\nfooter\nreference\nequation",
        dev: true,
        detShape: [640, 640],
        ort,
        onProgress: (t, a, n) => {
            if (t === "rec") {
                console.log(n / a);
                pro.value = n / a;
            }
            if (t === "det") {
                pro.value = 1;
            }
        },
    });
    pro.value = 0;
    // const src = "imgs/ch.svg";
    const src = "../c.png";
    x.ocr(src).then((v) => {
        for (const i of v.parragraphs) {
            const p = document.createElement("p");
            p.innerText = i.text;
            document.body.append(p);
        }
    });
}
