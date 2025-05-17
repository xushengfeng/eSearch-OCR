const { init } = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");

start();

async function start() {
    const pro = document.createElement("progress");
    document.body.append(pro);
    const modelBasePath = "./m/v4/";
    const localOcr = await init({
        detPath: `${modelBasePath}/ppocr_det.onnx`,
        recPath: `${modelBasePath}/ppocr_rec.onnx`,
        dic: fs.readFileSync("../assets/ppocr_keys_v1.txt").toString(),
        layoutDic: "text\ntitle\nfigure\nfigure_caption\ntable\ntable_caption\nheader\nfooter\nreference\nequation",
        dev: true,
        ort,
        onProgress: (t, a, n) => {
            if (t === "rec") {
                if (a !== 0) pro.value = n / a;
            }
            if (t === "det") {
                pro.value = 1;
            }
        },
        onDet: (d) => {
            console.log("det", d);
        },
        onRec: (i, r) => {
            console.log("r", r);
        },
    });
    pro.value = 0;
    const src = "imgs/ch.svg";
    // const src = "../c.png";
    const ocrResult = await localOcr.ocr(src);
    for (const i of ocrResult.parragraphs) {
        const p = document.createElement("p");
        p.innerText = i.text;
        document.body.append(p);
    }
}
