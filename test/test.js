const { init } = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");

start();

async function start() {
    const pro = document.createElement("progress");
    document.body.append(pro);
    const modelBasePath = "./m/v4/";
    const localOcr = await init({
        det: { input: `${modelBasePath}/ppocr_det.onnx` },
        rec: {
            input: `${modelBasePath}/ppocr_rec.onnx`,
            decodeDic: fs.readFileSync("../assets/ppocr_keys_v1.txt").toString(),
            on: (i, r, t) => {
                pro.value = (i + 1) / t;
            },
        },
        dev: true,
        ort,
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
