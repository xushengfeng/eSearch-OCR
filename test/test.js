const { init } = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");

start();

async function start() {
    const pro = document.createElement("progress");
    document.body.append(pro);
    const modelBasePath = "./m/v5/";
    const localOcr = await init({
        det: { input: `${modelBasePath}/ppocr_v5_mobile_det.onnx` },
        rec: {
            input: `${modelBasePath}/ppocr_v5_mobile_rec.onnx`,
            decodeDic: fs.readFileSync("../assets/ppocrv5_dict.txt").toString(),
            on: (i, r, t) => {
                pro.value = (i + 1) / t;
            },
            optimize: {
                space: false,
            },
        },
        dev: true,
        ort,
    });
    pro.value = 0;
    const src = "imgs/bg1.svg";
    // const src = "../c.png";
    const ocrResult = await localOcr.ocr(src);
    for (const i of ocrResult.parragraphs) {
        const p = document.createElement("p");
        p.innerText = i.text;
        document.body.append(p);
    }
}
