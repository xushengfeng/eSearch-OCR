const { init, setOCREnv } = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");

const { createCanvas, loadImage, createImageData } = require("canvas");

start();

async function start() {
    const detPath = "./m/v4/ppocr_det.onnx";
    const recPath = "./m/v4/ppocr_rec.onnx";
    const dicPath = "../assets/ppocr_keys_v1.txt";
    const imgPath = "imgs/long_small.svg";

    setOCREnv({
        canvas: (w, h) => createCanvas(w, h),
        imageData: createImageData,
    });

    const localOCR = await init({
        detPath: detPath,
        recPath: recPath,
        dic: fs.readFileSync(dicPath).toString(),
        ort,
    });
    const img = await loadImage(imgPath);

    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const ocrResult = await localOCR.ocr(ctx.getImageData(0, 0, img.width, img.height));
    const tl = ocrResult.parragraphs.map((i) => i.text);
    console.log(tl.join("\n"));
}
