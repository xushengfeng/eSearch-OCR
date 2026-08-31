const { init, setOCREnv } = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");
const { getModelPath, checkAndWarn } = require("./model_paths");

const { createCanvas, loadImage, createImageData } = require("canvas");

start();

async function start() {
    const version = "v6_small";
    if (!checkAndWarn(version)) {
        console.log("请先下载模型文件，参考上面的下载地址");
        return;
    }

    const paths = getModelPath(version);
    const detPath = paths.det;
    const recPath = paths.rec;
    const dicPath = paths.dic;
    const imgPath = "imgs/long_small.svg";

    setOCREnv({
        canvas: (w, h) => createCanvas(w, h),
        imageData: createImageData,
    });

    const localOCR = await init({
        det: {
            input: fs.readFileSync(detPath).buffer, // 可以直接传入path或buffer
        },
        rec: {
            input: recPath,
            decodeDic: fs.readFileSync(dicPath).toString(),
        },
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
