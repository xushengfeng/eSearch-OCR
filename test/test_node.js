const x = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");

const { createCanvas, loadImage, createImageData } = require("canvas");

start();

async function start() {
    await x.init({
        detPath: "./m/v4/ppocr_det.onnx",
        recPath: "./m/v4/ppocr_rec.onnx",
        dic: fs.readFileSync("../assets/ppocr_keys_v1.txt").toString(),
        detShape: [640, 640],
        ort,
        canvas: (w, h) => createCanvas(w, h),
        imageData: createImageData,
    });
    const myimg = loadImage("3.jpg");

    myimg.then((img) => {
        const canvas = createCanvas(img.width, img.height);
        canvas.getContext("2d").drawImage(img, 0, 0);
        x.ocr(canvas.getContext("2d").getImageData(0, 0, img.width, img.height)).then((v) => {
            const tl = v.parragraphs.map((i) => i.text);
            console.log(tl.join("\n"));
        });
    });
}
