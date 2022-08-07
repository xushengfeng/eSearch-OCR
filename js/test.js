const x = require("./js");

const fs = require("fs");
const ort = require("onnxruntime-node");
start();

async function start() {
    const det = await ort.InferenceSession.create("./m/ch_PP-OCRv2_det_infer.onnx");
    const rec = await ort.InferenceSession.create("./m/ch_PP-OCRv2_rec_infer.onnx");
    let dic = fs.readFileSync("../assets/ppocr_keys_v1.txt").toString().split("\n");
    let img = document.createElement("img");
    img.src = "../a.png";
    img.onload = () => {
        let canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        x(canvas.getContext("2d").getImageData(0, 0, img.width, img.height), det, rec, dic);
    };
}
