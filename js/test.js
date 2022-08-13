const x = require("./js");

start();

async function start() {
    await x.init({
        det_path: "./m/ch_PP-OCRv2_det_infer.onnx",
        rec_path: "./m/ch_PP-OCRv2_rec_infer.onnx",
        dic_path: "../assets/ppocr_keys_v1.txt",
    });
    let img = document.createElement("img");
    img.src = "../a.png";
    img.onload = () => {
        let canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        x.ocr(canvas.getContext("2d").getImageData(0, 0, img.width, img.height));
    };
}
