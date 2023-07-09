const x = require("../");
const fs = require("fs");

start();

async function start() {
    await x.init({
        detPath: "./m/ch_PP-OCRv3_det_infer.onnx",
        recPath: "./m/ch_PP-OCRv3_rec_infer.onnx",
        dic: fs.readFileSync("../assets/ppocr_keys_v1.txt").toString(),
        dev: true,
        node: true,
    });
    let img = document.createElement("img");
    img.src = "../a.png";
    img.onload = () => {
        let canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        x.ocr(canvas.getContext("2d").getImageData(0, 0, img.width, img.height)).then((v) => {
            let tl = [];
            for (let i of v) {
                tl.push(i.text);
            }
            let p = document.createElement("p");
            p.innerText = tl.join("\n");
            document.body.append(p);
        });
    };
}
