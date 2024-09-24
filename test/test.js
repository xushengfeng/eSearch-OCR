const x = require("../");
const fs = require("fs");
const ort = require("onnxruntime-node");

start();

async function start() {
    await x.init({
        detPath: "./m/v4/ppocr_det.onnx",
        recPath: "./m/v4/ppocr_rec.onnx",
        dic: fs.readFileSync("../assets/ppocr_keys_v1.txt").toString(),
        layoutDic: "text\ntitle\nfigure\nfigure_caption\ntable\ntable_caption\nheader\nfooter\nreference\nequation",
        dev: true,
        detShape: [640, 640],
        ort,
    });
    let img = document.createElement("img");
    img.src = "../c.png";
    img.onload = () => {
        let canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        x.ocr(canvas.getContext("2d").getImageData(0, 0, img.width, img.height)).then((v) => {
            let tl = [];
            for (let i of v.parse) {
                tl.push(i.text);
            }
            let p = document.createElement("p");
            p.innerText = tl.join("\n");
            document.body.append(p);
        });
    };
}
