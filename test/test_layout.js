const x = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");

start();

async function start() {
    const pro = document.createElement("progress");
    document.body.append(pro);
    const modelBasePath = "./m/v4/";
    await x.init({
        detPath: `${modelBasePath}/ppocr_det.onnx`,
        recPath: `${modelBasePath}/ppocr_v4_rec_doc.onnx`,
        dic: fs.readFileSync("../assets/ppocrv4_doc_dict.txt").toString(),
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
    });
    pro.value = 0;
    const src = "../test_img/eSearch-2025-04-27-18-39-34-295.png";
    const detResult = await x.det(src);
    const mainLine = detResult.map((i, n) => ({ ...i, text: n.toString(), mean: 1 }));
    const l = x.analyzeLayout(mainLine);
}
