const { init, loadImg, warpDet } = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");
const { getModelPath, checkAndWarn } = require("./model_paths");

start();

async function start() {
    const pro = document.createElement("progress");
    document.body.append(pro);

    const version = "v6_small";
    if (!checkAndWarn(version)) {
        console.log("请先下载模型文件，参考上面的下载地址");
        return;
    }

    const paths = getModelPath(version);
    const localOcr = await init({
        det: { input: paths.det },
        rec: {
            input: paths.rec,
            decodeDic: fs.readFileSync(paths.dic).toString(),
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
    const d = await localOcr.det(await loadImg(src));
    const r = await localOcr.recRaw(d);
    console.log(r);
}
