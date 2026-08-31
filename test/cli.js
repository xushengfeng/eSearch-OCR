#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const ort = require("onnxruntime-node");
const { init, setOCREnv } = require("../");
const { createCanvas, loadImage, createImageData } = require("canvas");
const { getModelPath, checkAndWarn } = require("./model_paths");

const VERSION = "v6_small";

async function main() {
    const args = process.argv.slice(2);
    const raw = args.includes("--raw");
    const imgArgs = args.filter((a) => !a.startsWith("--"));

    if (imgArgs.length === 0) {
        console.log("用法: node cli.js <图片路径> [--raw]");
        process.exit(1);
    }

    const imgPath = imgArgs[0];
    if (!fs.existsSync(imgPath)) {
        process.exit(1);
    }

    if (!checkAndWarn(VERSION)) {
        process.exit(1);
    }

    setOCREnv({
        canvas: (w, h) => createCanvas(w, h),
        imageData: createImageData,
    });

    const paths = getModelPath(VERSION);

    const docClsPath = path.join(paths.basePath, "doc_cls.onnx");
    const hasDocCls = fs.existsSync(docClsPath);

    const ocr = await init({
        det: {
            input: fs.readFileSync(paths.det).buffer,
        },
        rec: {
            input: fs.readFileSync(paths.rec).buffer,
            decodeDic: fs.readFileSync(paths.dic).toString(),
        },
        docCls: hasDocCls ? { input: docClsPath } : undefined,
        ort,
    });

    const img = await loadImage(imgPath);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);

    const result = await ocr.ocr(imageData);

    if (raw) {
        console.log(JSON.stringify(result));
    } else {
        for (const p of result.parragraphs) {
            console.log(p.text);
        }
    }
}

main().catch(() => {
    process.exit(1);
});
