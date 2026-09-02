#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const ort = require("onnxruntime-node");
const { initRec, setOCREnv } = require("../");
const { createCanvas, createImageData } = require("canvas");
const { getModelPath, checkAndWarn } = require("./model_paths");

const VERSION = "v6_small";

const MARGIN_RATIOS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];

const CH_TEXT = "你好世界这是测试文字";
const EN_TEXT = "Hello World Test Text";

function toImageData(canvas) {
    const ctx = canvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return createImageData(new Uint8ClampedArray(imgData.data), imgData.width, imgData.height);
}

function createTextImage(text, fontSize, isVertical) {
    const canvas = createCanvas(1, 1);
    const ctx = canvas.getContext("2d");
    ctx.font = `${fontSize}px sans-serif`;

    const metrics = ctx.measureText(text);
    const textWidth = Math.ceil(metrics.width);
    const textHeight = Math.ceil(fontSize);

    let imgWidth, imgHeight;

    if (isVertical) {
        imgWidth = textHeight;
        imgHeight = textWidth;
    } else {
        imgWidth = textWidth;
        imgHeight = textHeight;
    }

    const textCanvas = createCanvas(imgWidth, imgHeight);
    const textCtx = textCanvas.getContext("2d");

    textCtx.fillStyle = "white";
    textCtx.fillRect(0, 0, imgWidth, imgHeight);

    textCtx.fillStyle = "black";
    textCtx.font = `${fontSize}px sans-serif`;
    textCtx.textBaseline = "top";

    if (isVertical) {
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            textCtx.fillText(char, 0, i * fontSize);
        }
    } else {
        textCtx.fillText(text, 0, 0);
    }

    return toImageData(textCanvas);
}

function addMargin(img, hRatio, vRatio) {
    const newWidth = Math.max(1, Math.floor(img.width * (1 + hRatio * 2)));
    const newHeight = Math.max(1, Math.floor(img.height * (1 + vRatio * 2)));

    const canvas = createCanvas(newWidth, newHeight);
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, newWidth, newHeight);

    const offsetX = Math.floor((newWidth - img.width) / 2);
    const offsetY = Math.floor((newHeight - img.height) / 2);

    const imgCanvas = createCanvas(img.width, img.height);
    const imgCtx = imgCanvas.getContext("2d");
    imgCtx.putImageData(img, 0, 0);

    ctx.drawImage(imgCanvas, offsetX, offsetY);

    return toImageData(canvas);
}

function calculateAccuracy(recognized, expected) {
    if (!recognized || !expected) return 0;

    let correct = 0;
    const minLen = Math.min(recognized.length, expected.length);

    for (let i = 0; i < minLen; i++) {
        if (recognized[i] === expected[i]) {
            correct++;
        }
    }

    return correct / expected.length;
}

async function main() {
    const args = process.argv.slice(2);

    if (args.includes("--help") || args.includes("-h")) {
        console.log("用法: node analyze-rec-margin.js [选项]");
        console.log("");
        console.log("分析 rec 模型在不同边距比例下的准确率");
        console.log("文字图片会自动添加指定比例的白色边距，然后测试识别准确率");
        console.log("");
        console.log("选项:");
        console.log("  --help, -h     显示帮助信息");
        console.log("  --vertical     测试竖排文本（文字向左旋转90度）");
        console.log("  --lang=LANG    指定语言 (ch|en|both，默认 both)");
        console.log("  --csv          输出 CSV 格式结果");
        console.log("");
        console.log("示例:");
        console.log("  node analyze-rec-margin.js");
        console.log("  node analyze-rec-margin.js --vertical");
        console.log("  node analyze-rec-margin.js --lang=ch");
        console.log("  node analyze-rec-margin.js --csv > result.csv");
        process.exit(0);
    }

    const isVertical = args.includes("--vertical");
    const langArg = args.find((a) => a.startsWith("--lang="));
    const lang = langArg ? langArg.split("=")[1] : "both";
    const outputCSV = args.includes("--csv");

    if (!checkAndWarn(VERSION)) {
        console.error("模型文件缺失");
        process.exit(1);
    }

    setOCREnv({
        canvas: (w, h) => createCanvas(w, h),
        imageData: createImageData,
    });

    const paths = getModelPath(VERSION);

    const rec = await initRec({
        input: fs.readFileSync(paths.rec).buffer,
        decodeDic: fs.readFileSync(paths.dic).toString(),
        ort,
    });

    const testCases = [];

    if (lang === "ch" || lang === "both") {
        testCases.push({ name: "中文", text: CH_TEXT, fontSize: 24 });
    }

    if (lang === "en" || lang === "both") {
        testCases.push({ name: "英文", text: EN_TEXT, fontSize: 20 });
    }

    if (!outputCSV) {
        console.log("开始分析 rec 边距比例...");
        console.log(`模式: ${isVertical ? "竖排" : "横排"}`);
        console.log(`语言: ${lang}`);
        console.log("");
    }

    const allResults = [];

    for (const testCase of testCases) {
        if (!outputCSV) {
            console.log(`=== 测试 ${testCase.name} ===`);
            console.log(`测试文本: ${testCase.text}`);
        }

        const baseImg = createTextImage(testCase.text, testCase.fontSize, isVertical);
        const results = [];

        for (const hRatio of MARGIN_RATIOS) {
            for (const vRatio of MARGIN_RATIOS) {
                const paddedImg = addMargin(baseImg, hRatio, vRatio);

                const detResult = [
                    {
                        box: [
                            [0, 0],
                            [paddedImg.width, 0],
                            [paddedImg.width, paddedImg.height],
                            [0, paddedImg.height],
                        ],
                        img: paddedImg,
                    },
                ];

                try {
                    const result = await rec.rec(detResult);
                    const recognizedText = result[0]?.text || "";
                    const accuracy = calculateAccuracy(recognizedText, testCase.text);

                    results.push({
                        hRatio,
                        vRatio,
                        accuracy,
                        text: recognizedText,
                        lang: testCase.name,
                        isVertical,
                    });
                } catch (e) {
                    results.push({
                        hRatio,
                        vRatio,
                        accuracy: 0,
                        text: "",
                        lang: testCase.name,
                        isVertical,
                    });
                }
            }
        }

        allResults.push(...results);

        if (!outputCSV) {
            results.sort((a, b) => b.accuracy - a.accuracy);

            const best = results[0];
            console.log(`最佳边距比例: 水平=${(best.hRatio * 100).toFixed(0)}%, 垂直=${(best.vRatio * 100).toFixed(0)}%`);
            console.log(`准确率: ${(best.accuracy * 100).toFixed(2)}%`);
            console.log(`识别文本: ${best.text}`);
            console.log("");

            console.log("Top 5 结果:");
            for (let i = 0; i < Math.min(5, results.length); i++) {
                const r = results[i];
                console.log(`  ${i + 1}. 水平=${(r.hRatio * 100).toFixed(0)}%, 垂直=${(r.vRatio * 100).toFixed(0)}% => ${(r.accuracy * 100).toFixed(2)}%`);
            }
            console.log("");
        }
    }

    if (outputCSV) {
        console.log("language,vertical,horizontal_ratio,vertical_ratio,accuracy,recognized_text");
        for (const r of allResults) {
            const escapedText = `"${r.text.replace(/"/g, '""')}"`;
            console.log(`${r.lang},${r.isVertical},${r.hRatio},${r.vRatio},${r.accuracy},${escapedText}`);
        }
    } else {
        console.log("分析完成!");
    }
}

main().catch((err) => {
    console.error("错误:", err);
    process.exit(1);
});
