import fs from "node:fs";
import path from "node:path";
import { createImageData } from "canvas";
import DiffMatchPatch from "diff-match-patch";
import ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";
import { init, setOCREnv } from "../src/main";
import { checkAndWarn, getModelPath } from "./model_paths";
import { createCanvas, loadImage, setupCanvas } from "./setup";

const dmp = new DiffMatchPatch();

const VERSION = "v6_small";

function toImageData(canvasImageData: any): ImageData {
    return createImageData(
        new Uint8ClampedArray(canvasImageData.data),
        canvasImageData.width,
        canvasImageData.height,
    ) as unknown as ImageData;
}

describe("accuracy", async () => {
    function calcAccuracy(recognized: string, expected: string): number {
        const diff = dmp.diff_main(recognized, expected);
        let score = 0;
        for (const [op, text] of diff) {
            if (op === 0) {
                score += text.length;
            } else {
                score -= text.length * 0.5;
            }
        }
        return score / expected.length;
    }

    async function accuracy(imgPath: string, minAccuracy: number) {
        const img = await loadImage(imgPath);
        const canvas = createCanvas(img.width, img.height);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const imageData = toImageData(ctx.getImageData(0, 0, img.width, img.height));

        const result = await ocr.ocr(imageData);
        const recognizedText = result.parragraphs.map((p) => p.text).join("\n");
        const expectedText = fs.readFileSync(imgPath.replace(".svg", ".txt"), "utf-8").trim();

        const accuracy = calcAccuracy(recognizedText, expectedText);
        const sampleName = path.basename(imgPath, path.extname(imgPath));
        console.log(`${sampleName} 准确率: ${(accuracy * 100).toFixed(2)}%`);

        expect(accuracy).toBeGreaterThanOrEqual(minAccuracy);
    }

    if (!checkAndWarn(VERSION)) {
        console.warn("跳过测试：模型文件缺失");
        return;
    }

    const env = setupCanvas();
    setOCREnv(env);

    const paths = getModelPath(VERSION);
    const ocr = await init({
        det: {
            input: fs.readFileSync(paths.det).buffer,
            ratio: 0.75,
        },
        rec: {
            input: fs.readFileSync(paths.rec).buffer,
            decodeDic: fs.readFileSync(paths.dic).toString(),
        },
        ort,
        ortOption: {
            executionProviders: ["webgpu"],
        },
    });

    it("ch (中文)", () => accuracy("imgs/ch.svg", 0.98));
    it("en (英文)", () => accuracy("imgs/en.svg", 0.98));
    it("bg1", () => accuracy("imgs/bg1.svg", 0.98));
    it("bg2", () => accuracy("imgs/bg2.svg", 0.98));
    it("long", () => accuracy("imgs/long.svg", 0.98));
    it("long_small", () => accuracy("imgs/long_small.svg", 0.5));

    // layout_img
    it("1", () => accuracy("layout_img/1.svg", 0.99));
    it("2", () => accuracy("layout_img/2.svg", 0.99));
    it("3", () => accuracy("layout_img/3.svg", 0.99));
    it("4", () => accuracy("layout_img/4.svg", 0.99));
    it("5", () => accuracy("layout_img/5.svg", 0.99));
    it("6", () => accuracy("layout_img/6.svg", 0.99));
    it("7", () => accuracy("layout_img/7.svg", 0.99));
    it("8", () => accuracy("layout_img/8.svg", 0.95));
    it("9", () => accuracy("layout_img/9.svg", 0.99));
    it("10", () => accuracy("layout_img/10.svg", 0.98));
    it("11", () => accuracy("layout_img/11.svg", 0.97));
});
