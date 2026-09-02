import { describe, expect, it, beforeAll } from "vitest";
import fs from "node:fs";
import path from "node:path";
import ort from "onnxruntime-node";
import DiffMatchPatch from "diff-match-patch";
import {
    init,
    setOCREnv,
    initDet,
    initRec,
    initDocDirCls,
    analyzeLayout,
    detectReadingDir,
    type ReadingDir,
} from "../src/main";
import { getModelPath, checkAndWarn } from "./model_paths";
import { setupCanvas, loadImage, createCanvas } from "./setup";
import { createImageData } from "canvas";

const dmp = new DiffMatchPatch();

const VERSION = "v6_small";

function toImageData(canvasImageData: any): ImageData {
    return createImageData(
        new Uint8ClampedArray(canvasImageData.data),
        canvasImageData.width,
        canvasImageData.height,
    ) as unknown as ImageData;
}

describe("OCR", () => {
    beforeAll(() => {
        const env = setupCanvas();
        setOCREnv(env);
    });

    describe("det", () => {
        it("should detect text regions", async () => {
            if (!checkAndWarn(VERSION)) {
                console.warn("跳过测试：模型文件缺失");
                return;
            }

            const paths = getModelPath(VERSION);
            const det = await initDet({
                input: fs.readFileSync(paths.det).buffer,
                ort,
            });

            const img = await loadImage("imgs/long_small.svg");
            const canvas = createCanvas(img.width, img.height);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const imageData = toImageData(ctx.getImageData(0, 0, img.width, img.height));

            const result = await det.det(imageData);

            expect(result).toBeDefined();
            expect(Array.isArray(result)).toBe(true);
            expect(result.length).toBeGreaterThan(0);

            for (const item of result) {
                expect(item.box).toBeDefined();
                expect(item.box.length).toBe(4);
                expect(item.img).toBeDefined();
                expect(item.img.data).toBeDefined();
                expect(item.img.width).toBeGreaterThan(0);
                expect(item.img.height).toBeGreaterThan(0);
            }
        });
    });

    describe("rec", () => {
        it("should recognize text from detected regions", async () => {
            if (!checkAndWarn(VERSION)) {
                console.warn("跳过测试：模型文件缺失");
                return;
            }

            const paths = getModelPath(VERSION);
            const det = await initDet({
                input: fs.readFileSync(paths.det).buffer,
                ort,
            });

            const rec = await initRec({
                input: fs.readFileSync(paths.rec).buffer,
                decodeDic: fs.readFileSync(paths.dic).toString(),
                ort,
            });

            const img = await loadImage("imgs/long_small.svg");
            const canvas = createCanvas(img.width, img.height);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const imageData = toImageData(ctx.getImageData(0, 0, img.width, img.height));

            const detResult = await det.det(imageData);
            const recResult = await rec.rec(detResult);

            expect(recResult).toBeDefined();
            expect(Array.isArray(recResult)).toBe(true);
            expect(recResult.length).toBeGreaterThan(0);

            for (const item of recResult) {
                expect(item.text).toBeDefined();
                expect(typeof item.text).toBe("string");
                expect(item.mean).toBeGreaterThan(0);
                expect(item.box).toBeDefined();
            }

            const allText = recResult.map((i) => i.text).join("");
            expect(allText).toContain("你好");
        });
    });

    describe("layout", () => {
        it("should perform layout analysis", async () => {
            if (!checkAndWarn(VERSION)) {
                console.warn("跳过测试：模型文件缺失");
                return;
            }

            const paths = getModelPath(VERSION);
            const ocr = await init({
                det: {
                    input: fs.readFileSync(paths.det).buffer,
                },
                rec: {
                    input: fs.readFileSync(paths.rec).buffer,
                    decodeDic: fs.readFileSync(paths.dic).toString(),
                },
                ort,
            });

            const img = await loadImage("imgs/long_small.svg");
            const canvas = createCanvas(img.width, img.height);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const imageData = toImageData(ctx.getImageData(0, 0, img.width, img.height));

            const result = await ocr.ocr(imageData);

            expect(result).toBeDefined();
            expect(result.parragraphs).toBeDefined();
            expect(Array.isArray(result.parragraphs)).toBe(true);

            for (const p of result.parragraphs) {
                expect(p.text).toBeDefined();
                expect(typeof p.text).toBe("string");
                expect(p.box).toBeDefined();
            }
        });
    });

    describe("docCls", () => {
        it("should classify document direction", async () => {
            if (!checkAndWarn(VERSION)) {
                console.warn("跳过测试：模型文件缺失");
                return;
            }

            const paths = getModelPath(VERSION);

            const docClsPath = path.join(paths.basePath, "..", "doc_cls.onnx");
            if (!fs.existsSync(docClsPath)) {
                console.warn("跳过 docCls 测试：doc_cls.onnx 模型缺失");
                return;
            }

            const docCls = await initDocDirCls({
                input: docClsPath,
                ort,
            });

            const img = await loadImage("imgs/long_small.svg");
            const canvas = createCanvas(img.width, img.height);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const imageData = toImageData(ctx.getImageData(0, 0, img.width, img.height));

            const result = await docCls.docCls(imageData);

            expect(result).toBeDefined();
            expect(typeof result).toBe("number");
            expect([0, 90, 180, 270]).toContain(result);
        });
    });

    describe("layout direction", () => {
        async function checkDirection(file: string, isVertical: boolean) {
            if (!checkAndWarn(VERSION)) {
                console.warn("跳过测试：模型文件缺失");
                return;
            }

            const paths = getModelPath(VERSION);
            const det = await initDet({
                input: fs.readFileSync(paths.det).buffer,
                ort,
            });

            const img = await loadImage(path.join("layout_img", file));
            const canvas = createCanvas(img.width, img.height);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const imageData = toImageData(ctx.getImageData(0, 0, img.width, img.height));

            const detResult = await det.det(imageData);
            const layoutResult = analyzeLayout(detResult.map((i, n) => ({ ...i, text: n.toString(), mean: 1 })));

            const inlineAngle = layoutResult.angle.reading.inline;
            const blockAngle = layoutResult.angle.reading.block;

            const normalizeAngle = (a: number) => ((a % 360) + 360) % 360;
            const normInline = normalizeAngle(inlineAngle);
            const normBlock = normalizeAngle(blockAngle);

            const distInlineTo0 = Math.min(normInline, 360 - normInline);
            const distInlineTo90 = Math.abs(normInline - 90);
            const detectedVertical = distInlineTo90 < distInlineTo0;

            const distBlockTo90 = Math.abs(normBlock - 90);
            const distBlockTo180 = Math.abs(normBlock - 180);
            const detectedBlock180 = distBlockTo180 < distBlockTo90;

            console.log(`inline=${inlineAngle.toFixed(1)}° block=${blockAngle.toFixed(1)}°`);

            expect(detectedVertical).toBe(isVertical);
            expect(detectedBlock180).toBe(isVertical);
        }

        it("1.svg is horizontal", () => checkDirection("1.svg", false));
        it("2.svg is horizontal", () => checkDirection("2.svg", false));
        it("3.svg is horizontal", () => checkDirection("3.svg", false));
        it("4.svg is horizontal", () => checkDirection("4.svg", false));
        it("5.svg is horizontal", () => checkDirection("5.svg", false));
        it("6.svg is horizontal", () => checkDirection("6.svg", false));
        it("7.svg is horizontal", () => checkDirection("7.svg", false));
        it("8.svg is vertical", () => checkDirection("8.svg", true));
        it("9.svg is horizontal", () => checkDirection("9.svg", false));
        it("10.svg is vertical", () => checkDirection("10.svg", true));
        it("11.svg is vertical", () => checkDirection("11.svg", true));
    });

    describe("detectReadingDir", () => {
        it("should return default direction for empty input", () => {
            const result = detectReadingDir([]);
            expect(result.readingDir).toEqual({ block: "tb", inline: "lr" });
            expect(result.angle.reading.inline).toBe(0);
            expect(result.angle.reading.block).toBe(90);
        });

        it("should detect horizontal direction for 0 degree angles", () => {
            // 横排文本 inline 角度接近 0 度
            const angles = [0, 0, 0, 0];
            const result = detectReadingDir(angles);
            expect(result.readingDir.inline).toBe("lr");
            expect(result.readingDir.block).toBe("tb");
        });

        it("should detect vertical direction for 90 degree angles", () => {
            // 竖排文本 inline 角度接近 90 度
            const angles = [90, 90, 90, 90];
            const result = detectReadingDir(angles);
            expect(result.readingDir.inline).toBe("tb");
            expect(result.readingDir.block).toBe("rl");
        });

        it("should detect direction with slight angle variation", () => {
            // 允许轻微角度偏差
            const angles = [85, 88, 92, 90];
            const result = detectReadingDir(angles);
            expect(result.readingDir.inline).toBe("tb");
            expect(result.readingDir.block).toBe("rl");
        });

        it("有其他方向", () => {
            const angles = [0, 88, 92, 90, 91, 2];
            const result = detectReadingDir(angles);
            expect(result.readingDir.inline).toBe("tb");
            expect(result.readingDir.block).toBe("rl");
        });

        it("存在反向", () => {
            const angles = [91, 89, 270, 269, 271];
            // 应该tb而不是bt，因为提示限制
            const result = detectReadingDir(angles);
            expect(result.readingDir.inline).toBe("tb");
            expect(result.readingDir.block).toBe("rl");
        });

        it("should respect custom docDirs", () => {
            const angles = [0, 0, 0];
            const customDirs = [{ block: "tb", inline: "lr" }] as ReadingDir[];
            const result = detectReadingDir(angles, customDirs);
            expect(result.readingDir.inline).toBe("lr");
            expect(result.readingDir.block).toBe("tb");
        });
    });

    describe("accuracy", () => {
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
            if (!checkAndWarn(VERSION)) {
                console.warn("跳过测试：模型文件缺失");
                return;
            }

            const paths = getModelPath(VERSION);
            const ocr = await init({
                det: {
                    input: fs.readFileSync(paths.det).buffer,
                    ratio: 0.75
                },
                rec: {
                    input: fs.readFileSync(paths.rec).buffer,
                    decodeDic: fs.readFileSync(paths.dic).toString(),
                },
                ort,
            });

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
        it("8", () => accuracy("layout_img/8.svg", 0.99));
        it("9", () => accuracy("layout_img/9.svg", 0.99));
        it("10", () => accuracy("layout_img/10.svg", 0.99));
        it("11", () => accuracy("layout_img/11.svg", 0.99));
    });
});
