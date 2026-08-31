import { describe, expect, it, beforeAll } from "vitest";
import fs from "node:fs";
import path from "node:path";
import ort from "onnxruntime-node";
import { init, setOCREnv, initDet, initRec, initDocDirCls, warpDet, analyzeLayout } from "../";
import { getModelPath, checkAndWarn } from "./model_paths";
import { setupCanvas, loadImage, createCanvas } from "./setup";
import { createImageData } from "canvas";

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
            const rec = await initRec({
                input: fs.readFileSync(paths.rec).buffer,
                decodeDic: fs.readFileSync(paths.dic).toString(),
                ort,
            });

            const img = await loadImage(path.join("layout_img", file));
            const canvas = createCanvas(img.width, img.height);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            const imageData = toImageData(ctx.getImageData(0, 0, img.width, img.height));

            const detResult = await det.det(imageData);
            const recResult = await rec.rec(detResult);
            const layoutResult = analyzeLayout(recResult);

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
});
