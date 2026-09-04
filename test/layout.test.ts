import fs from "node:fs";
import path from "node:path";
import { createImageData } from "canvas";
import ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";
import { type ReadingDir, analyzeLayout, detectReadingDir, init, initDet, setOCREnv } from "../src/main";
import { checkAndWarn, getModelPath } from "./model_paths";
import { createCanvas, loadImage, setupCanvas } from "./setup";

const VERSION = "v6_small";

function toImageData(canvasImageData: any): ImageData {
    return createImageData(
        new Uint8ClampedArray(canvasImageData.data),
        canvasImageData.width,
        canvasImageData.height,
    ) as unknown as ImageData;
}

describe("layout", () => {
    beforeAll(() => {
        const env = setupCanvas();
        setOCREnv(env);
    });

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

describe("layout direction", () => {
    beforeAll(() => {
        const env = setupCanvas();
        setOCREnv(env);
    });

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
        const angles = [0, 0, 0, 0];
        const result = detectReadingDir(angles);
        expect(result.readingDir.inline).toBe("lr");
        expect(result.readingDir.block).toBe("tb");
    });

    it("should detect vertical direction for 90 degree angles", () => {
        const angles = [90, 90, 90, 90];
        const result = detectReadingDir(angles);
        expect(result.readingDir.inline).toBe("tb");
        expect(result.readingDir.block).toBe("rl");
    });

    it("should detect direction with slight angle variation", () => {
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
