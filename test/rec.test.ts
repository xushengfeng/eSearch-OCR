import fs from "node:fs";
import { createImageData } from "canvas";
import ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";
import { initDet, initRec, setOCREnv } from "../src/main";
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

describe("rec", () => {
    beforeAll(() => {
        const env = setupCanvas();
        setOCREnv(env);
    });

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
