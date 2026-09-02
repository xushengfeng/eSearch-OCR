import fs from "node:fs";
import { createImageData } from "canvas";
import ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";
import { initDet, setOCREnv } from "../src/main";
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

describe("det", () => {
    beforeAll(() => {
        const env = setupCanvas();
        setOCREnv(env);
    });

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
