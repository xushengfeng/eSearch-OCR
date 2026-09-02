import fs from "node:fs";
import path from "node:path";
import { createImageData } from "canvas";
import ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";
import { initDocDirCls, setOCREnv } from "../src/main";
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

describe("docCls", () => {
    beforeAll(() => {
        const env = setupCanvas();
        setOCREnv(env);
    });

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
