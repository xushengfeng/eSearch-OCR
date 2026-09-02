import { createCanvas, createImageData } from "canvas";
import { describe, expect, it } from "vitest";
import { matchBestBox } from "../src/main";

type BoxType = [[number, number], [number, number], [number, number], [number, number]];
type color = [number, number, number];

function toImageData(canvasImageData: { data: Uint8ClampedArray; width: number; height: number }): ImageData {
    return createImageData(
        new Uint8ClampedArray(canvasImageData.data),
        canvasImageData.width,
        canvasImageData.height,
    ) as unknown as ImageData;
}

function createTextImage(
    width: number,
    height: number,
    bgColor: string,
    textColor: string,
    text: string,
    x: number,
    y: number,
    fontSize = 20,
): ImageData {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");

    // 绘制背景
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);

    // 绘制文字
    ctx.fillStyle = textColor;
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillText(text, x, y);

    return toImageData(ctx.getImageData(0, 0, width, height));
}

describe("matchBestBox", () => {
    it("should match text color and adjust box boundaries", () => {
        const width = 200;
        const height = 100;
        const img = createTextImage(width, height, "white", "black", "Hello", 50, 50);
        const textEdgeColor: color = [0, 0, 0];

        const box: BoxType = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ];

        const result = matchBestBox(box, img, textEdgeColor);

        expect(result).toBeDefined();
        expect(result.length).toBe(4);

        // 边界框应该被调整以包含文字
        expect(result[0][0]).toBeGreaterThan(0);
        expect(result[0][1]).toBeGreaterThan(0);
        expect(result[2][0]).toBeLessThan(width);
        expect(result[2][1]).toBeLessThan(height);
    });

    it("should handle text at edges", () => {
        const width = 200;
        const height = 100;
        // 文字在顶部
        const img = createTextImage(width, height, "white", "black", "Top", 10, 20);
        const textEdgeColor: color = [0, 0, 0];

        const box: BoxType = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ];

        const result = matchBestBox(box, img, textEdgeColor);

        expect(result).toBeDefined();
        // 边界框应该调整到顶部
        expect(result[0][1]).toBeLessThan(height / 2);
    });

    it("should handle no matching text color", () => {
        const width = 200;
        const height = 100;
        // 全白图像，没有黑色文字
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);
        const img = toImageData(ctx.getImageData(0, 0, width, height));

        const textEdgeColor: color = [0, 0, 0];

        const box: BoxType = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ];

        const result = matchBestBox(box, img, textEdgeColor);

        expect(result).toBeDefined();
        expect(result.length).toBe(4);
    });

    it("should handle colored text on colored background", () => {
        const width = 200;
        const height = 100;
        const img = createTextImage(width, height, "blue", "red", "Color", 60, 60);
        const textEdgeColor: color = [255, 0, 0];

        const box: BoxType = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ];

        const result = matchBestBox(box, img, textEdgeColor);

        expect(result).toBeDefined();
        expect(result.length).toBe(4);

        // 边界框应该调整到文字区域
        expect(result[0][0]).toBeGreaterThan(0);
        expect(result[0][1]).toBeGreaterThan(0);
        expect(result[2][0]).toBeLessThan(width);
        expect(result[2][1]).toBeLessThan(height);
    });

    it("should handle text in different positions", () => {
        const width = 200;
        const height = 100;
        // 文字在右下角
        const img = createTextImage(width, height, "white", "black", "Bottom", 120, 80);
        const textEdgeColor: color = [0, 0, 0];

        const box: BoxType = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ];

        const result = matchBestBox(box, img, textEdgeColor);

        expect(result).toBeDefined();
        expect(result.length).toBe(4);

        // 边界框应该调整以包含文字
        expect(result[0][0]).toBeGreaterThanOrEqual(0);
        expect(result[0][1]).toBeGreaterThanOrEqual(0);
        expect(result[2][0]).toBeLessThanOrEqual(width);
        expect(result[2][1]).toBeLessThanOrEqual(height);
    });

    it("should preserve box structure", () => {
        const width = 200;
        const height = 100;
        const img = createTextImage(width, height, "white", "black", "Test", 80, 50);
        const textEdgeColor: color = [0, 0, 0];

        const box: BoxType = [
            [10, 10],
            [190, 10],
            [190, 90],
            [10, 90],
        ];

        const result = matchBestBox(box, img, textEdgeColor);

        expect(result).toBeDefined();
        expect(Array.isArray(result)).toBe(true);
        expect(result.length).toBe(4);

        for (const point of result) {
            expect(Array.isArray(point)).toBe(true);
            expect(point.length).toBe(2);
            expect(typeof point[0]).toBe("number");
            expect(typeof point[1]).toBe("number");
        }
    });

    it("should handle multiple lines of text", () => {
        const width = 200;
        const height = 100;
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");

        // 白色背景
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        // 多行黑色文字
        ctx.fillStyle = "black";
        ctx.font = "16px sans-serif";
        ctx.fillText("Line 1", 20, 30);
        ctx.fillText("Line 2", 20, 50);
        ctx.fillText("Line 3", 20, 70);

        const img = toImageData(ctx.getImageData(0, 0, width, height));
        const textEdgeColor: color = [0, 0, 0];

        const box: BoxType = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ];

        const result = matchBestBox(box, img, textEdgeColor);

        expect(result).toBeDefined();
        expect(result.length).toBe(4);

        // 边界框应该包含所有文字
        expect(result[0][0]).toBeGreaterThan(0);
        expect(result[0][1]).toBeGreaterThan(0);
        expect(result[2][0]).toBeLessThan(width);
        expect(result[2][1]).toBeLessThan(height);
    });
});
