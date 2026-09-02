import { createCanvas, createImageData } from "canvas";
import { describe, expect, it } from "vitest";
import { getImgColor } from "../src/main";

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
    ctx.fillText(text, 10, height / 2);

    return toImageData(ctx.getImageData(0, 0, width, height));
}

describe("getImgColor", () => {
    it("should detect white background with black text", () => {
        const img = createTextImage(200, 100, "white", "black", "Hello World");
        const result = getImgColor(img);

        // 背景应该是白色
        expect(result.bg[0]).toBeGreaterThan(200);
        expect(result.bg[1]).toBeGreaterThan(200);
        expect(result.bg[2]).toBeGreaterThan(200);

        // 文字颜色应该与背景不同
        expect(result.text).toBeDefined();
        expect(result.text.length).toBe(3);
    });

    it("should detect black background with white text", () => {
        const img = createTextImage(200, 100, "black", "white", "Hello World");
        const result = getImgColor(img);

        // 背景应该是黑色
        expect(result.bg[0]).toBeLessThan(50);
        expect(result.bg[1]).toBeLessThan(50);
        expect(result.bg[2]).toBeLessThan(50);

        // 文字颜色应该与背景不同
        expect(result.text).toBeDefined();
        expect(result.text.length).toBe(3);
    });

    it("should detect colored background and text", () => {
        const img = createTextImage(200, 100, "blue", "red", "Hello World");
        const result = getImgColor(img);

        // 背景应该是蓝色
        expect(result.bg[2]).toBeGreaterThan(200); // B 通道

        // 文字颜色应该与背景不同
        expect(result.text).toBeDefined();
        expect(result.text.length).toBe(3);
    });

    it("should return default colors for empty image", () => {
        const canvas = createCanvas(100, 100);
        const ctx = canvas.getContext("2d");
        // 空白图像（透明）
        const img = toImageData(ctx.getImageData(0, 0, 100, 100));
        const result = getImgColor(img);

        expect(result.bg).toBeDefined();
        expect(result.text).toBeDefined();
        expect(result.bg.length).toBe(3);
        expect(result.text.length).toBe(3);
    });

    it("should handle similar colors by finding alternative text color", () => {
        const canvas = createCanvas(200, 100);
        const ctx = canvas.getContext("2d");

        // 浅灰色背景
        ctx.fillStyle = "rgb(200, 200, 200)";
        ctx.fillRect(0, 0, 200, 100);

        // 深灰色文字（与背景有一定差异）
        ctx.fillStyle = "rgb(50, 50, 50)";
        ctx.font = "20px sans-serif";
        ctx.fillText("Hello", 50, 50);

        const img = toImageData(ctx.getImageData(0, 0, 200, 100));
        const result = getImgColor(img);

        // 背景应该是浅灰色
        expect(result.bg[0]).toBeGreaterThan(180);
        expect(result.bg[1]).toBeGreaterThan(180);
        expect(result.bg[2]).toBeGreaterThan(180);

        // 文字颜色应该与背景不同
        expect(result.text).toBeDefined();
    });

    it("should return textEdge color", () => {
        const img = createTextImage(200, 100, "white", "black", "Hello World");
        const result = getImgColor(img);

        expect(result.textEdge).toBeDefined();
        expect(result.textEdge.length).toBe(3);
    });

    it("should handle multiple text colors", () => {
        const canvas = createCanvas(300, 100);
        const ctx = canvas.getContext("2d");

        // 白色背景
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, 300, 100);

        // 黑色文字
        ctx.fillStyle = "black";
        ctx.font = "20px sans-serif";
        ctx.fillText("Black", 20, 40);

        // 红色文字
        ctx.fillStyle = "red";
        ctx.fillText("Red", 120, 40);

        // 蓝色文字
        ctx.fillStyle = "blue";
        ctx.fillText("Blue", 200, 40);

        const img = toImageData(ctx.getImageData(0, 0, 300, 100));
        const result = getImgColor(img);

        // 背景应该是白色
        expect(result.bg[0]).toBeGreaterThan(200);
        expect(result.bg[1]).toBeGreaterThan(200);
        expect(result.bg[2]).toBeGreaterThan(200);

        // 文字颜色应该与背景不同
        expect(result.text).toBeDefined();
    });
});
