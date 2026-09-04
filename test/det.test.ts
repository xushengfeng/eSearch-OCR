import fs from "node:fs";
import path from "node:path";
import { createCanvas, createImageData } from "canvas";
import ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";
import { getImgColor, initDet, matchBestBox, setOCREnv } from "../src/main";
import { checkAndWarn, getModelPath } from "./model_paths";
import { createCanvas as createCanvasNode, loadImage, setupCanvas } from "./setup";

const VERSION = "v6_small";
const OUTPUT_DIR = "det_output";

type Box = [[number, number], [number, number], [number, number], [number, number]];

function toImageData(canvasImageData: { data: Uint8ClampedArray; width: number; height: number }): ImageData {
    return createImageData(
        new Uint8ClampedArray(canvasImageData.data),
        canvasImageData.width,
        canvasImageData.height,
    ) as unknown as ImageData;
}

function drawBoxesOnCanvas(width: number, height: number, boxes: Box[]): ReturnType<typeof createCanvas> {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "black";
    for (const box of boxes) {
        const minX = Math.min(box[0][0], box[1][0], box[2][0], box[3][0]);
        const minY = Math.min(box[0][1], box[1][1], box[2][1], box[3][1]);
        const maxX = Math.max(box[0][0], box[1][0], box[2][0], box[3][0]);
        const maxY = Math.max(box[0][1], box[1][1], box[2][1], box[3][1]);
        ctx.fillRect(minX, minY, maxX - minX, maxY - minY);
    }

    return canvas;
}

function calculateCanvasIoU(
    canvas1: ReturnType<typeof createCanvas>,
    canvas2: ReturnType<typeof createCanvas>,
): number {
    const width = canvas1.width;
    const height = canvas1.height;

    const ctx1 = canvas1.getContext("2d");
    const ctx2 = canvas2.getContext("2d");

    const data1 = ctx1.getImageData(0, 0, width, height);
    const data2 = ctx2.getImageData(0, 0, width, height);

    let intersection = 0;
    let count1 = 0;
    let count2 = 0;

    for (let i = 0; i < data1.data.length; i += 4) {
        const black1 = data1.data[i] < 128;
        const black2 = data2.data[i] < 128;

        if (black1) count1++;
        if (black2) count2++;
        if (black1 && black2) intersection++;
    }

    const union = count1 + count2 - intersection;
    return union > 0 ? intersection / union : 0;
}

function getBoxBounds(box: Box) {
    const minX = Math.min(box[0][0], box[1][0], box[2][0], box[3][0]);
    const minY = Math.min(box[0][1], box[1][1], box[2][1], box[3][1]);
    const maxX = Math.max(box[0][0], box[1][0], box[2][0], box[3][0]);
    const maxY = Math.max(box[0][1], box[1][1], box[2][1], box[3][1]);
    return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

function saveComparisonImage(
    name: string,
    sourceCanvas: ReturnType<typeof createCanvas>,
    expectedBoxes: Box[],
    detectedBoxes: Box[],
    iou: number,
) {
    const width = sourceCanvas.width;
    const height = sourceCanvas.height;

    const padding = 10;
    const labelHeight = 30;
    const totalWidth = width * 3 + padding * 4;
    const totalHeight = height + labelHeight + padding * 3;

    const comparisonCanvas = createCanvas(totalWidth, totalHeight);
    const ctx = comparisonCanvas.getContext("2d");

    ctx.fillStyle = "#f0f0f0";
    ctx.fillRect(0, 0, totalWidth, totalHeight);

    ctx.fillStyle = "black";
    ctx.font = "16px sans-serif";
    ctx.fillText(
        `${name} | IoU: ${(iou * 100).toFixed(1)}% | 期望: ${expectedBoxes.length} | 检测: ${detectedBoxes.length}`,
        padding,
        20,
    );

    ctx.drawImage(sourceCanvas, padding, labelHeight + padding, width, height);

    ctx.fillStyle = "green";
    ctx.globalAlpha = 0.3;
    for (const box of expectedBoxes) {
        const bounds = getBoxBounds(box);
        ctx.fillRect(padding + bounds.minX, labelHeight + padding + bounds.minY, bounds.width, bounds.height);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "green";
    ctx.lineWidth = 2;
    for (const box of expectedBoxes) {
        const bounds = getBoxBounds(box);
        ctx.strokeRect(padding + bounds.minX, labelHeight + padding + bounds.minY, bounds.width, bounds.height);
    }

    ctx.drawImage(sourceCanvas, width + padding * 2, labelHeight + padding, width, height);

    ctx.fillStyle = "red";
    ctx.globalAlpha = 0.3;
    for (const box of detectedBoxes) {
        const bounds = getBoxBounds(box);
        ctx.fillRect(
            width + padding * 2 + bounds.minX,
            labelHeight + padding + bounds.minY,
            bounds.width,
            bounds.height,
        );
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    for (const box of detectedBoxes) {
        const bounds = getBoxBounds(box);
        ctx.strokeRect(
            width + padding * 2 + bounds.minX,
            labelHeight + padding + bounds.minY,
            bounds.width,
            bounds.height,
        );
    }

    ctx.drawImage(sourceCanvas, width * 2 + padding * 3, labelHeight + padding, width, height);

    ctx.fillStyle = "green";
    ctx.globalAlpha = 0.2;
    for (const box of expectedBoxes) {
        const bounds = getBoxBounds(box);
        ctx.fillRect(
            width * 2 + padding * 3 + bounds.minX,
            labelHeight + padding + bounds.minY,
            bounds.width,
            bounds.height,
        );
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "green";
    ctx.lineWidth = 2;
    for (const box of expectedBoxes) {
        const bounds = getBoxBounds(box);
        ctx.strokeRect(
            width * 2 + padding * 3 + bounds.minX,
            labelHeight + padding + bounds.minY,
            bounds.width,
            bounds.height,
        );
    }

    ctx.fillStyle = "red";
    ctx.globalAlpha = 0.2;
    for (const box of detectedBoxes) {
        const bounds = getBoxBounds(box);
        ctx.fillRect(
            width * 2 + padding * 3 + bounds.minX,
            labelHeight + padding + bounds.minY,
            bounds.width,
            bounds.height,
        );
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    for (const box of detectedBoxes) {
        const bounds = getBoxBounds(box);
        ctx.strokeRect(
            width * 2 + padding * 3 + bounds.minX,
            labelHeight + padding + bounds.minY,
            bounds.width,
            bounds.height,
        );
    }

    ctx.strokeStyle = "#ccc";
    ctx.lineWidth = 1;
    ctx.strokeRect(padding, labelHeight + padding, width, height);
    ctx.strokeRect(width + padding * 2, labelHeight + padding, width, height);
    ctx.strokeRect(width * 2 + padding * 3, labelHeight + padding, width, height);

    ctx.fillStyle = "black";
    ctx.font = "12px sans-serif";
    ctx.fillText("原图", padding + width / 2 - 15, height + labelHeight + padding + 15);
    ctx.fillText("期望框(绿)", width + padding * 2 + width / 2 - 30, height + labelHeight + padding + 15);
    ctx.fillText("检测框(红)", width * 2 + padding * 3 + width / 2 - 30, height + labelHeight + padding + 15);

    const buffer = comparisonCanvas.toBuffer("image/png");
    const filePath = path.join(OUTPUT_DIR, `${name}.png`);
    fs.writeFileSync(filePath, buffer);
    console.log(`  已保存: ${filePath}`);
}

describe("det", () => {
    beforeAll(() => {
        const env = setupCanvas();
        setOCREnv(env);
        if (!fs.existsSync(OUTPUT_DIR)) {
            fs.mkdirSync(OUTPUT_DIR, { recursive: true });
        }
    });

    it("不同字体大小", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const fontSizes = [14, 20, 28, 40];
        const iouResults: number[] = [];

        for (const fontSize of fontSizes) {
            const width = 700;
            const height = 250;
            const canvas = createCanvas(width, height);
            const ctx = canvas.getContext("2d");

            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = "black";
            ctx.font = `${fontSize}px sans-serif`;

            const lineHeight = fontSize * 1.5;
            const lines = [
                `这是${fontSize}px的测试文本，用于检测不同字体大小的识别效果`,
                `This is ${fontSize}px text for testing different font sizes`,
                "中英混合English混合123数字",
            ];

            const expectedBoxes: Box[] = [];
            for (let i = 0; i < lines.length; i++) {
                const y = fontSize + 10 + i * lineHeight;
                ctx.fillText(lines[i], 20, y);
                const metrics = ctx.measureText(lines[i]);
                const ascent = metrics.actualBoundingBoxAscent || fontSize * 0.8;
                const descent = metrics.actualBoundingBoxDescent || fontSize * 0.2;
                expectedBoxes.push([
                    [20, y - ascent],
                    [20 + metrics.width, y - ascent],
                    [20 + metrics.width, y + descent],
                    [20, y + descent],
                ]);
            }

            const imageData = toImageData(ctx.getImageData(0, 0, width, height));
            const result = await det.det(imageData);
            const detectedBoxes = result.map((r) => r.box);

            const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
            const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
            const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

            iouResults.push(iou);
            console.log(
                `  ${fontSize}px - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}, 期望行数: ${lines.length}`,
            );

            saveComparisonImage(`字体大小_${fontSize}px`, canvas, expectedBoxes, detectedBoxes, iou);
        }

        const avgIoU = iouResults.reduce((a, b) => a + b, 0) / iouResults.length;
        console.log(`不同字体大小 - 平均IoU: ${(avgIoU * 100).toFixed(1)}%`);

        expect(avgIoU).toBeGreaterThanOrEqual(0.3);
    });

    it("不同背景颜色", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const backgrounds = ["#ffffff", "#f5f5f5", "#e0e0e0", "#cccccc", "#333333"];
        const textColors = ["#000000", "#000000", "#000000", "#000000", "#ffffff"];
        const iouResults: number[] = [];

        for (let i = 0; i < backgrounds.length; i++) {
            const width = 700;
            const height = 200;
            const canvas = createCanvas(width, height);
            const ctx = canvas.getContext("2d");

            ctx.fillStyle = backgrounds[i];
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = textColors[i];
            ctx.font = "22px sans-serif";

            const lines = [
                `背景颜色测试 ${backgrounds[i]} 用于验证不同背景下的检测效果`,
                "Background color test for detecting text on various backgrounds",
                "中英混合English混合123",
            ];

            const expectedBoxes: Box[] = [];
            for (let j = 0; j < lines.length; j++) {
                const y = 40 + j * 55;
                ctx.fillText(lines[j], 20, y);
                const metrics = ctx.measureText(lines[j]);
                const ascent = metrics.actualBoundingBoxAscent || 22 * 0.8;
                const descent = metrics.actualBoundingBoxDescent || 22 * 0.2;
                expectedBoxes.push([
                    [20, y - ascent],
                    [20 + metrics.width, y - ascent],
                    [20 + metrics.width, y + descent],
                    [20, y + descent],
                ]);
            }

            const imageData = toImageData(ctx.getImageData(0, 0, width, height));
            const result = await det.det(imageData);
            const detectedBoxes = result.map((r) => r.box);

            const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
            const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
            const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

            iouResults.push(iou);
            console.log(`  ${backgrounds[i]} - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}`);

            saveComparisonImage(`背景颜色_${backgrounds[i]}`, canvas, expectedBoxes, detectedBoxes, iou);
        }

        const avgIoU = iouResults.reduce((a, b) => a + b, 0) / iouResults.length;
        console.log(`不同背景颜色 - 平均IoU: ${(avgIoU * 100).toFixed(1)}%`);

        expect(avgIoU).toBeGreaterThanOrEqual(0.3);
    });

    it("不同文字颜色", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const colors = ["#000000", "#cc0000", "#0066cc", "#009933", "#ff6600"];
        const colorNames = ["黑色", "红色", "蓝色", "绿色", "橙色"];
        const iouResults: number[] = [];

        for (let i = 0; i < colors.length; i++) {
            const width = 700;
            const height = 200;
            const canvas = createCanvas(width, height);
            const ctx = canvas.getContext("2d");

            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = colors[i];
            ctx.font = "24px sans-serif";

            const lines = [
                `${colorNames[i]}文字测试 ${colors[i]} 颜色检测效果验证`,
                `Color text test for ${colorNames[i]} color detection`,
                "中英混合English混合123",
            ];

            const expectedBoxes: Box[] = [];
            for (let j = 0; j < lines.length; j++) {
                const y = 40 + j * 55;
                ctx.fillText(lines[j], 20, y);
                const metrics = ctx.measureText(lines[j]);
                const ascent = metrics.actualBoundingBoxAscent || 24 * 0.8;
                const descent = metrics.actualBoundingBoxDescent || 24 * 0.2;
                expectedBoxes.push([
                    [20, y - ascent],
                    [20 + metrics.width, y - ascent],
                    [20 + metrics.width, y + descent],
                    [20, y + descent],
                ]);
            }

            const imageData = toImageData(ctx.getImageData(0, 0, width, height));
            const result = await det.det(imageData);
            const detectedBoxes = result.map((r) => r.box);

            const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
            const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
            const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

            iouResults.push(iou);
            console.log(
                `  ${colorNames[i]}(${colors[i]}) - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}`,
            );

            saveComparisonImage(`文字颜色_${colorNames[i]}`, canvas, expectedBoxes, detectedBoxes, iou);
        }

        const avgIoU = iouResults.reduce((a, b) => a + b, 0) / iouResults.length;
        console.log(`不同文字颜色 - 平均IoU: ${(avgIoU * 100).toFixed(1)}%`);

        expect(avgIoU).toBeGreaterThanOrEqual(0.3);
    });

    it("混合字体大小", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const width = 800;
        const height = 500;
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        const sections = [
            {
                fontSize: 48,
                lines: ["主标题：文档标题Main Title", "这是大标题的副标题说明文字"],
                startY: 60,
            },
            {
                fontSize: 32,
                lines: ["章节标题：第一章内容概述", "Section title with English text"],
                startY: 180,
            },
            {
                fontSize: 20,
                lines: [
                    "正文段落：这是一段较长的正文内容，用于测试中等字体大小的检测效果",
                    "Body paragraph: This is a longer text for testing medium font size detection",
                    "中英混合English混合123数字测试",
                ],
                startY: 280,
            },
            {
                fontSize: 14,
                lines: [
                    "小字注释：这是注释文字，字体较小，用于测试小字体的检测效果",
                    "Footnote: Small text for testing small font detection",
                ],
                startY: 420,
            },
        ];

        const expectedBoxes: Box[] = [];
        for (const section of sections) {
            ctx.fillStyle = "black";
            ctx.font = `${section.fontSize}px sans-serif`;
            const lineHeight = section.fontSize * 1.4;
            for (let i = 0; i < section.lines.length; i++) {
                const y = section.startY + i * lineHeight;
                ctx.fillText(section.lines[i], 30, y);
                const metrics = ctx.measureText(section.lines[i]);
                const ascent = metrics.actualBoundingBoxAscent || section.fontSize * 0.8;
                const descent = metrics.actualBoundingBoxDescent || section.fontSize * 0.2;
                expectedBoxes.push([
                    [30, y - ascent],
                    [30 + metrics.width, y - ascent],
                    [30 + metrics.width, y + descent],
                    [30, y + descent],
                ]);
            }
        }

        const imageData = toImageData(ctx.getImageData(0, 0, width, height));
        const result = await det.det(imageData);
        const detectedBoxes = result.map((r) => r.box);

        const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
        const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
        const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

        console.log(
            `混合字体大小 - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}, 期望: ${expectedBoxes.length}`,
        );

        saveComparisonImage("混合字体大小", canvas, expectedBoxes, detectedBoxes, iou);

        expect(detectedBoxes.length).toBeGreaterThan(0);
        expect(iou).toBeGreaterThanOrEqual(0.3);
    });

    it("混合颜色", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const width = 700;
        const height = 400;
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        const colorSections = [
            {
                color: "#000000",
                name: "黑色",
                lines: [
                    "黑色文字：这是黑色文字的测试内容，用于验证黑色文字的检测效果",
                    "Black text: Testing black color detection with longer content",
                ],
                startY: 50,
            },
            {
                color: "#cc0000",
                name: "红色",
                lines: [
                    "红色文字：这是红色文字的测试内容，用于验证红色文字的检测效果",
                    "Red text: Testing red color detection with longer content",
                ],
                startY: 150,
            },
            {
                color: "#0066cc",
                name: "蓝色",
                lines: [
                    "蓝色文字：这是蓝色文字的测试内容，用于验证蓝色文字的检测效果",
                    "Blue text: Testing blue color detection with longer content",
                ],
                startY: 250,
            },
        ];

        const expectedBoxes: Box[] = [];
        for (const section of colorSections) {
            ctx.fillStyle = section.color;
            ctx.font = "24px sans-serif";
            const lineHeight = 40;
            for (let i = 0; i < section.lines.length; i++) {
                const y = section.startY + i * lineHeight;
                ctx.fillText(section.lines[i], 30, y);
                const metrics = ctx.measureText(section.lines[i]);
                const ascent = metrics.actualBoundingBoxAscent || 24 * 0.8;
                const descent = metrics.actualBoundingBoxDescent || 24 * 0.2;
                expectedBoxes.push([
                    [30, y - ascent],
                    [30 + metrics.width, y - ascent],
                    [30 + metrics.width, y + descent],
                    [30, y + descent],
                ]);
            }
        }

        const imageData = toImageData(ctx.getImageData(0, 0, width, height));
        const result = await det.det(imageData);
        const detectedBoxes = result.map((r) => r.box);

        const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
        const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
        const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

        console.log(
            `混合颜色 - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}, 期望: ${expectedBoxes.length}`,
        );

        saveComparisonImage("混合颜色", canvas, expectedBoxes, detectedBoxes, iou);

        expect(detectedBoxes.length).toBeGreaterThan(0);
        expect(iou).toBeGreaterThanOrEqual(0.3);
    });

    it("竖排文本", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const width = 600;
        const height = 700;
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        const expectedBoxes: Box[] = [];

        const verticalTexts = [
            { text: "竖排文字测试第一列", x: 50, fontSize: 28, color: "black" },
            { text: "竖排文字测试第二列", x: 150, fontSize: 28, color: "black" },
            { text: "VERTICAL COLUMN", x: 250, fontSize: 24, color: "blue" },
            { text: "中英混合竖排", x: 380, fontSize: 26, color: "red" },
        ];

        for (const vt of verticalTexts) {
            ctx.fillStyle = vt.color;
            ctx.font = `${vt.fontSize}px sans-serif`;
            let y = 50;
            for (const char of vt.text) {
                ctx.fillText(char, vt.x, y);
                const metrics = ctx.measureText(char);
                const ascent = metrics.actualBoundingBoxAscent || vt.fontSize * 0.8;
                const descent = metrics.actualBoundingBoxDescent || vt.fontSize * 0.2;
                expectedBoxes.push([
                    [vt.x, y - ascent],
                    [vt.x + metrics.width, y - ascent],
                    [vt.x + metrics.width, y + descent],
                    [vt.x, y + descent],
                ]);
                y += vt.fontSize + 4;
            }
        }

        const imageData = toImageData(ctx.getImageData(0, 0, width, height));
        const result = await det.det(imageData);
        const detectedBoxes = result.map((r) => r.box);

        const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
        const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
        const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

        console.log(
            `竖排文本 - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}, 期望: ${expectedBoxes.length}`,
        );

        saveComparisonImage("竖排文本", canvas, expectedBoxes, detectedBoxes, iou);

        expect(detectedBoxes.length).toBeGreaterThan(0);
    });

    it("中英混合", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const width = 800;
        const height = 400;
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        const texts = [
            { text: "Hello你好World世界，这是中英混合的测试文本，用于检测OCR识别效果", x: 30, y: 50, fontSize: 24 },
            { text: "测试test混合mix文本内容，包含中文English和数字123的混合", x: 30, y: 100, fontSize: 22 },
            { text: "OCR光学字符识别Optical Character Recognition技术测试", x: 30, y: 150, fontSize: 26 },
            { text: "中文Chinese英文English混合mixed文本text测试test内容content", x: 30, y: 200, fontSize: 20 },
            { text: "123数字Numbers测试，包含各种符号!@#$%^&*()的文本", x: 30, y: 250, fontSize: 22 },
            {
                text: "长文本测试：这是一段很长的中英文混合文本，用于测试OCR对长文本的检测和识别能力",
                x: 30,
                y: 300,
                fontSize: 18,
            },
            { text: "Final line: 最后一行测试文本，包含中英文混合内容", x: 30, y: 350, fontSize: 24 },
        ];

        const expectedBoxes: Box[] = [];
        for (const t of texts) {
            ctx.fillStyle = "black";
            ctx.font = `${t.fontSize}px sans-serif`;
            ctx.fillText(t.text, t.x, t.y);
            const metrics = ctx.measureText(t.text);
            const ascent = metrics.actualBoundingBoxAscent || t.fontSize * 0.8;
            const descent = metrics.actualBoundingBoxDescent || t.fontSize * 0.2;
            expectedBoxes.push([
                [t.x, t.y - ascent],
                [t.x + metrics.width, t.y - ascent],
                [t.x + metrics.width, t.y + descent],
                [t.x, t.y + descent],
            ]);
        }

        const imageData = toImageData(ctx.getImageData(0, 0, width, height));
        const result = await det.det(imageData);
        const detectedBoxes = result.map((r) => r.box);

        const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
        const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
        const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

        console.log(
            `中英混合 - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}, 期望: ${expectedBoxes.length}`,
        );

        saveComparisonImage("中英混合", canvas, expectedBoxes, detectedBoxes, iou);

        expect(detectedBoxes.length).toBeGreaterThan(0);
        expect(iou).toBeGreaterThanOrEqual(0.3);
    });

    it("复杂布局", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const width = 900;
        const height = 600;
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        const sections = [
            {
                fontSize: 48,
                color: "black",
                lines: ["文档标题：OCR技术测试报告"],
                startY: 60,
            },
            {
                fontSize: 32,
                color: "#333333",
                lines: ["第一章：测试背景与目的", "Chapter 1: Test Background and Objectives"],
                startY: 140,
            },
            {
                fontSize: 20,
                color: "#666666",
                lines: [
                    "本测试旨在验证OCR光学字符识别技术对不同字体大小、背景颜色、文字颜色的检测效果",
                    "This test aims to verify OCR detection performance for various font sizes, backgrounds, and colors",
                    "测试内容包含中文、英文、数字以及各种符号的混合文本",
                ],
                startY: 240,
            },
            {
                fontSize: 18,
                color: "blue",
                lines: [
                    "技术说明：OCR（Optical Character Recognition）是光学字符识别的缩写",
                    "它能够将图片中的文字转换为可编辑的文本格式",
                ],
                startY: 380,
            },
            {
                fontSize: 16,
                color: "red",
                lines: [
                    "测试结论：经过多轮测试，OCR系统对不同字体大小、颜色和背景的文本具有良好的检测能力",
                    "Conclusion: The OCR system shows good detection capability for various text conditions",
                ],
                startY: 480,
            },
        ];

        const expectedBoxes: Box[] = [];
        for (const section of sections) {
            ctx.fillStyle = section.color;
            ctx.font = `${section.fontSize}px sans-serif`;
            const lineHeight = section.fontSize * 1.5;
            for (let i = 0; i < section.lines.length; i++) {
                const y = section.startY + i * lineHeight;
                ctx.fillText(section.lines[i], 40, y);
                const metrics = ctx.measureText(section.lines[i]);
                const ascent = metrics.actualBoundingBoxAscent || section.fontSize * 0.8;
                const descent = metrics.actualBoundingBoxDescent || section.fontSize * 0.2;
                expectedBoxes.push([
                    [40, y - ascent],
                    [40 + metrics.width, y - ascent],
                    [40 + metrics.width, y + descent],
                    [40, y + descent],
                ]);
            }
        }

        const imageData = toImageData(ctx.getImageData(0, 0, width, height));
        const result = await det.det(imageData);
        const detectedBoxes = result.map((r) => r.box);

        const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
        const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
        const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

        console.log(
            `复杂布局 - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}, 期望: ${expectedBoxes.length}`,
        );

        saveComparisonImage("复杂布局", canvas, expectedBoxes, detectedBoxes, iou);

        expect(detectedBoxes.length).toBeGreaterThan(0);
        expect(iou).toBeGreaterThanOrEqual(0.3);
    });

    it("紧密排布文本", async () => {
        if (!checkAndWarn(VERSION)) {
            console.warn("跳过测试：模型文件缺失");
            return;
        }

        const paths = getModelPath(VERSION);
        const det = await initDet({
            input: fs.readFileSync(paths.det).buffer,
            ort,
        });

        const width = 700;
        const height = 400;
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = "black";

        const fontSize = 20;
        ctx.font = `${fontSize}px sans-serif`;
        const lineHeight = fontSize * 1;

        const lines = [
            "紧密排布测试：行间距仅为1倍",
            "短文本",
            "这是一段较长的文本行，用于测试紧密排布时OCR检测模型是否能够正确区分每一行文字",
            "中等长度文本行测试",
            "This is a longer English text line to test if the detection model can handle tight spacing with mixed line lengths correctly",
            "混合中英文短行",
            "包含数字123和符号!@#的较长文本行测试内容，验证紧密排布下的检测效果",
        ];

        const expectedBoxes: Box[] = [];
        for (let i = 0; i < lines.length; i++) {
            const y = fontSize + 10 + i * lineHeight;
            ctx.fillText(lines[i], 20, y);
            const metrics = ctx.measureText(lines[i]);
            const ascent = metrics.actualBoundingBoxAscent || fontSize * 0.8;
            const descent = metrics.actualBoundingBoxDescent || fontSize * 0.2;
            expectedBoxes.push([
                [20, y - ascent],
                [20 + metrics.width, y - ascent],
                [20 + metrics.width, y + descent],
                [20, y + descent],
            ]);
        }

        const imageData = toImageData(ctx.getImageData(0, 0, width, height));
        const result = await det.det(imageData);
        const detectedBoxes = result.map((r) => r.box);

        const expectedCanvas = drawBoxesOnCanvas(width, height, expectedBoxes);
        const detectedCanvas = drawBoxesOnCanvas(width, height, detectedBoxes);
        const iou = calculateCanvasIoU(expectedCanvas, detectedCanvas);

        console.log(
            `紧密排布文本 - IoU: ${(iou * 100).toFixed(1)}%, 检测: ${detectedBoxes.length}, 期望: ${expectedBoxes.length}`,
        );

        saveComparisonImage("紧密排布文本", canvas, expectedBoxes, detectedBoxes, iou);

        expect(detectedBoxes.length).toBeGreaterThan(0);
        expect(iou).toBeGreaterThanOrEqual(0.3);
    });
});

function toImageDataForColor(canvasImageData: { data: Uint8ClampedArray; width: number; height: number }): ImageData {
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

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = textColor;
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillText(text, 10, height / 2);

    return toImageDataForColor(ctx.getImageData(0, 0, width, height));
}

describe("getImgColor", () => {
    it("should detect white background with black text", () => {
        const img = createTextImage(200, 100, "white", "black", "Hello World");
        const result = getImgColor(img);

        expect(result.bg[0]).toBeGreaterThan(200);
        expect(result.bg[1]).toBeGreaterThan(200);
        expect(result.bg[2]).toBeGreaterThan(200);

        expect(result.text).toBeDefined();
        expect(result.text.length).toBe(3);
    });

    it("should detect black background with white text", () => {
        const img = createTextImage(200, 100, "black", "white", "Hello World");
        const result = getImgColor(img);

        expect(result.bg[0]).toBeLessThan(50);
        expect(result.bg[1]).toBeLessThan(50);
        expect(result.bg[2]).toBeLessThan(50);

        expect(result.text).toBeDefined();
        expect(result.text.length).toBe(3);
    });

    it("should detect colored background and text", () => {
        const img = createTextImage(200, 100, "blue", "red", "Hello World");
        const result = getImgColor(img);

        expect(result.bg[2]).toBeGreaterThan(200);

        expect(result.text).toBeDefined();
        expect(result.text.length).toBe(3);
    });

    it("should return default colors for empty image", () => {
        const canvas = createCanvas(100, 100);
        const ctx = canvas.getContext("2d");
        const img = toImageDataForColor(ctx.getImageData(0, 0, 100, 100));
        const result = getImgColor(img);

        expect(result.bg).toBeDefined();
        expect(result.text).toBeDefined();
        expect(result.bg.length).toBe(3);
        expect(result.text.length).toBe(3);
    });

    it("should handle similar colors by finding alternative text color", () => {
        const canvas = createCanvas(200, 100);
        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "rgb(200, 200, 200)";
        ctx.fillRect(0, 0, 200, 100);

        ctx.fillStyle = "rgb(50, 50, 50)";
        ctx.font = "20px sans-serif";
        ctx.fillText("Hello", 50, 50);

        const img = toImageDataForColor(ctx.getImageData(0, 0, 200, 100));
        const result = getImgColor(img);

        expect(result.bg[0]).toBeGreaterThan(180);
        expect(result.bg[1]).toBeGreaterThan(180);
        expect(result.bg[2]).toBeGreaterThan(180);

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

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, 300, 100);

        ctx.fillStyle = "black";
        ctx.font = "20px sans-serif";
        ctx.fillText("Black", 20, 40);

        ctx.fillStyle = "red";
        ctx.fillText("Red", 120, 40);

        ctx.fillStyle = "blue";
        ctx.fillText("Blue", 200, 40);

        const img = toImageDataForColor(ctx.getImageData(0, 0, 300, 100));
        const result = getImgColor(img);

        expect(result.bg[0]).toBeGreaterThan(200);
        expect(result.bg[1]).toBeGreaterThan(200);
        expect(result.bg[2]).toBeGreaterThan(200);

        expect(result.text).toBeDefined();
    });
});

type BoxType = [[number, number], [number, number], [number, number], [number, number]];
type color = [number, number, number];

function createTextImageForBox(
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

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = textColor;
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillText(text, x, y);

    return toImageDataForColor(ctx.getImageData(0, 0, width, height));
}

describe("matchBestBox", () => {
    it("should match text color and adjust box boundaries", () => {
        const width = 200;
        const height = 100;
        const img = createTextImageForBox(width, height, "white", "black", "Hello", 50, 50);
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

        expect(result[0][0]).toBeGreaterThan(0);
        expect(result[0][1]).toBeGreaterThan(0);
        expect(result[2][0]).toBeLessThan(width);
        expect(result[2][1]).toBeLessThan(height);
    });

    it("should handle text at edges", () => {
        const width = 200;
        const height = 100;
        const img = createTextImageForBox(width, height, "white", "black", "Top", 10, 20);
        const textEdgeColor: color = [0, 0, 0];

        const box: BoxType = [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ];

        const result = matchBestBox(box, img, textEdgeColor);

        expect(result).toBeDefined();
        expect(result[0][1]).toBeLessThan(height / 2);
    });

    it("should handle no matching text color", () => {
        const width = 200;
        const height = 100;
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);
        const img = toImageDataForColor(ctx.getImageData(0, 0, width, height));

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
        const img = createTextImageForBox(width, height, "blue", "red", "Color", 60, 60);
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

        expect(result[0][0]).toBeGreaterThan(0);
        expect(result[0][1]).toBeGreaterThan(0);
        expect(result[2][0]).toBeLessThan(width);
        expect(result[2][1]).toBeLessThan(height);
    });

    it("should handle text in different positions", () => {
        const width = 200;
        const height = 100;
        const img = createTextImageForBox(width, height, "white", "black", "Bottom", 120, 80);
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

        expect(result[0][0]).toBeGreaterThanOrEqual(0);
        expect(result[0][1]).toBeGreaterThanOrEqual(0);
        expect(result[2][0]).toBeLessThanOrEqual(width);
        expect(result[2][1]).toBeLessThanOrEqual(height);
    });

    it("should preserve box structure", () => {
        const width = 200;
        const height = 100;
        const img = createTextImageForBox(width, height, "white", "black", "Test", 80, 50);
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

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        ctx.fillStyle = "black";
        ctx.font = "16px sans-serif";
        ctx.fillText("Line 1", 20, 30);
        ctx.fillText("Line 2", 20, 50);
        ctx.fillText("Line 3", 20, 70);

        const img = toImageDataForColor(ctx.getImageData(0, 0, width, height));
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

        expect(result[0][0]).toBeGreaterThan(0);
        expect(result[0][1]).toBeGreaterThan(0);
        expect(result[2][0]).toBeLessThan(width);
        expect(result[2][1]).toBeLessThan(height);
    });
});
