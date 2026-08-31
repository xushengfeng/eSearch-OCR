import { createCanvas, createImageData, loadImage } from "canvas";

// 为 Node.js 环境提供 canvas 实现
export function setupCanvas() {
    return {
        canvas: (w: number, h: number) => createCanvas(w, h),
        imageData: createImageData,
    };
}

export { loadImage, createCanvas };
