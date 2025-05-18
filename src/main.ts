import { runLayout } from "./layout";
import { Cls } from "./cls";
import {
    newCanvas,
    setCanvas,
    toPaddleInput,
    type SessionType,
    type AsyncType,
    data2canvas,
    resizeImg,
    int,
    tLog,
    clip,
} from "./untils";
import { type Contour, findContours, minAreaRect, type Point } from "./cv";

export {
    setOCREnv,
    init,
    /** @deprecated use return obj from init */
    x as ocr,
    loadImg,
    /** @deprecated use return obj from init */
    Det as det,
    /** @deprecated use return obj from init */
    Rec as rec,
    afAfRec as analyzeLayout,
    initDet,
    initRec,
    initDocDirCls,
    rotateImg,
};
export type initType = AsyncType<ReturnType<typeof init>>;

type InitOcrBase = {
    detPath: string;
    recPath: string;
    layoutPath?: string;
    docClsPath?: string;
    dic: string;
    layoutDic?: string;
    dev?: boolean;
    log?: boolean;
    imgh?: number;
    detRatio?: number;
    ort: typeof import("onnxruntime-common");
    ortOption?: import("onnxruntime-common").InferenceSession.SessionOptions;
};

type InitOcrGlobal = {
    /** @deprecated use setOCREnv instead */
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    canvas?: (w: number, h: number) => any;
    /** @deprecated use setOCREnv instead */
    imageData?;
};

type InitOcrCb = {
    onProgress?: (type: "det" | "rec", total: number, count: number) => void;
    onDet?: (r: detResultType) => void;
    onRec?: (index: number, result: { text: string; mean: number }) => void;
};

type onProgressType = (type: "det" | "rec", total: number, count: number) => void;

type loadImgType = string | HTMLImageElement | HTMLCanvasElement | ImageData;
type detResultType = { box: BoxType; img: ImageData; style: { bg: color; text: color } }[];
type detDataType = {
    data: AsyncType<ReturnType<typeof runDet>>["data"];
    width: number;
    height: number;
};
type pointType = [number, number];
type BoxType = [pointType, pointType, pointType, pointType];
type pointsType = pointType[];
type resultType = { text: string; mean: number; box: BoxType; style: { bg: color; text: color } }[];

type ReadingDirPart = "lr" | "rl" | "tb" | "bt";
type ReadingDir = {
    block: ReadingDirPart;
    inline: ReadingDirPart;
};

const task = new tLog("t");
const task2 = new tLog("af_det");

let dev = false;
let canlog = false;

let globalOCR: AsyncType<ReturnType<typeof initOCR>> | null = null;

function putImgDom(img: OffscreenCanvas, id?: string) {
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d")!.drawImage(img, 0, 0);
    if (id) canvas.id = id;
    try {
        document?.body?.append(canvas);
    } catch (error) {}
}

let createImageData = (data: Uint8ClampedArray, w: number, h: number) => {
    return new ImageData(data, w, h);
};

function log(...args: any[]) {
    if (canlog) console.log(...args);
}
function logSrc(...args: any[]) {
    if (canlog) console.log(...args.map((i) => structuredClone(i)));
}

function logColor(...args: string[]) {
    if (canlog) {
        console.log(args.map((x) => `%c${x}`).join(""), ...args.map((x) => `color: ${x}`));
    }
}

async function init(op: InitOcrBase & InitOcrCb & InitOcrGlobal) {
    setOCREnv(op);
    const x = await initOCR(op);
    globalOCR = x;
    return x;
}

function setOCREnv(op: {
    dev?: boolean;
    log?: boolean;
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    canvas?: (w: number, h: number) => any;
    imageData?;
}) {
    dev = Boolean(op.dev);
    canlog = dev || Boolean(op.log);
    if (!dev) {
        task.l = () => {};
        task2.l = () => {};
    }
    if (op.canvas) setCanvas(op.canvas);
    if (op.imageData) createImageData = op.imageData;
}

async function loadImg(src: loadImgType) {
    let img: HTMLImageElement | HTMLCanvasElement | ImageData;
    if (typeof window === "undefined") {
        const x = src as ImageData;
        if (!x.data || !x.width || !x.height) throw new Error("invalid image data");
        return x;
    }
    if (typeof src === "string") {
        img = new Image();
        img.src = src;
        await new Promise((resolve) => {
            (img as HTMLImageElement).onload = resolve;
        });
    } else if (src instanceof ImageData) {
        img = src;
    } else {
        img = src;
    }
    if (img instanceof HTMLImageElement) {
        const canvas = newCanvas(img.width, img.height);
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("canvas context is null");
        ctx.drawImage(img, 0, 0);
        img = ctx.getImageData(0, 0, img.width, img.height);
    }
    if (img instanceof HTMLCanvasElement) {
        const ctx = img.getContext("2d");
        if (!ctx) throw new Error("canvas context is null");
        img = ctx.getImageData(0, 0, img.width, img.height);
    }
    return img;
}

function checkNode() {
    try {
        newCanvas(1, 1);
        createImageData(new Uint8ClampedArray(4), 1, 1);
    } catch (error) {
        console.log("nodejs need set canvas, please use setOCREnv to set canvas and imageData");
        throw error;
    }
}

async function x(i: loadImgType) {
    if (!globalOCR) throw new Error("need init");
    return globalOCR.ocr(i);
}

async function Det(s: ImageData) {
    if (!globalOCR) throw new Error("need init");
    return globalOCR.det(s);
}

async function Rec(box: detResultType) {
    if (!globalOCR) throw new Error("need init");
    return globalOCR.rec(box);
}

/** 主要操作 */
async function initOCR(op: InitOcrBase & InitOcrCb) {
    checkNode();

    // @ts-ignore
    const docCls = op.docClsPath ? await initDocDirCls(op) : undefined;
    const det = await initDet(op);
    const rec = await initRec(op);
    return {
        ocr: async (srcimg: loadImgType) => {
            let img = await loadImg(srcimg);

            let dir = 0;
            if (docCls) {
                dir = await docCls.docCls(img);
                log("dir", dir);
                img = rotateImg(img, 360 - dir);
            }

            const box = await det.det(img);

            if (op.onDet) op.onDet(box);

            const mainLine = await rec.rec(box);
            const newMainLine = afAfRec(mainLine);
            log(mainLine, newMainLine);
            task.l("end");
            return { src: mainLine, ...newMainLine, docDir: dir };
        },
        det: det.det,
        rec: rec.rec,
    };
}

async function initDocDirCls(op: {
    docClsPath: string;
    ort: typeof import("onnxruntime-common");
    ortOption?: import("onnxruntime-common").InferenceSession.SessionOptions;
    onProgress?: onProgressType;
}) {
    const cls = await op.ort.InferenceSession.create(op.docClsPath, op.ortOption);
    const docCls = async (img: ImageData) => {
        return Cls(img, op.ort, cls, [0, 90, 180, 270], 224, 224);
    };
    return { docCls };
}

async function initDet(op: {
    detPath: string;
    detRatio?: number;
    ort: typeof import("onnxruntime-common");
    ortOption?: import("onnxruntime-common").InferenceSession.SessionOptions;
    onProgress?: onProgressType;
}) {
    checkNode();

    let detRatio = 1;
    let onProgress: onProgressType = () => {};

    const det = await op.ort.InferenceSession.create(op.detPath, op.ortOption);
    if (op.detRatio !== undefined) detRatio = op.detRatio;
    if (op.onProgress) onProgress = op.onProgress;

    async function Det(srcimg: ImageData) {
        const img = srcimg;

        if (dev) {
            const srcCanvas = data2canvas(img);
            putImgDom(srcCanvas);
        }

        task.l("pre_det");
        const { data: beforeDetData, width: resizeW, height: resizeH } = beforeDet(img, detRatio);
        const { transposedData, image } = beforeDetData;
        task.l("det");
        onProgress("det", 1, 0);
        const detResults = await runDet(transposedData, image, det, op.ort);

        task.l("aft_det");
        const box = afterDet(
            { data: detResults.data, width: detResults.dims[3], height: detResults.dims[2] },
            resizeW,
            resizeH,
            img,
        );

        onProgress("det", 1, 1);
        return box;
    }

    return { det: Det };
}

async function initRec(op: {
    recPath: string;
    dic: string;
    imgh?: number;
    ort: typeof import("onnxruntime-common");
    ortOption?: import("onnxruntime-common").InferenceSession.SessionOptions;
    onProgress?: (type: "det" | "rec", total: number, count: number) => void;
    onRec?: (index: number, result: { text: string; mean: number }) => void;
}) {
    checkNode();

    let imgh = 48;
    let onProgress: (type: "det" | "rec", total: number, count: number) => void = () => {};
    const rec = await op.ort.InferenceSession.create(op.recPath, op.ortOption);
    const dic = op.dic?.split(/\r\n|\r|\n/) || [];
    if (dic.at(-1) === "") {
        // 多出的换行
        dic[dic.length - 1] = " ";
    } else {
        dic.push(" ");
    }
    if (op.imgh) imgh = op.imgh;
    if (op.onProgress) onProgress = op.onProgress;

    async function Rec(box: detResultType) {
        const mainLine: resultType = [];
        task.l("bf_rec");
        const recL = beforeRec(box, imgh);
        let runCount = 0;
        op?.onProgress?.("rec", recL.length, runCount);
        const mainLine0: { text: string; mean: number }[] = [];
        for (const [index, item] of recL.entries()) {
            const { b, imgH, imgW } = item;
            const recResults = await runRec(b, imgH, imgW, rec, op.ort);
            const result = afterRec(recResults, dic)[0];
            mainLine.push({
                text: result.text,
                mean: result.mean,
                box: box[index].box,
                style: box[index].style,
            });
            op?.onRec?.(index, result);
            runCount++;
            op?.onProgress?.("rec", recL.length, runCount);
            mainLine0.push(...afterRec(recResults, dic));
        }
        task.l("rec_end");
        return mainLine.filter((x) => x.mean >= 0.5) as resultType;
    }

    return { rec: Rec };
}

async function runDet(
    transposedData: number[][][],
    image: ImageData,
    det: SessionType,
    ort: typeof import("onnxruntime-common"),
) {
    const detData = Float32Array.from(transposedData.flat(3));

    const detTensor = new ort.Tensor("float32", detData, [1, 3, image.height, image.width]);
    const detFeed = {};
    detFeed[det.inputNames[0]] = detTensor;

    const detResults = await det.run(detFeed);
    return detResults[det.outputNames[0]];
}

async function runRec(
    b: number[][][],
    imgH: number,
    imgW: number,
    rec: SessionType,
    ort: typeof import("onnxruntime-common"),
) {
    const recData = Float32Array.from(b.flat(3));

    const recTensor = new ort.Tensor("float32", recData, [1, 3, imgH, imgW]);
    const recFeed = {};
    recFeed[rec.inputNames[0]] = recTensor;

    const recResults = await rec.run(recFeed);
    return recResults[rec.outputNames[0]];
}

function beforeDet(srcImg: ImageData, detRatio: number) {
    const resizeH = Math.max(Math.round((srcImg.height * detRatio) / 32) * 32, 32);
    const resizeW = Math.max(Math.round((srcImg.width * detRatio) / 32) * 32, 32);

    if (dev) {
        const srcCanvas = data2canvas(srcImg);
        putImgDom(srcCanvas);
    }

    const image = resizeImg(srcImg, resizeW, resizeH, "fill");

    const transposedData = toPaddleInput(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    log(image);
    if (dev) {
        const srcCanvas = data2canvas(image);
        putImgDom(srcCanvas);
    }
    return { data: { transposedData, image }, width: resizeW, height: resizeH };
}

function afterDet(dataSet: detDataType, _resizeW: number, _resizeH: number, srcData: ImageData) {
    task2.l("");

    // 考虑到fill模式，小的不变动
    const w = Math.min(srcData.width, _resizeW);
    const h = Math.min(srcData.height, _resizeH);

    const { data, width, height } = dataSet;
    const bitData = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i++) {
        const v = (data[i] as number) > 0.3 ? 255 : 0;
        bitData[i] = v;
    }

    if (dev) {
        const clipData = new Uint8ClampedArray(width * height * 4);
        for (let i = 0; i < data.length; i++) {
            const n = i * 4;
            const v = (data[i] as number) > 0.3 ? 255 : 0;
            clipData[n] = clipData[n + 1] = clipData[n + 2] = v;
            clipData[n + 3] = 255;
            bitData[i] = v;
        }
        const myImageData = createImageData(clipData, width, height);
        const srcCanvas = data2canvas(myImageData);
        putImgDom(srcCanvas, "det_ru");
    }

    task2.l("edge");

    const edgeRect: detResultType = [];

    const src2: number[][] = [];
    for (let y = 0; y < height; y++) {
        src2.push(Array.from(bitData.slice(y * width, y * width + width)));
    }

    const contours2: Point[][] = [];

    findContours(src2, contours2);

    if (dev) {
        const xctx = (document.querySelector("#det_ru") as HTMLCanvasElement).getContext("2d")!;

        for (const item of contours2) {
            xctx.moveTo(item[0].x, item[0].y);
            for (const p of item) {
                xctx.lineTo(p.x, p.y);
            }
            xctx.strokeStyle = "red";
            xctx.closePath();
            xctx.stroke();
        }
    }

    for (let i = 0; i < contours2.length; i++) {
        task2.l("get_box");
        const minSize = 3;
        const l: Contour = contours2[i];

        const { points, sside } = getMiniBoxes(l);
        if (sside < minSize) continue;
        // TODO sort fast

        const resultObj = unclip2(points);
        const box = resultObj.points;
        if (resultObj.sside < minSize + 2) {
            continue;
        }

        const rx = srcData.width / w;
        const ry = srcData.height / h;

        for (let i = 0; i < box.length; i++) {
            box[i][0] *= rx;
            box[i][1] *= ry;
        }
        task2.l("order");

        const box1 = orderPointsClockwise(box);
        for (const item of box1) {
            item[0] = clip(Math.round(item[0]), 0, srcData.width);
            item[1] = clip(Math.round(item[1]), 0, srcData.height);
        }
        const rect_width = int(linalgNorm(box1[0], box1[1]));
        const rect_height = int(linalgNorm(box1[0], box1[3]));
        if (rect_width <= 3 || rect_height <= 3) continue;

        drawBox(box, "", "red", "det_ru");

        task2.l("crop");

        const c = getRotateCropImage(srcData, box);

        task2.l("match best");

        const { bg, text } = getImgColor(c);

        const bb = matchBestBox(box, c, text);

        edgeRect.push({ box: bb, img: c, style: { bg, text } });
    }
    task2.l("e");

    log(edgeRect);

    return edgeRect;
}

function polygonPolygonArea(polygon: pointsType) {
    let i = -1;
    const n = polygon.length;
    let a: pointType;
    let b = polygon[n - 1];
    let area = 0;

    while (++i < n) {
        a = b;
        b = polygon[i];
        area += a[1] * b[0] - a[0] * b[1];
    }

    return area / 2;
}
function polygonPolygonLength(polygon: pointsType) {
    let i = -1;
    const n = polygon.length;
    let b = polygon[n - 1];
    let xa: number;
    let ya: number;
    let xb = b[0];
    let yb = b[1];
    let perimeter = 0;

    while (++i < n) {
        xa = xb;
        ya = yb;
        b = polygon[i];
        xb = b[0];
        yb = b[1];
        xa -= xb;
        ya -= yb;
        perimeter += Math.hypot(xa, ya);
    }

    return perimeter;
}

function unclip2(box: pointsType) {
    const unclip_ratio = 1.5;
    const area = Math.abs(polygonPolygonArea(box));
    const length = polygonPolygonLength(box);
    const distance = (area * unclip_ratio) / length;

    const expandedArr: pointType[] = [];

    for (const [i, p] of box.entries()) {
        const lastPoint = box.at((i - 1) % 4)!;
        const nextPoint = box.at((i + 1) % 4)!;

        const x1 = p[0] - lastPoint[0];
        const y1 = p[1] - lastPoint[1];
        const d1 = Math.sqrt(x1 ** 2 + y1 ** 2);
        const dx1 = (x1 / d1) * distance;
        const dy1 = (y1 / d1) * distance;

        const x2 = p[0] - nextPoint[0];
        const y2 = p[1] - nextPoint[1];
        const d2 = Math.sqrt(x2 ** 2 + y2 ** 2);
        const dx2 = (x2 / d2) * distance;
        const dy2 = (y2 / d2) * distance;

        expandedArr.push([p[0] + dx1 + dx2, p[1] + dy1 + dy2]);
    }

    const v1 = [expandedArr[0][0] - expandedArr[1][0], expandedArr[0][1] - expandedArr[1][1]];
    const v2 = [expandedArr[2][0] - expandedArr[1][0], expandedArr[2][1] - expandedArr[1][1]];
    const cross = v1[0] * v2[1] - v1[1] * v2[0];

    return { points: expandedArr as BoxType, sside: Math.abs(cross) };
}

function boxPoints(center: { x: number; y: number }, size: { width: number; height: number }, angle: number) {
    const width = size.width;
    const height = size.height;

    const theta = (angle * Math.PI) / 180.0;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);

    const cx = center.x;
    const cy = center.y;

    const dx = width * 0.5;
    const dy = height * 0.5;

    const rotatedPoints: [number, number][] = [];

    // Top-Left
    const x1 = cx - dx * cosTheta + dy * sinTheta;
    const y1 = cy - dx * sinTheta - dy * cosTheta;
    rotatedPoints.push([x1, y1]);

    // Top-Right
    const x2 = cx + dx * cosTheta + dy * sinTheta;
    const y2 = cy + dx * sinTheta - dy * cosTheta;
    rotatedPoints.push([x2, y2]);

    // Bottom-Right
    const x3 = cx + dx * cosTheta - dy * sinTheta;
    const y3 = cy + dx * sinTheta + dy * cosTheta;
    rotatedPoints.push([x3, y3]);

    // Bottom-Left
    const x4 = cx - dx * cosTheta - dy * sinTheta;
    const y4 = cy - dx * sinTheta + dy * cosTheta;
    rotatedPoints.push([x4, y4]);

    return rotatedPoints;
}

function getMiniBoxes(contour: Point[]) {
    const l = contour;
    const boundingBox = minAreaRect(l);
    const points = Array.from(boxPoints(boundingBox.center, boundingBox.size, boundingBox.angle)).sort(
        (a, b) => a[0] - b[0],
    ) as pointsType;

    let index_1 = 0;
    let index_2 = 1;
    let index_3 = 2;
    let index_4 = 3;
    if (points[1][1] > points[0][1]) {
        index_1 = 0;
        index_4 = 1;
    } else {
        index_1 = 1;
        index_4 = 0;
    }
    if (points[3][1] > points[2][1]) {
        index_2 = 2;
        index_3 = 3;
    } else {
        index_2 = 3;
        index_3 = 2;
    }

    const box = [points[index_1], points[index_2], points[index_3], points[index_4]] as BoxType;
    const side = Math.min(boundingBox.size.height, boundingBox.size.width);
    return { points: box, sside: side };
}

function flatten(arr: number[] | number[][]) {
    return arr.flat();
}
function linalgNorm(p0: pointType, p1: pointType) {
    return Math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2);
}
function orderPointsClockwise(pts: BoxType) {
    const rect: BoxType = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ];
    const s = pts.map((pt) => pt[0] + pt[1]);
    rect[0] = pts[s.indexOf(Math.min(...s))];
    rect[2] = pts[s.indexOf(Math.max(...s))];
    const tmp = pts.filter((pt) => pt !== rect[0] && pt !== rect[2]);
    const diff = tmp[1].map((e, i) => e - tmp[0][i]);
    rect[1] = tmp[diff.indexOf(Math.min(...diff))];
    rect[3] = tmp[diff.indexOf(Math.max(...diff))];
    return rect;
}
function getRotateCropImage(img: ImageData, points: BoxType) {
    // todo 根据曲线裁切
    const [p0, p1, p2, p3] = points.map((p) => ({ x: p[0], y: p[1] }));
    // 计算原始宽高
    const width = Math.sqrt((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2);
    const height = Math.sqrt((p3.x - p0.x) ** 2 + (p3.y - p0.y) ** 2);

    // 计算变换矩阵参数
    const dx1 = p1.x - p0.x;
    const dy1 = p1.y - p0.y;
    const dx3 = p3.x - p0.x;
    const dy3 = p3.y - p0.y;

    const determinant = dx1 * dy3 - dx3 * dy1;
    if (determinant === 0) throw new Error("点共线，无法形成矩形");

    const a = (width * dy3) / determinant;
    const c = (-dx3 * width) / determinant;
    const b = (-height * dy1) / determinant;
    const d = (dx1 * height) / determinant;
    const e = -a * p0.x - c * p0.y;
    const f = -b * p0.x - d * p0.y;

    const inputCanvas = data2canvas(img);

    // 创建输出Canvas
    const outputCanvas = newCanvas(Math.ceil(width), Math.ceil(height));
    const ctx = outputCanvas.getContext("2d")!;

    // 应用变换并绘制
    ctx.setTransform(a, b, c, d, e, f);
    ctx.drawImage(inputCanvas, 0, 0);

    // 重置变换以进行后续操作
    ctx.resetTransform();

    return ctx.getImageData(0, 0, outputCanvas.width, outputCanvas.height);
}

type color = [number, number, number];

function getImgColor(img: ImageData) {
    const histogram = new Map<string, number>();
    const data = img.data;

    for (let i = 0; i < data.length; i += 4) {
        const x = (i / 4) % img.width;
        if (x > img.height * 4) continue;
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const colorKey = [r, g, b].join(",");
        histogram.set(colorKey, (histogram.get(colorKey) || 0) + 1);
    }

    const colorList = getHighestFrequency(histogram, 20).map((c) => ({
        el: c.el.split(",").map(Number) as color,
        count: c.count,
    }));
    const bg = colorList.at(0)?.el || [255, 255, 255];
    const textEdge = colorList.at(1)?.el || [0, 0, 0];

    let text = textEdge;

    const colorD = 100;

    if (areColorsSimilar(textEdge, bg) < colorD) {
        const colorSplit = colorList.slice(1).filter((c) => areColorsSimilar(c.el, bg) > 50);
        if (colorSplit.length > 0) {
            text = [0, 1, 2] // rgb各自平均
                .map((i) =>
                    Math.round(average2(colorSplit.map((c) => [c.el[i], c.count] as [number, number]))),
                ) as color;
        }
        if (colorSplit.length === 0 || areColorsSimilar(text, bg) < colorD) text = bg.map((x) => 255 - x) as color;
        logColor(`rgb(${text.join(",")})`);
    }

    return {
        bg: bg,
        text: text,
        textEdge: textEdge,
    };
}

function areColorsSimilar(color1: color, color2: color) {
    const rgb1 = color1;
    const rgb2 = color2;

    const distance = Math.sqrt((rgb1[0] - rgb2[0]) ** 2 + (rgb1[1] - rgb2[1]) ** 2 + (rgb1[2] - rgb2[2]) ** 2);

    return distance;
}

function getHighestFrequency<t>(map: Map<t, number>, c = 1) {
    let l: { el: t; count: number }[] = [];
    map.forEach((count, name) => {
        if (l.length === 0) l.push({ el: name, count });
        else {
            if (l.length < c) {
                l.push({ el: name, count });
            } else if (l.find((i) => i.count <= count)) {
                l.push({ el: name, count });
            }
            l.sort((a, b) => b.count - a.count);
            if (l.length > c) {
                l = l.slice(0, c);
            }
        }
    });

    return l;
}

function matchBestBox(box: BoxType, img: ImageData, textEdgeColor: color) {
    let yFromTop = 0;
    let yFromBottom = img.height;
    let xFromLeft = 0;
    let xFromRight = img.width;

    function match(pix: color) {
        return areColorsSimilar(pix, textEdgeColor) < 200;
    }

    yt: for (let y = yFromTop; y < img.height; y++) {
        for (let x = 0; x < img.width; x++) {
            const pix = getImgPix(img, x, y);
            if (match(pix)) {
                yFromTop = y;
                break yt;
            }
        }
    }

    yb: for (let y = yFromBottom - 1; y >= 0; y--) {
        for (let x = 0; x < img.width; x++) {
            const pix = getImgPix(img, x, y);
            if (match(pix)) {
                yFromBottom = y;
                break yb;
            }
        }
    }

    xl: for (let x = xFromLeft; x < img.width; x++) {
        for (let y = yFromTop; y <= yFromBottom; y++) {
            const pix = getImgPix(img, x, y);
            if (match(pix)) {
                xFromLeft = x;
                break xl;
            }
        }
    }

    xr: for (let x = xFromRight - 1; x >= 0; x--) {
        for (let y = yFromTop; y <= yFromBottom; y++) {
            const pix = getImgPix(img, x, y);
            if (match(pix)) {
                xFromRight = x;
                break xr;
            }
        }
    }

    const dyT = clip(yFromTop - 1, 0, 4);
    const dyB = clip(img.height - yFromBottom - 1, 0, 4);
    const dxL = clip(xFromLeft - 1, 0, 4);
    const dxR = clip(img.width - xFromRight - 1, 0, 4);

    const newBox = [
        [box[0][0] + dxL, box[0][1] + dyT],
        [box[1][0] - dxR, box[1][1] + dyT],
        [box[2][0] - dxR, box[2][1] - dyB],
        [box[3][0] + dxL, box[3][1] - dyB],
    ] as BoxType;

    return newBox;
}

function getImgPix(img: ImageData, x: number, y: number) {
    const index = (y * img.width + x) * 4;
    return Array.from(img.data.slice(index, index + 4)) as color;
}

function beforeRec(box: { box: BoxType; img: ImageData }[], imgH: number) {
    const l: { b: number[][][]; imgH: number; imgW: number }[] = [];
    function resizeNormImg(img: ImageData) {
        const w = Math.floor(imgH * (img.width / img.height));
        const d = resizeImg(img, w, imgH, undefined, false);
        if (dev) putImgDom(data2canvas(d, w, imgH));
        return { data: d, w, h: imgH };
    }

    for (const r of box) {
        const reImg = resizeNormImg(r.img);
        l.push({ b: toPaddleInput(reImg.data, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), imgH: reImg.h, imgW: reImg.w });
    }
    log(l);
    return l;
}

function afterRec(data: AsyncType<ReturnType<typeof runRec>>, character: string[]) {
    const predLen = data.dims[2];
    const line: { text: string; mean: number }[] = [];
    let ml = data.dims[0] - 1;

    function getChar(i: number) {
        return character.at(i - 1) ?? "";
    }

    for (let l = 0; l < data.data.length; l += predLen * data.dims[1]) {
        const predsIdx: number[] = [];
        const predsProb: number[] = [];

        for (let i = l; i < l + predLen * data.dims[1]; i += predLen) {
            const tmpArr = data.data.slice(i, i + predLen) as Float32Array;

            let tmpMax = Number.NEGATIVE_INFINITY;
            let tmpIdx = -1;
            let tmpSecond = Number.NEGATIVE_INFINITY;
            let tmpSecondI = -1;

            for (let j = 0; j < tmpArr.length; j++) {
                const currentValue = tmpArr[j];
                if (currentValue > tmpMax) {
                    tmpSecond = tmpMax;
                    tmpMax = currentValue;
                    tmpIdx = j;
                } else if (currentValue > tmpSecond && currentValue < tmpMax) {
                    tmpSecond = currentValue;
                    tmpSecondI = j;
                }
            }
            if (tmpIdx === 0 && getChar(tmpSecondI) === " " && tmpSecond > 0.001) {
                tmpMax = tmpSecond;
                tmpIdx = tmpSecondI;
            }

            predsProb.push(tmpMax);
            predsIdx.push(tmpIdx);
        }
        line[ml] = decode(predsIdx, predsProb);
        ml--;
    }
    function decode(textIndex: number[], textProb: number[]) {
        const charList: string[] = [];
        const confList: number[] = [];
        const isRemoveDuplicate = true;
        for (let idx = 0; idx < textIndex.length; idx++) {
            if (textIndex[idx] === 0) continue;
            if (isRemoveDuplicate) {
                if (idx > 0 && textIndex[idx - 1] === textIndex[idx]) {
                    continue;
                }
            }
            charList.push(getChar(textIndex[idx]));
            confList.push(textProb[idx]);
        }
        let text = "";
        let mean = 0;
        if (charList.length) {
            text = charList.join("").trim();
            let sum = 0;
            for (const item of confList) {
                sum += item;
            }
            mean = sum / confList.length;
        }
        return { text, mean };
    }
    return line;
}

/** 排版分析 */
function afAfRec(l: resultType) {
    log(l);

    // 假定阅读方向都是统一的
    // 假定横平竖直

    const dirs: ReadingDir[] = [
        { block: "tb", inline: "lr" },
        { block: "rl", inline: "tb" },
    ];
    const dir: ReadingDir = { block: "tb", inline: "lr" };

    const dirVector = {
        inline: [1, 0] as VectorType,
        block: [0, 1] as VectorType,
    };

    const baseVector = {
        inline: [1, 0] as VectorType,
        block: [0, 1] as VectorType,
    };

    const Point = {
        center: (p1: pointType, p2: pointType): pointType => [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2],
    };

    const Box = {
        inlineStart: (b: BoxType) => b[0][0],
        inlineEnd: (b: BoxType) => b[1][0],
        blockStart: (b: BoxType) => b[0][1],
        blockEnd: (b: BoxType) => b[3][1],
        inlineSize: (b: BoxType) => b[1][0] - b[0][0],
        blockSize: (b: BoxType) => b[3][1] - b[0][1],
        dis1: (a: number, b: number) => Math.abs(a - b),
        inlineStartDis: (a: BoxType, b: BoxType) => Math.abs(a[0][0] - b[0][0]),
        inlineEndDis: (a: BoxType, b: BoxType) => Math.abs(a[1][0] - b[1][0]),
        blockGap: (newB: BoxType, oldB: BoxType) => Math.abs(newB[0][1] - oldB[3][1]),
        inlineCenter: (b: BoxType) => (b[2][0] + b[0][0]) / 2, // 这里考虑了倾斜
        blockCenter: (b: BoxType) => (b[2][1] + b[0][1]) / 2,
        center: (b: BoxType) => Point.center(b[0], b[2]),
    };

    type VectorType = [number, number];

    const Vector = {
        fromPonts: (p1: pointType, p2: pointType): pointType => [p1[0] - p2[0], p1[1] - p2[1]],
        dotMup: (a: VectorType, b: VectorType) => a[0] * b[0] + a[1] * b[1],
        numMup: (a: VectorType, b: number) => [a[0] * b, a[1] * b] as VectorType,
        add: (a: VectorType, b: VectorType) => [a[0] + b[0], a[1] + b[1]] as VectorType,
    };

    function averLineAngles(a: number[]) {
        let iav = 0;
        for (const i of a) {
            const a1 = i > 180 ? i - 180 : i;
            const a2 = a1 - 180;
            iav = (iav + (Math.abs(a2 - iav) < Math.abs(a1 - iav) ? a2 : a1)) / 2;
        }
        return iav;
    }
    function lineAngleNear(a1: number, a2: number) {
        if (Math.abs(a1 - a2) < 45) return true;
        if (Math.abs(a1 - (a2 - 180)) < 45) return true;
        return false;
    }
    function dir2xy(d: ReadingDirPart) {
        if (d === "lr" || d === "rl") return "x";
        return "y";
    }
    function smallest<I>(l: I[], f: (a: I) => number) {
        let min = Number.POSITIVE_INFINITY;
        let minIndex = -1;
        for (let i = 0; i < l.length; i++) {
            const v = f(l[i]);
            if (v < min) {
                min = v;
                minIndex = i;
            }
        }
        return l[minIndex];
    }

    const tipV: Record<ReadingDirPart, VectorType> = {
        lr: [1, 0],
        rl: [-1, 0],
        tb: [0, 1],
        bt: [0, -1],
    };

    /** 坐标系变换 */
    function transXY(old: ReadingDir, target: ReadingDir) {
        const oX = tipV[old.inline];
        const oY = tipV[old.block];
        const tX = tipV[target.inline];
        const tY = tipV[target.block];
        const tInOX = [Vector.dotMup(tX, oX), Vector.dotMup(tX, oY)] as VectorType;
        const tInOY = [Vector.dotMup(tY, oX), Vector.dotMup(tY, oY)] as VectorType;
        return (p: pointType) => {
            return [Vector.dotMup(p, tInOX), Vector.dotMup(p, tInOY)] as pointType;
        };
    }

    function r(point: pointType, point2: pointType) {
        return Math.sqrt((point[0] - point2[0]) ** 2 + (point[1] - point2[1]) ** 2);
    }

    function outerRect(boxes: BoxType[]) {
        const points = boxes.flatMap((i) => i.map((i) => i));
        const x1 = Math.min(...points.map((p) => Vector.dotMup(p, baseVector.inline)));
        const x2 = Math.max(...points.map((p) => Vector.dotMup(p, baseVector.inline)));
        const y1 = Math.min(...points.map((p) => Vector.dotMup(p, baseVector.block)));
        const y2 = Math.max(...points.map((p) => Vector.dotMup(p, baseVector.block)));

        const o = Vector.add(Vector.numMup(baseVector.inline, x1), Vector.numMup(baseVector.block, y1));

        const w = Vector.numMup(baseVector.inline, x2 - x1);
        const h = Vector.numMup(baseVector.block, y2 - y1);

        return [o, Vector.add(o, w), Vector.add(Vector.add(o, w), h), Vector.add(o, h)] as BoxType;
    }

    function pushColumn(b: resultType[0]) {
        let nearest: number | null = null;
        let _jl = Number.POSITIVE_INFINITY;
        for (const i in columns) {
            const last = columns[i].at(-1);
            if (!last) continue;
            const jl = r(b.box[0], last.box[0]);
            if (jl < _jl) {
                nearest = Number(i);
                _jl = jl;
            }
        }
        if (nearest === null) {
            columns.push([b]);
            return;
        }

        const last = columns[nearest].at(-1) as resultType[0]; // 前面已经遍历过了，有-1的才能赋值到nearest
        const thisW = Box.inlineSize(b.box);
        const lastW = Box.inlineSize(last.box);
        const minW = Math.min(thisW, lastW);
        const em = Box.blockSize(b.box);

        if (
            // 左右至少有一边是相近的，中心距离要相近
            // 行之间也不要离太远
            (Box.inlineStartDis(b.box, last.box) < 3 * em ||
                Box.inlineEndDis(b.box, last.box) < 3 * em ||
                Box.dis1(Box.inlineCenter(b.box), Box.inlineCenter(last.box)) < minW * 0.4) &&
            Box.blockGap(b.box, last.box) < em * 1.1
        ) {
        } else {
            columns.push([b]);
            return;
        }

        columns[nearest].push(b);
    }

    function joinResult(p: resultType) {
        const cjkv = /\p{Ideographic}/u;
        const cjkf = /[。，！？；：“”‘’《》、【】（）…—]/;
        const res: resultType[0] = {
            box: outerRect(p.map((i) => i.box)),
            text: "",
            mean: average2(p.map((i) => [i.mean, i.text.length])),
            style: p[0].style,
        };
        for (const i of p) {
            const lastChar = res.text.at(-1);
            if (
                lastChar &&
                ((!lastChar.match(cjkv) && !lastChar.match(cjkf)) ||
                    (!i.text.at(0)?.match(cjkv) && !i.text.at(0)?.match(cjkf)))
            )
                res.text += " ";
            res.text += i.text;
        }
        return res;
    }

    // 获取角度 竖排 横排

    /** 以x轴为正方向，图形学坐标 */
    const rAngle = {
        inline: 0,
        block: 90,
    };
    const inlineAngles = l.map((i) => {
        const b = i.box;
        const w = b[1][0] - b[0][0];
        const h = b[3][1] - b[0][1];
        let v = { x: 0, y: 0 };
        if (w < h) {
            const p = Vector.fromPonts(Point.center(b[2], b[3]), Point.center(b[0], b[1]));
            v = { x: p[0], y: p[1] };
        } else {
            const p = Vector.fromPonts(Point.center(b[1], b[2]), Point.center(b[0], b[3]));
            v = { x: p[0], y: p[1] };
        }
        const a = normalAngle(Math.atan2(v.y, v.x) * (180 / Math.PI));
        return { b, a };
    });
    const iav = normalAngle(averLineAngles(inlineAngles.map((i) => i.a)));
    const inlineangle = normalAngle(averLineAngles(inlineAngles.flatMap((i) => (lineAngleNear(i.a, iav) ? i.a : []))));

    const blockangle = normalAngle(inlineangle + 90);

    const inlineDir = lineAngleNear(inlineangle, 0) ? "x" : "y";
    const blockDir = lineAngleNear(blockangle, 90) ? "y" : "x";
    const fdir = dirs.find((d) => inlineDir === dir2xy(d.inline) && blockDir === dir2xy(d.block)) ?? dirs.at(0);
    if (fdir) {
        dir.block = fdir.block;
        dir.inline = fdir.inline;
    }

    const tipAngle: Record<ReadingDirPart, number> = {
        lr: 0,
        rl: 180,
        tb: 90,
        bt: 270,
    };
    rAngle.inline = smallest([inlineangle, inlineangle - 180, inlineangle + 180], (a) =>
        Math.abs(a - tipAngle[dir.inline]),
    );
    rAngle.block = smallest([blockangle, blockangle - 180, blockangle + 180], (a) => Math.abs(a - tipAngle[dir.block]));

    dirVector.inline = [Math.cos(rAngle.inline * (Math.PI / 180)), Math.sin(rAngle.inline * (Math.PI / 180))];
    dirVector.block = [Math.cos(rAngle.block * (Math.PI / 180)), Math.sin(rAngle.block * (Math.PI / 180))];

    log("dir", dir, rAngle, dirVector, inlineangle, blockangle);

    // 按照阅读方向，把box内部点重新排序
    const reOrderMapX = [
        [dir.inline[0], dir.block[0]],
        [dir.inline[1], dir.block[0]],
        [dir.inline[1], dir.block[1]],
        [dir.inline[0], dir.block[1]],
    ];
    const reOrderMap = reOrderMapX.map(
        ([i, b]) =>
            ({
                lt: 0,
                rt: 1,
                rb: 2,
                lb: 3,
            })[i === "l" || i === "r" ? i + b : b + i],
    ) as number[];
    const xyT = transXY({ inline: "lr", block: "tb" }, dir);
    const logicL = l.map((i) => {
        const b = structuredClone(i.box);
        const newBox = [b[reOrderMap[0]], b[reOrderMap[1]], b[reOrderMap[2]], b[reOrderMap[3]]] as BoxType;
        for (const x of newBox) {
            const [a, b] = xyT(x);
            x[0] = a;
            x[1] = b;
        }

        return {
            ...i,
            box: newBox,
        };
    });

    // 不考虑整体旋转，只考虑倾斜
    baseVector.inline = xyT(dirVector.inline);
    baseVector.block = xyT(dirVector.block);
    log("相对坐标系", baseVector);

    // 短轴扩散，合并为段

    const newL_ = logicL.sort((a, b) => Box.blockStart(a.box) - Box.blockStart(b.box));
    const newLZ: resultType[0][][] = [];
    // 合并行
    for (const j of newL_) {
        const last = newLZ.at(-1)?.at(-1);
        if (!last) {
            newLZ.push([j]);
            continue;
        }
        const thisCy = Box.blockCenter(j.box);
        const lastCy = Box.blockCenter(last.box);
        if (Math.abs(thisCy - lastCy) < 0.5 * Box.blockSize(j.box)) {
            const lLast = newLZ.at(-1);
            if (!lLast) {
                newLZ.push([j]);
            } else {
                lLast.push(j);
            }
        } else {
            newLZ.push([j]);
        }
    }

    // 根据距离，合并或保持拆分
    const newL: (resultType[0] | null)[] = [];
    for (const l of newLZ) {
        if (l.length === 1) {
            newL.push(l.at(0)!);
            continue;
        }

        const em = average(l.map((i) => Box.blockSize(i.box)));
        l.sort((a, b) => Box.inlineStart(a.box) - Box.inlineStart(b.box));

        let last = l.at(0)!;

        for (const this_ of l.slice(1)) {
            const lastBoxInlineEnd = Box.inlineEnd(last.box) ?? Number.NEGATIVE_INFINITY;
            const thisInlineStart = Box.inlineStart(this_.box);
            if (thisInlineStart - lastBoxInlineEnd > em) {
                newL.push(last);
                last = this_;
            } else {
                last.text += this_.text;
                last.mean = (last.mean + this_.mean) / 2;
                last.box = outerRect([last.box, this_.box]);
            }
        }
        newL.push(last);
    }

    // todo 分割线为边界
    // 分栏

    // 按很细的粒度去分栏
    const columns: resultType[] = [];

    const minY = newL.reduce(
        (a, b) => Math.min(a, Math.min(b?.box[0][1] ?? 0, b?.box[1][1] ?? 0)),
        Number.POSITIVE_INFINITY,
    );
    const maxY = newL.reduce(
        (a, b) => Math.max(a, Math.max(b?.box[2][1] ?? 0, b?.box[3][1] ?? 0)),
        Number.NEGATIVE_INFINITY,
    );
    for (let i = minY; i <= maxY; i++) {
        for (const j in newL) {
            const b = newL[j];
            if (!b) continue;
            if (Box.blockStart(b.box) > i) break;
            if (Box.blockStart(b.box) <= i && i <= Box.blockEnd(b.box)) {
                pushColumn(b);
                newL[j] = null;
            }
        }
    }

    const columnsInYaxis: { src: resultType; outerBox: BoxType; x: number; w: number }[][] = [];
    for (const [i, c] of columns.entries()) {
        const outer = outerRect(c.map((b) => b.box));
        const x = Box.blockCenter(outer);
        const w = Box.inlineSize(outer);
        if (i === 0) {
            columnsInYaxis.push([{ src: c, outerBox: outer, x, w }]);
            continue;
        }
        const l = columnsInYaxis.find((oc) => {
            const r = oc.at(-1)!;
            const em = Box.blockSize(c.at(0)!.box);
            if (
                Box.inlineStartDis(r.outerBox, outer) < 3 * em &&
                Box.inlineEndDis(r.outerBox, outer) < 3 * em &&
                Box.blockGap(outer, r.outerBox) < em * 2.1
            )
                return true;
            return false;
        });
        if (l) {
            l.push({ src: c, outerBox: outer, x, w });
        } else {
            columnsInYaxis.push([{ src: c, outerBox: outer, x, w }]);
        }
    }

    for (const y of columnsInYaxis) {
        y.sort((a, b) => Box.blockStart(a.outerBox) - Box.blockStart(b.outerBox));
    }

    const newColumns: { src: resultType; outerBox: BoxType }[] = [];

    for (const c of columnsInYaxis) {
        const o = outerRect(c.map((i) => i.outerBox));
        const s = c.flatMap((i) => i.src);
        newColumns.push({ src: s, outerBox: o });
    }

    newColumns.sort((a, b) => {
        const em = a.src.at(0) ? Box.blockSize(a.src.at(0)!.box) : 2;
        if (Box.dis1(Box.blockStart(a.outerBox), Box.blockStart(b.outerBox)) < em) {
            return Box.inlineStart(a.outerBox) - Box.inlineStart(b.outerBox);
        }
        return Box.blockStart(a.outerBox) - Box.blockStart(b.outerBox);
    });

    // 宽度相近的行都合并了，但有两中不合并的，以行20字为例子：(1)20,20,2,20,20 (2)20,20,10,10,10,10,20]
    // 分别为段末和分栏
    // 合并情况：中间短的行数多
    const mergedColumns: typeof newColumns = [];
    for (const c of newColumns) {
        const last = mergedColumns.at(-1);
        if (!last) {
            mergedColumns.push(c);
            continue;
        }
        const lastOuter = last.outerBox;
        const em = Box.blockSize(c.src[0].box);
        if (
            (last.src.length === 1 && Box.inlineStartDis(lastOuter, c.outerBox) < 3 * em) || // 标题
            (c.src.length === 1 && Box.inlineStartDis(lastOuter, c.outerBox) < 3 * em) || // 末尾
            (Box.inlineStartDis(lastOuter, c.outerBox) < 3 * em && Box.inlineEndDis(lastOuter, c.outerBox) < 3 * em) // 前面短的合并了，后面也合并上去
        ) {
            last.src.push(...c.src);
            last.outerBox = outerRect(last.src.map((i) => i.box));
        } else {
            mergedColumns.push(c);
        }
    }

    if (dev) {
        const color: string[] = [];
        for (let h = 0; h < 360; h += Math.floor(360 / newColumns.length)) {
            color.push(`hsl(${h}, 100%, 50%)`);
        }

        for (const i in newColumns) {
            for (const b of newColumns[i].src) {
                // drawBox(b.box, b.text, color[i]);
            }
        }
    }

    // 长轴扩散，合并为行
    // 合并为段落

    const p = mergedColumns.map((v) => {
        const c = v.src;

        const distanceCounts: Record<number, number> = {};
        for (let i = 1; i < c.length; i++) {
            const b1 = c[i - 1].box;
            const b2 = c[i].box;
            const dis = Box.blockCenter(b2) - Box.blockCenter(b1);
            if (!distanceCounts[dis]) distanceCounts[dis] = 0;
            distanceCounts[dis]++;
        }

        // 聚类
        const avgLineHeight = average(c.map((i) => i.box[3][1] - i.box[0][1])); // todo 众数
        const distanceGroup: number[][] = [[]];
        for (const d of Object.keys(distanceCounts)
            .map((i) => Number(i))
            .sort()) {
            const lastG = distanceGroup.at(-1)!;
            const lastI = lastG.at(-1);
            if (lastI !== undefined) {
                if (Math.abs(lastI - d) < avgLineHeight * 0.5) {
                    lastG.push(d);
                } else {
                    distanceGroup.push([]);
                }
            } else {
                lastG.push(d);
            }
        }

        const d =
            distanceGroup
                .map((g) => average(g))
                .sort((a, b) => a - b)
                .at(0) || 0;

        log("d", distanceCounts, distanceGroup, d);

        const ps: resultType[] = [[c[0]]];
        let lastPara = c[0];
        for (let i = 1; i < c.length; i++) {
            const expectY = Box.blockCenter(lastPara.box) + d;
            const thisLeftCenter = Box.blockCenter(c[i].box);
            const em = Box.blockSize(c[i].box);
            // 上一行右侧不靠近外框 或 理论此行与实际有差别，即空行或行首空格
            if (
                Box.inlineEndDis(lastPara.box, v.outerBox) > 2 * em ||
                r([Box.inlineStart(v.outerBox), expectY], [Box.inlineStart(c[i].box), thisLeftCenter]) > em * 0.5
            ) {
                ps.push([c[i]]);
            } else {
                const last = ps.at(-1);
                if (!last) ps.push([c[i]]);
                else last.push(c[i]);
            }

            lastPara = c[i];
        }

        // todo 识别python类代码
        // todo 计算前缀空格和向上换行

        const xyT = transXY(dir, { inline: "lr", block: "tb" });

        for (const x of c) {
            for (const p of x.box) {
                const [a, b] = xyT(p);
                p[0] = a;
                p[1] = b;
            }
        }

        const vo = v.outerBox;
        for (const p of vo) {
            const [a, b] = xyT(p);
            p[0] = a;
            p[1] = b;
        }
        // todo reorder

        log(ps);
        return {
            src: c,
            outerBox: v.outerBox,
            parragraphs: ps.map((p) => ({ src: p, parse: joinResult(p) as resultType[0] })),
        };
    });

    const pss = p.flatMap((v) => v.parragraphs.map((p) => p.parse)) as resultType;

    return {
        columns: p,
        parragraphs: pss,
    };
}

function average(args: number[]) {
    return args.reduce((a, b) => a + b, 0) / args.length;
}
function average2(args: [number, number][]) {
    const xsum = args.map((i) => i[1]).reduce((a, b) => a + b, 0);
    let n = 0;
    for (const i of args) {
        n += (i[0] * i[1]) / xsum;
    }
    return n;
}

/**
 * 0-360
 */
function normalAngle(angle: number) {
    return ((angle % 360) + 360) % 360;
}

function rotateImg(img: ImageData, angle: number) {
    const a = normalAngle(angle);
    if (a === 0) return img;
    if (![90, 180, 270].includes(a)) throw new Error("只支持90度的旋转");
    const newData = new Uint8ClampedArray(img.height * img.width * 4);

    for (let y = 0; y < img.height; y++) {
        for (let x = 0; x < img.width; x++) {
            const index = y * img.width + x;
            const newIndex =
                a === 90
                    ? x * img.height + (img.height - y - 1)
                    : a === 180
                      ? img.width - x - 1 + (img.height - y - 1) * img.width
                      : (img.width - x - 1) * img.height + y;
            newData.set(img.data.slice(index * 4, index * 4 + 4), newIndex * 4);
        }
    }

    const newWidth = a === 90 || a === 270 ? img.height : img.width;
    const newHeight = a === 90 || a === 270 ? img.width : img.height;
    return createImageData(newData, newWidth, newHeight);
}

function drawBox(box: BoxType, id = "", color = "red", qid?: string) {
    if (!dev) return;
    const canvas = document.querySelector(qid ? `#${qid}` : "canvas") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.moveTo(box[0][0], box[0][1]);
    ctx.lineTo(box[1][0], box[1][1]);
    ctx.lineTo(box[2][0], box[2][1]);
    ctx.lineTo(box[3][0], box[3][1]);
    ctx.lineTo(box[0][0], box[0][1]);
    ctx.stroke();
    ctx.strokeStyle = "black";
    ctx.strokeText(id, box[0][0], box[0][1]);
}

function drawBox2(box: BoxType, id = "", bg = "white", color = "red") {
    if (!dev) return;
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.fillStyle = bg;
    ctx.rect(box[0][0], box[0][1], 16, 16);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillRect(box[0][0], box[0][1], 6, 6);
}
