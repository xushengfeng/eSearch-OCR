import { runLayout } from "./layout";
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

export { init, x as ocr, Det as det, Rec as rec };
export type initType = AsyncType<ReturnType<typeof init>>;

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

let ort: typeof import("onnxruntime-common");

const task = new tLog("t");
const task2 = new tLog("af_det");

let dev = true;
let det: SessionType;
let rec: SessionType;
let layout: SessionType;
let dic: string[];
let imgH = 48;
let detRatio = 1;
let layoutDic: string[];

let onProgress = (type: "det" | "rec", total: number, count: number) => {};

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
    if (dev) console.log(...args);
}
function logSrc(...args: any[]) {
    if (dev) console.log(...args.map((i) => structuredClone(i)));
}

function logColor(...args: string[]) {
    if (dev) {
        console.log(args.map((x) => `%c${x}`).join(""), ...args.map((x) => `color: ${x}`));
    }
}

async function init(op: {
    detPath?: string;
    recPath?: string;
    layoutPath?: string;
    dic?: string;
    layoutDic?: string;
    dev?: boolean;
    imgh?: number;
    detRatio?: number;
    ort: typeof import("onnxruntime-common");
    ortOption?: import("onnxruntime-common").InferenceSession.SessionOptions;

    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    canvas?: (w: number, h: number) => any;
    imageData?;
    cv?;

    onProgress?: (type: "det" | "rec", total: number, count: number) => void;
}) {
    ort = op.ort;
    dev = Boolean(op.dev);
    if (!dev) {
        task.l = () => {};
        task2.l = () => {};
    }
    if (op.detPath) det = await ort.InferenceSession.create(op.detPath, op.ortOption);
    if (op.recPath) rec = await ort.InferenceSession.create(op.recPath, op.ortOption);
    if (op.layoutPath) layout = await ort.InferenceSession.create(op.layoutPath, op.ortOption);
    dic = op.dic?.split(/\r\n|\r|\n/) || [];
    if (dic.at(-1) === "") {
        // 多出的换行
        dic[dic.length - 1] = " ";
    } else {
        dic.push(" ");
    }
    layoutDic = op.layoutDic?.split(/\r\n|\r|\n/) || [];
    if (op.imgh) imgH = op.imgh;
    if (op.detRatio) detRatio = op.detRatio;
    if (op.canvas) setCanvas(op.canvas);
    if (op.imageData) createImageData = op.imageData;
    if (op.onProgress) onProgress = op.onProgress;
    return { ocr: x, det: Det, rec: Rec };
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

/** 主要操作 */
async function x(srcimg: loadImgType) {
    task.l("");

    const img = await loadImg(srcimg);

    if (layout) {
        const sr = await runLayout(img, ort, layout, layoutDic);
    }

    const box = await Det(img);

    const mainLine = await Rec(box);
    // const mainLine = box.map((i, n) => ({ text: n.toString(), box: i.box, mean: 1 }));
    // if (dev)
    //     for (const x of box) {
    //         drawBox2(x.box, "hi", `rgb(${x.style.bg.join(",")})`, `rgb(${x.style.text.join(",")})`);
    //     }
    // return;
    const newMainLine = afAfRec(mainLine);
    log(mainLine, newMainLine);
    task.l("end");
    return { src: mainLine, ...newMainLine };
}

async function Det(srcimg: loadImgType) {
    const img = await loadImg(srcimg);

    if (dev) {
        const srcCanvas = data2canvas(img);
        putImgDom(srcCanvas);
    }

    task.l("pre_det");
    const { data: beforeDetData, width: resizeW, height: resizeH } = beforeDet(img);
    const { transposedData, image } = beforeDetData;
    task.l("det");
    onProgress("det", 1, 0);
    const detResults = await runDet(transposedData, image, det);

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

async function Rec(box: detResultType) {
    const mainLine: resultType = [];
    task.l("bf_rec");
    const recL = beforeRec(box);
    let runCount = 0;
    onProgress("rec", recL.length, runCount);
    const mainLine0: { text: string; mean: number }[] = [];
    for (const item of recL) {
        const { b, imgH, imgW } = item;
        const recResults = await runRec(b, imgH, imgW, rec);
        runCount++;
        onProgress("rec", recL.length, runCount);
        mainLine0.push(...afterRec(recResults, dic));
    }
    mainLine0.reverse();
    task.l("rec_end");
    for (const i in mainLine0) {
        const b = box[mainLine0.length - Number(i) - 1].box;
        mainLine[i] = {
            mean: mainLine0[i].mean,
            text: mainLine0[i].text,
            box: b,
            style: box[mainLine0.length - Number(i) - 1].style,
        };
    }
    return mainLine.filter((x) => x.mean >= 0.5) as resultType;
}

async function runDet(transposedData: number[][][], image: ImageData, det: SessionType) {
    const detData = Float32Array.from(transposedData.flat(3));

    const detTensor = new ort.Tensor("float32", detData, [1, 3, image.height, image.width]);
    const detFeed = {};
    detFeed[det.inputNames[0]] = detTensor;

    const detResults = await det.run(detFeed);
    return detResults[det.outputNames[0]];
}

async function runRec(b: number[][][], imgH: number, imgW: number, rec: SessionType) {
    const recData = Float32Array.from(b.flat(3));

    const recTensor = new ort.Tensor("float32", recData, [1, 3, imgH, imgW]);
    const recFeed = {};
    recFeed[rec.inputNames[0]] = recTensor;

    const recResults = await rec.run(recFeed);
    return recResults[rec.outputNames[0]];
}

function beforeDet(srcImg: ImageData) {
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

function beforeRec(box: { box: BoxType; img: ImageData }[]) {
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

    // 获取角度 竖排 横排

    // 短轴扩散，合并为段

    const newL_ = structuredClone(l).sort((a, b) => a.box[0][1] - b.box[0][1]) as resultType[0][];
    const newLZ: resultType[0][][] = [];
    // 合并行
    for (const j of newL_) {
        const last = newLZ.at(-1)?.at(-1);
        if (!last) {
            newLZ.push([j]);
            continue;
        }
        const thisCy = (j.box[2][1] + j.box[0][1]) / 2;
        const lastCy = (last.box[2][1] + last.box[0][1]) / 2;
        if (Math.abs(thisCy - lastCy) < 0.5 * (j.box[3][1] - j.box[0][1])) {
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
            newL.push(l.at(0) as resultType[0]);
            continue;
        }

        const em = average(l.map((i) => i.box[3][1] - i.box[0][1]));
        l.sort((a, b) => a.box[0][0] - b.box[0][0]);

        let last = l.at(0) as resultType[0];

        for (const this_ of l.slice(1)) {
            const lastBoxRightX = last.box[1][0] ?? Number.NEGATIVE_INFINITY;
            const thisLeftX = this_.box[0][0];
            if (thisLeftX - lastBoxRightX > em) {
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

    const columns: resultType[] = [];

    const maxY = newL.reduce((a, b) => Math.max(a, Math.max(b?.box[2][1] ?? 0, b?.box[3][1] ?? 0)), 0);
    for (let i = 0; i <= maxY; i++) {
        for (const j in newL) {
            const b = newL[j];
            if (!b) continue;
            if (b.box[0][1] > i) break;
            if (b.box[0][1] <= i && i <= b.box[3][1]) {
                pushColumn(b);
                newL[j] = null;
            }
        }
    }

    function centerPoint(points: pointsType) {
        const n = points.length;
        let x = 0;
        let y = 0;
        for (const p of points) {
            x += p[0];
            y += p[1];
        }
        return [x / n, y / n] as pointType;
    }

    function r(point: pointType, point2: pointType) {
        return Math.sqrt((point[0] - point2[0]) ** 2 + (point[1] - point2[1]) ** 2);
    }

    function outerRect(boxes: BoxType[]) {
        const [p0, p1, p2, p3] = structuredClone(boxes[0]);
        for (const b of boxes) {
            p0[0] = Math.min(p0[0], b[0][0]);
            p0[1] = Math.min(p0[1], b[0][1]);
            p1[0] = Math.max(p1[0], b[1][0]);
            p1[1] = Math.min(p1[1], b[1][1]);
            p2[0] = Math.max(p2[0], b[2][0]);
            p2[1] = Math.max(p2[1], b[2][1]);
            p3[0] = Math.min(p3[0], b[3][0]);
            p3[1] = Math.max(p3[1], b[3][1]);
        }
        return [p0, p1, p2, p3] as BoxType;
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
        const thisW = b.box[1][0] - b.box[0][0];
        const lastW = last.box[1][0] - last.box[0][0];
        const minW = Math.min(thisW, lastW);
        const em = b.box[3][1] - b.box[0][1];

        if (
            // 左右至少有一边是相近的，中心距离要相近
            // 特别是处理在长行后分栏的情况
            Math.abs(b.box[0][0] - last.box[0][0]) < em ||
            Math.abs(b.box[1][0] - last.box[1][0]) < em ||
            Math.abs((b.box[1][0] + b.box[0][0]) / 2 - (last.box[1][0] + last.box[0][0]) / 2) < minW * 0.4
        ) {
        } else {
            const smallBox = thisW < lastW ? b : last;
            const bigBox = thisW < lastW ? last : b;
            const maxW = Math.max(thisW, lastW);
            const ax = bigBox.box[0][0] + maxW / 2;
            const bx = bigBox.box[1][0] - maxW / 2;
            if (ax < smallBox.box[0][0] || bx > smallBox.box[1][0]) {
                columns.push([b]);
                return;
            }
        }

        columns[nearest].push(b);
    }

    const columnsInYaxis: { src: resultType; outerBox: BoxType; x: number; w: number }[][] = [];
    for (const [i, c] of columns.entries()) {
        const outer = outerRect(c.map((b) => b.box));
        const x = (outer[0][0] + outer[1][0]) / 2;
        const w = outer[1][0] - outer[0][0];
        if (i === 0) {
            columnsInYaxis.push([{ src: c, outerBox: outer, x, w }]);
            continue;
        }
        const lastY = columnsInYaxis.at(-1) as (typeof columnsInYaxis)[0]; // 上面的代码保证了至少有一个元素
        let hasSame = false;
        for (const r of lastY) {
            const minW = Math.min(r.w, w);
            if (Math.abs(r.x - x) < minW * 0.4) {
                lastY.push({ src: c, outerBox: outer, x, w });
                hasSame = true;
                break;
            }
        }
        if (!hasSame) {
            columnsInYaxis.push([{ src: c, outerBox: outer, x, w }]);
        }
    }

    columnsInYaxis.sort((a, b) => average(a.map((i) => i.x)) - average(b.map((i) => i.x)));

    for (const y of columnsInYaxis) {
        y.sort((a, b) => a.outerBox[0][1] - b.outerBox[0][1]);
    }

    const newColumns: { src: resultType; outerBox: BoxType }[] = columnsInYaxis.flat();

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

    const p = newColumns.map((v) => {
        const c = v.src;
        // gap feq
        const gs: Record<number, number> = {};
        for (let i = 1; i < c.length; i++) {
            const b1 = c[i - 1].box;
            const b2 = c[i].box;
            const gap = b2[0][1] - b1[2][1];
            if (!gs[gap]) gs[gap] = 0;
            gs[gap]++;
        }
        log(gs);

        let splitGap = 0;

        if (Object.keys(gs).length >= 2) {
            let maxN = Math.max(...Object.values(gs));
            let maxGap = 0;
            let maxGapDelta = 0;
            if (Object.values(gs).filter((i) => i === maxN)) maxN++; // 有多个最大值就不去了
            const gapsL = Object.keys(gs)
                .map(Number)
                .sort((a, b) => a - b)
                .filter((g) => gs[g] !== maxN); // 去掉一个最大值
            for (let i = 1; i < gapsL.length; i++) {
                const delta = Math.abs((gs[gapsL[i]] - gs[gapsL[i - 1]]) / (gapsL[i] - gapsL[i - 1]));
                if (delta >= maxGapDelta) {
                    maxGap = gapsL[i];
                    maxGapDelta = delta;
                }
            }
            splitGap = Math.max(maxGap, ((gapsL.at(0) as number) + (gapsL.at(-1) as number)) / 2);
        } else {
            splitGap = Number(Object.keys(gs)[0] || 0);
        }
        log(splitGap);

        // todo 相差太大也分段

        const ps: resultType[] = [[c[0]]];
        for (let i = 1; i < c.length; i++) {
            const b1 = c[i - 1].box;
            const b2 = c[i].box;
            const gap = b2[0][1] - b1[2][1];
            if (gap >= splitGap) {
                ps.push([c[i]]);
            } else {
                const last = ps.at(-1);
                if (!last) ps.push([c[i]]);
                else last.push(c[i]);
            }
        }
        log(ps);
        return {
            src: c,
            outerBox: v.outerBox,
            parragraphs: ps.map((p) => ({ src: p, parse: joinResult(p) as resultType[0] })),
        };
    });

    if (dev) {
        const color: string[] = [];
        for (let h = 10; h < 360; h += Math.floor(360 / p.length)) {
            color.push(`hsl(${h}, 100%, 50%)`);
        }

        for (const i in p) {
            for (const b of p[i].parragraphs) {
                drawBox(b.parse.box, b.parse.text, color[i]);
            }
        }
    }

    const pss = p.flatMap((v) => v.parragraphs.map((p) => p.parse)) as resultType;

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

    // 识别行首空格

    // for (const i of l) {
    //     drawBox(i.box);
    // }

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
