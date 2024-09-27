// biome-ignore lint/suspicious/noImplicitAnyLet: 可自定义cv
let cv;
let ort: typeof import("onnxruntime-common");

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

const task = new tLog("t");
const task2 = new tLog("af_det");

function putImgDom(img: HTMLElement) {
    try {
        document?.body?.append(img);
    } catch (error) {}
}

let createImageData = (data: Uint8ClampedArray, w: number, h: number) => {
    return new ImageData(data, w, h);
};

function log(...args: any[]) {
    if (dev) console.log(...args);
}

export { init, x as ocr, Det as det, Rec as rec };
export type initType = AsyncType<ReturnType<typeof init>>;

let dev = true;
let det: SessionType;
let rec: SessionType;
let layout: SessionType;
let dic: string[];
let limitSideLen = 960;
let imgH = 48;
let imgW = 320;
let detShape = [Number.NaN, Number.NaN];
let layoutDic: string[];

async function init(op: {
    detPath?: string;
    recPath?: string;
    layoutPath?: string;
    dic?: string;
    layoutDic?: string;
    dev?: boolean;
    maxSide?: number;
    imgh?: number;
    imgw?: number;
    ort: typeof import("onnxruntime-common");
    detShape?: [number, number];
    ortOption: import("onnxruntime-common").InferenceSession.SessionOptions;

    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    canvas?: (w: number, h: number) => any;
    imageData?;
    cv?;
}) {
    ort = op.ort;
    dev = op.dev;
    if (!dev) {
        task.l = () => {};
        task2.l = () => {};
    }
    if (op.detPath) det = await ort.InferenceSession.create(op.detPath, op.ortOption);
    if (op.recPath) rec = await ort.InferenceSession.create(op.recPath, op.ortOption);
    if (op.layoutPath) layout = await ort.InferenceSession.create(op.layoutPath, op.ortOption);
    dic = op.dic.split(/\r\n|\r|\n/);
    if (dic.at(-1) === "") {
        // 多出的换行
        dic[dic.length - 1] = " ";
    } else {
        dic.push(" ");
    }
    layoutDic = op.layoutDic?.split(/\r\n|\r|\n/);
    if (op.maxSide) limitSideLen = op.maxSide;
    if (op.imgh) imgH = op.imgh;
    if (op.imgw) imgW = op.imgw;
    if (op.detShape) detShape = op.detShape;
    if (op.canvas) setCanvas(op.canvas);
    if (op.imageData) createImageData = op.imageData;
    if (op.cv) cv = op.cv;
    else if (typeof require !== "undefined") cv = require("opencv.js");
    return { ocr: x, det: Det, rec: Rec };
}

/** 主要操作 */
async function x(img: ImageData) {
    task.l("");

    if (layout) {
        const sr = await runLayout(img, ort, layout, layoutDic);
    }

    const box = await Det(img);

    const mainLine = await Rec(box);
    // const mainLine = box.map((i, n) => ({ text: n.toString(), box: i.box, mean: 1 }));
    const newMainLine = afAfRec(mainLine);
    console.log(mainLine, newMainLine);
    task.l("end");
    return { src: mainLine, ...newMainLine };
}

async function Det(img: ImageData) {
    let h = img.height;
    let w = img.width;
    const _r = 0.6;
    const _h = h;
    const _w = w;
    if (_h < _w * _r || _w < _h * _r) {
        if (_h < _w * _r) h = Math.floor(_w * _r);
        if (_w < _h * _r) w = Math.floor(_h * _r);
        const c = newCanvas(w, h);
        const ctx = c.getContext("2d");
        ctx.putImageData(img, 0, 0);
        // biome-ignore lint: 规范化
        img = ctx.getImageData(0, 0, w, h);
    }

    task.l("pre_det");
    const { transposedData, image } = beforeDet(img, detShape[0], detShape[1]);
    task.l("det");
    const detResults = await runDet(transposedData, image, det);

    task.l("aft_det");
    const box = afterDet(detResults.data, detResults.dims[3], detResults.dims[2], img);
    return box;
}

async function Rec(box: { box: BoxType; img: ImageData }[]) {
    let mainLine: resultType = [];
    task.l("bf_rec");
    const recL = beforeRec(box);
    const recPromises = recL.map(async (item) => {
        const { b, imgH, imgW } = item;
        const recResults = await runRec(b, imgH, imgW, rec);
        return afterRec(recResults, dic);
    });
    const l = await Promise.all(recPromises);
    mainLine = l.flat().reverse();
    task.l("rec_end");
    for (const i in mainLine) {
        const b = box[mainLine.length - Number(i) - 1].box;
        mainLine[i].box = b;
    }
    mainLine = mainLine.filter((x) => x.mean >= 0.5);
    return mainLine;
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

function beforeDet(image: ImageData, shapeH: number, shapeW: number) {
    let ratio = 1;
    const h = image.height;
    const w = image.width;
    if (Math.max(h, w) > limitSideLen) {
        if (h > w) {
            ratio = limitSideLen / h;
        } else {
            ratio = limitSideLen / w;
        }
    }
    let resizeH = shapeH || h * ratio;
    let resizeW = shapeW || w * ratio;

    if (dev) {
        const srcCanvas = data2canvas(image);
        putImgDom(srcCanvas);
    }

    resizeH = Math.max(Math.round(resizeH / 32) * 32, 32);
    resizeW = Math.max(Math.round(resizeW / 32) * 32, 32);
    // biome-ignore lint: 规范化
    image = resizeImg(image, resizeW, resizeH);

    const transposedData = toPaddleInput(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    console.log(image);
    if (dev) {
        const srcCanvas = data2canvas(image);
        putImgDom(srcCanvas);
    }
    return { transposedData, image };
}

function afterDet(data: AsyncType<ReturnType<typeof runDet>>["data"], w: number, h: number, srcData: ImageData) {
    task2.l("");
    const myImageData = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < data.length; i++) {
        const n = i * 4;
        const v = (data[i] as number) > 0.3 ? 255 : 0;
        myImageData[n] = myImageData[n + 1] = myImageData[n + 2] = v;
        myImageData[n + 3] = 255;
    }
    task2.l("edge");

    const edgeRect: { box: BoxType; img: ImageData }[] = [];

    let src = cvImRead(createImageData(myImageData, w, h));

    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
        task2.l("get_box");
        const minSize = 3;
        const cnt = contours.get(i);
        const { points, sside } = getMiniBoxes(cnt);
        if (sside < minSize) continue;
        // TODO sort fast

        const clipBox = unclip(points);

        const boxMap = new cv.matFromArray(clipBox.length / 2, 1, cv.CV_32SC2, clipBox);

        const resultObj = getMiniBoxes(boxMap);
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

        task2.l("crop");

        const c = getRotateCropImage(srcData, box);

        edgeRect.push({ box, img: c });
    }
    task2.l("e");

    console.log(edgeRect);

    src.delete();
    contours.delete();
    hierarchy.delete();

    src = contours = hierarchy = null;

    return edgeRect;
}

type pointType = [number, number];
type BoxType = [pointType, pointType, pointType, pointType];
type pointsType = pointType[];
type resultType = { text: string; mean: number; box?: BoxType }[];
import clipper from "js-clipper";
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
function unclip(box: pointsType) {
    const unclip_ratio = 1.5;
    const area = Math.abs(polygonPolygonArea(box));
    const length = polygonPolygonLength(box);
    const distance = (area * unclip_ratio) / length;
    const tmpArr = [];
    for (const item of box) {
        const obj = {
            X: 0,
            Y: 0,
        };
        obj.X = item[0];
        obj.Y = item[1];
        tmpArr.push(obj);
    }
    const offset = new clipper.ClipperOffset();
    offset.AddPath(tmpArr, clipper.JoinType.jtRound, clipper.EndType.etClosedPolygon);
    const expanded: { X: number; Y: number }[][] = [];
    offset.Execute(expanded, distance);
    let expandedArr: pointsType = [];
    for (const item of expanded[0] || []) {
        expandedArr.push([item.X, item.Y]);
    }
    expandedArr = [].concat(...expandedArr);

    return expandedArr;
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

function getMiniBoxes(contour: any) {
    const boundingBox = cv.minAreaRect(contour);
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
    const img_crop_width = int(Math.max(linalgNorm(points[0], points[1]), linalgNorm(points[2], points[3])));
    const img_crop_height = int(Math.max(linalgNorm(points[0], points[3]), linalgNorm(points[1], points[2])));
    const pts_std = [
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height],
    ];

    const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, flatten(points));
    const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, flatten(pts_std));

    // 获取到目标矩阵
    const M = cv.getPerspectiveTransform(srcTri, dstTri);
    const src = cvImRead(img);
    const dst = new cv.Mat();
    const dsize = new cv.Size(img_crop_width, img_crop_height);
    // 透视转换
    cv.warpPerspective(src, dst, M, dsize, cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());

    const dst_img_height = dst.matSize[0];
    const dst_img_width = dst.matSize[1];
    // biome-ignore lint/suspicious/noImplicitAnyLet: <explanation>
    let dst_rot;
    // 图像旋转
    if (dst_img_height / dst_img_width >= 1.5) {
        dst_rot = new cv.Mat();
        const dsize_rot = new cv.Size(dst.rows, dst.cols);
        const center = new cv.Point(dst.cols / 2, dst.cols / 2);
        const M = cv.getRotationMatrix2D(center, 90, 1);
        cv.warpAffine(dst, dst_rot, M, dsize_rot, cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
    }

    const d = cvImShow(dst_rot || dst);

    src.delete();
    dst.delete();
    srcTri.delete();
    dstTri.delete();
    return d;
}

function cvImRead(img: ImageData) {
    return cv.matFromImageData(img);
}

function cvImShow(mat) {
    const img = new cv.Mat();
    const depth = mat.type() % 8;
    const scale = depth <= cv.CV_8S ? 1 : depth <= cv.CV_32S ? 1 / 256 : 255;
    const shift = depth === cv.CV_8S || depth === cv.CV_16S ? 128 : 0;
    mat.convertTo(img, cv.CV_8U, scale, shift);
    switch (img.type()) {
        case cv.CV_8UC1:
            cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA);
            break;
        case cv.CV_8UC3:
            cv.cvtColor(img, img, cv.COLOR_RGB2RGBA);
            break;
        case cv.CV_8UC4:
            break;
        default:
            throw new Error("Bad number of channels (Source image must have 1, 3 or 4 channels)");
    }
    const imgData = createImageData(new Uint8ClampedArray(img.data), img.cols, img.rows);
    img.delete();
    return imgData;
}

function beforeRec(box: { box: BoxType; img: ImageData }[]) {
    const l: { b: number[][][]; imgH: number; imgW: number }[] = [];
    function resizeNormImg(img: ImageData) {
        const w = Math.floor(imgH * (img.width / img.height));
        const d = resizeImg(img, w, imgH);
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
        return character.at(i - 1);
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
        const charList = [];
        const confList = [];
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

// TODO 使用板式识别代替
/** 组成行 */
function afAfRec(l: resultType) {
    log(l);

    // 获取角度 竖排 横排

    // 短轴扩散，合并为段
    // todo 分割线为边界
    // 分栏

    const columns: resultType[] = [];

    const newL = structuredClone(l).sort((a, b) => a.box[0][1] - b.box[0][1]);
    const maxY = newL.reduce((a, b) => Math.max(a, b.box[2][1]), 0);
    for (let i = 0; i <= maxY; i++) {
        for (const j in newL) {
            const b = newL[j];
            if (!b) continue;
            if (b.box[0][1] > i) break;
            if (b.box[0][1] <= i && i <= b.box[2][1]) {
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

    function average(args: number[]) {
        return args.reduce((a, b) => a + b, 0) / args.length;
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
        let nearest: number = null;
        let _jl = Number.POSITIVE_INFINITY;
        for (const i in columns) {
            const last = columns[i].at(-1);
            const jl = r(centerPoint(b.box), centerPoint(last.box));
            if (jl < _jl) {
                nearest = Number(i);
                _jl = jl;
            }
        }
        if (nearest === null) {
            columns.push([b]);
            return;
        }

        const last = columns[nearest].at(-1);
        const thisW = b.box[1][0] - b.box[0][0];
        const lastW = last.box[1][0] - last.box[0][0];
        const minW = Math.min(thisW, lastW);
        const em = b.box[2][1] - b.box[0][1];

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
        const lastY = columnsInYaxis.at(-1);
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
        const color = [];
        for (let h = 0; h < 360; h += Math.floor(360 / newColumns.length)) {
            color.push(`hsl(${h}, 100%, 50%)`);
        }

        for (const i in newColumns) {
            for (const b of newColumns[i].src) {
                drawBox(b.box, b.text, color[i]);
            }
        }
    }

    // 长轴扩散，合并为行

    const p = newColumns.map((v) => {
        const c = v.src;
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
            let maxGap = 0;
            let maxGapDelta = 0;
            const maxN = Math.max(...Object.values(gs));
            const gapsL = Object.keys(gs)
                .sort()
                .map(Number)
                .filter((g) => g > 0)
                .filter((g) => gs[g] !== maxN); // 去掉一个最大值
            for (let i = 1; i < gapsL.length; i++) {
                const delta = Math.abs((gs[gapsL[i]] - gs[gapsL[i - 1]]) / (gapsL[i] - gapsL[i - 1]));
                if (delta >= maxGapDelta) {
                    maxGap = gapsL[i];
                    maxGapDelta = delta;
                }
            }
            splitGap = maxGap;
        } else {
            splitGap = Number(Object.keys(gs)[0] || 0);
        }
        log(splitGap);

        const ps: resultType[] = [[c[0]]];
        for (let i = 1; i < c.length; i++) {
            const b1 = c[i - 1].box;
            const b2 = c[i].box;
            const gap = b2[0][1] - b1[2][1];
            if (gap >= splitGap) {
                ps.push([c[i]]);
            } else {
                if (!ps.at(-1)) ps.push([]);
                ps.at(-1).push(c[i]);
            }
        }
        log(ps);
        return { src: c, outerBox: v.outerBox, parragraphs: ps.map((p) => ({ src: p, parse: joinResult(p) })) };
    });

    const pss = p.flatMap((v) => v.parragraphs.map((p) => p.parse));

    function joinResult(p: resultType) {
        const cjkv = /\p{Ideographic}/u;
        const res: resultType[0] = {
            box: outerRect(p.map((i) => i.box)),
            text: "",
            mean: average(p.map((i) => i.mean)),
        };
        for (const i of p) {
            if (res.text.at(-1) && (!res.text.at(-1).match(cjkv) || !i.text.at(0)?.match(cjkv))) res.text += " ";
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

function drawBox(box: BoxType, id = "", color = "red") {
    if (!dev) return;
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.rect(box[0][0], box[0][1], box[2][0] - box[0][0], box[2][1] - box[0][1]);
    ctx.stroke();
    ctx.strokeText(id, box[0][0], box[0][1]);
}
