var cv = require("opencv.js");
var ort: typeof import("onnxruntime-web");

export { x as ocr, init };

var dev = true;
type AsyncType<T> = T extends Promise<infer U> ? U : never;
type SessionType = AsyncType<ReturnType<typeof import("onnxruntime-web").InferenceSession.create>>;
var det: SessionType, rec: SessionType, dic: string[];
var limitSideLen = 960,
    imgH = 48,
    imgW = 320;
var detShape = [NaN, NaN];

async function init(x: {
    detPath: string;
    recPath: string;
    dic: string;
    node?: boolean;
    dev?: boolean;
    maxSide?: number;
    imgh?: number;
    imgw?: number;
    ort?: typeof import("onnxruntime-web");
    detShape?: [number, number];
}) {
    if (x.ort) {
        ort = x.ort;
    } else {
        if (x.node) {
            ort = require("onnxruntime-node");
        } else {
            ort = require("onnxruntime-web");
        }
    }
    dev = x.dev;
    det = await ort.InferenceSession.create(x.detPath);
    rec = await ort.InferenceSession.create(x.recPath);
    dic = x.dic.split(/\r\n|\r|\n/);
    if (x.maxSide) limitSideLen = x.maxSide;
    if (x.imgh) imgH = x.imgh;
    if (x.imgw) imgW = x.imgw;
    if (x.detShape) detShape = x.detShape;
    return new Promise((rs) => rs(true));
}

/** 主要操作 */
async function x(img: ImageData) {
    console.time();
    let h = img.height,
        w = img.width;
    let { transposedData, image } = beforeDet(img, detShape[0], detShape[1]);
    const detResults = await runDet(transposedData, image, det);

    let box = afterDet(detResults.data, detResults.dims[3], detResults.dims[2], img);

    let mainLine: resultType = [];
    for (let i of beforeRec(box)) {
        let { b, imgH, imgW } = i;
        const recResults = await runRec(b, imgH, imgW, rec);
        if (dic.at(-1) == "") {
            // 多出的换行
            dic[dic.length - 1] = " ";
        } else {
            dic.push(" ");
        }
        let line = afterRec(recResults, dic);
        mainLine = line.concat(mainLine);
    }
    for (let i in mainLine) {
        let rx = w / image.width,
            ry = h / image.height;
        let b = box[mainLine.length - Number(i) - 1].box;
        for (let p of b) {
            p[0] = p[0] * rx;
            p[1] = p[1] * ry;
        }
        mainLine[i]["box"] = b;
    }
    mainLine = mainLine.filter((x) => x.mean >= 0.5);
    mainLine = afAfRec(mainLine);
    console.log(mainLine);
    console.timeEnd();
    return mainLine;
}

async function runDet(transposedData: number[][][], image: ImageData, det: SessionType) {
    let x = transposedData.flat(Infinity) as number[];
    const detData = Float32Array.from(x);

    const detTensor = new ort.Tensor("float32", detData, [1, 3, image.height, image.width]);
    let detFeed = {};
    detFeed[det.inputNames[0]] = detTensor;

    const detResults = await det.run(detFeed);
    return detResults[det.outputNames[0]];
}

async function runRec(b: number[][][], imgH: number, imgW: number, rec: SessionType) {
    const recData = Float32Array.from(b.flat(Infinity) as number[]);

    const recTensor = new ort.Tensor("float32", recData, [b.length, 3, imgH, imgW]);
    let recFeed = {};
    recFeed[rec.inputNames[0]] = recTensor;

    const recResults = await rec.run(recFeed);
    return recResults[rec.outputNames[0]];
}

function data2canvas(data: ImageData, w?: number, h?: number) {
    let x = document.createElement("canvas");
    x.width = w || data.width;
    x.height = h || data.height;
    x.getContext("2d").putImageData(data, 0, 0);
    return x;
}

/**
 *
 * @param {ImageData} data 原图
 * @param {number} w 输出宽
 * @param {number} h 输出高
 */
function resizeImg(data: ImageData, w: number, h: number) {
    let x = data2canvas(data);
    let src = document.createElement("canvas");
    src.width = w;
    src.height = h;
    src.getContext("2d").scale(w / data.width, h / data.height);
    src.getContext("2d").drawImage(x, 0, 0);
    return src.getContext("2d").getImageData(0, 0, w, h);
}

function beforeDet(image: ImageData, shapeH: number, shapeW: number) {
    let ratio = 1;
    let h = image.height,
        w = image.width;
    if (Math.max(h, w) > limitSideLen) {
        if (h > w) {
            ratio = limitSideLen / h;
        } else {
            ratio = limitSideLen / w;
        }
    }
    let resizeH = shapeH || h * ratio;
    let resizeW = shapeW || w * ratio;

    resizeH = Math.max(Math.round(resizeH / 32) * 32, 32);
    resizeW = Math.max(Math.round(resizeW / 32) * 32, 32);
    image = resizeImg(image, resizeW, resizeH);

    const transposedData = toPaddleInput(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    console.log(image);
    if (dev) {
        let srcCanvas = data2canvas(image);
        document.body.append(srcCanvas);
    }
    return { transposedData, image };
}

function afterDet(data: AsyncType<ReturnType<typeof runDet>>["data"], w: number, h: number, srcData: ImageData) {
    var myImageData = new ImageData(w, h);
    for (let i in data) {
        let n = Number(i) * 4;
        const v = (data[i] as number) > 0.3 ? 255 : 0;
        myImageData.data[n] = myImageData.data[n + 1] = myImageData.data[n + 2] = v;
        myImageData.data[n + 3] = 255;
    }
    let canvas = data2canvas(myImageData);

    let edgeRect: { box: BoxType; img: ImageData }[] = [];

    let src = cv.imread(canvas);

    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
        let minSize = 3;
        let cnt = contours.get(i);
        let { points, sside } = getMiniBoxes(cnt);
        if (sside < minSize) continue;
        // TODO sort fast

        let clipBox = unclip(points);

        const boxMap = new cv.matFromArray(clipBox.length / 2, 1, cv.CV_32SC2, clipBox);

        const resultObj = getMiniBoxes(boxMap);
        let box = resultObj.points;
        if (resultObj.sside < minSize + 2) {
            continue;
        }
        function clip(n: number, min: number, max: number) {
            return Math.max(min, Math.min(n, max));
        }

        let rx = srcData.width / w;
        let ry = srcData.height / h;

        for (let i = 0; i < box.length; i++) {
            box[i][0] *= rx;
            box[i][1] *= ry;
        }

        let box1 = orderPointsClockwise(box);
        box1.forEach((item) => {
            item[0] = clip(Math.round(item[0]), 0, srcData.width);
            item[1] = clip(Math.round(item[1]), 0, srcData.height);
        });
        let rect_width = int(linalgNorm(box1[0], box1[1]));
        let rect_height = int(linalgNorm(box1[0], box1[3]));
        if (rect_width <= 3 || rect_height <= 3) continue;

        let c0 = data2canvas(srcData);

        let c = getRotateCropImage(c0, box);

        edgeRect.push({ box, img: c.getContext("2d").getImageData(0, 0, c.width, c.height) });
    }

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
const clipper = require("js-clipper");
function polygonPolygonArea(polygon: pointsType) {
    let i = -1,
        n = polygon.length,
        a: pointType,
        b = polygon[n - 1],
        area = 0;

    while (++i < n) {
        a = b;
        b = polygon[i];
        area += a[1] * b[0] - a[0] * b[1];
    }

    return area / 2;
}
function polygonPolygonLength(polygon: pointsType) {
    let i = -1,
        n = polygon.length,
        b = polygon[n - 1],
        xa: number,
        ya: number,
        xb = b[0],
        yb = b[1],
        perimeter = 0;

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
    box.forEach((item) => {
        const obj = {
            X: 0,
            Y: 0,
        };
        obj.X = item[0];
        obj.Y = item[1];
        tmpArr.push(obj);
    });
    const offset = new clipper.ClipperOffset();
    offset.AddPath(tmpArr, clipper.JoinType.jtRound, clipper.EndType.etClosedPolygon);
    const expanded: { X: number; Y: number }[][] = [];
    offset.Execute(expanded, distance);
    let expandedArr: pointsType = [];
    expanded[0] &&
        expanded[0].forEach((item) => {
            expandedArr.push([item.X, item.Y]);
        });
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

    const rotatedPoints: any[] = [];

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
        (a, b) => a[0] - b[0]
    ) as pointsType;

    let index_1 = 0,
        index_2 = 1,
        index_3 = 2,
        index_4 = 3;
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

function int(num: number) {
    return num > 0 ? Math.floor(num) : Math.ceil(num);
}
function flatten(arr: number[] | number[][]) {
    return arr
        .toString()
        .split(",")
        .map((item) => +item);
}
function linalgNorm(p0: pointType, p1: pointType) {
    return Math.sqrt(Math.pow(p0[0] - p1[0], 2) + Math.pow(p0[1] - p1[1], 2));
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
function getRotateCropImage(img: HTMLCanvasElement | HTMLImageElement, points: BoxType) {
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
    const src = cv.imread(img);
    const dst = new cv.Mat();
    const dsize = new cv.Size(img_crop_width, img_crop_height);
    // 透视转换
    cv.warpPerspective(src, dst, M, dsize, cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
    console.log(dst);

    const dst_img_height = dst.matSize[0];
    const dst_img_width = dst.matSize[1];
    let dst_rot;
    // 图像旋转
    if (dst_img_height / dst_img_width >= 1.5) {
        dst_rot = new cv.Mat();
        const dsize_rot = new cv.Size(dst.rows, dst.cols);
        const center = new cv.Point(dst.cols / 2, dst.cols / 2);
        const M = cv.getRotationMatrix2D(center, 90, 1);
        cv.warpAffine(dst, dst_rot, M, dsize_rot, cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
    }

    let c = document.createElement("canvas");
    if (dst_rot) {
        c.width = dst_rot.matSize[1];
        c.height = dst_rot.matSize[0];
    } else {
        c.width = dst_img_width;
        c.height = dst_img_height;
    }
    cv.imshow(c, dst_rot || dst);
    if (dev) document.body.append(c);

    src.delete();
    dst.delete();
    srcTri.delete();
    dstTri.delete();
    return c;
}

function toPaddleInput(image: ImageData, mean: number[], std: number[]) {
    const imagedata = image.data;
    const redArray: number[][] = [];
    const greenArray: number[][] = [];
    const blueArray: number[][] = [];
    let x = 0,
        y = 0;
    for (let i = 0; i < imagedata.length; i += 4) {
        if (!blueArray[y]) blueArray[y] = [];
        if (!greenArray[y]) greenArray[y] = [];
        if (!redArray[y]) redArray[y] = [];
        redArray[y][x] = (imagedata[i] / 255 - mean[0]) / std[0];
        greenArray[y][x] = (imagedata[i + 1] / 255 - mean[1]) / std[1];
        blueArray[y][x] = (imagedata[i + 2] / 255 - mean[2]) / std[2];
        x++;
        if (x == image.width) {
            x = 0;
            y++;
        }
    }

    return [blueArray, greenArray, redArray];
}

function beforeRec(box: { box: BoxType; img: ImageData }[]) {
    let l: { b: number[][][]; imgH: number; imgW: number }[] = [];
    function resizeNormImg(img: ImageData) {
        imgW = Math.floor(imgH * maxWhRatio);
        let h = img.height,
            w = img.width;
        let ratio = w / h;
        let resizedW: number;
        if (Math.ceil(imgH * ratio) > imgW) {
            resizedW = imgW;
        } else {
            resizedW = Math.floor(Math.ceil(imgH * ratio));
        }
        let d = resizeImg(img, resizedW, imgH);
        let cc = data2canvas(d, imgW, imgH);
        if (dev) document.body.append(cc);
        return cc.getContext("2d").getImageData(0, 0, imgW, imgH);
    }

    let boxes = [];
    let nowWidth = 0;
    for (let i of box) {
        if (Math.abs(i.img.width - nowWidth) > 32) {
            nowWidth = i.img.width;
            boxes.push([i]);
        } else {
            if (!boxes[boxes.length - 1]) boxes.push([]);
            boxes[boxes.length - 1].push(i);
        }
    }
    let maxWhRatio = 0;
    for (let box of boxes) {
        maxWhRatio = 0;
        for (let r of box) {
            maxWhRatio = Math.max(r.img.width / r.img.height, maxWhRatio);
        }
        let b = [];
        for (let r of box) {
            b.push(toPaddleInput(resizeNormImg(r.img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]));
        }
        l.push({ b, imgH, imgW });
    }
    console.log(l);
    return l;
}

function afterRec(data: AsyncType<ReturnType<typeof runRec>>, character: string[]) {
    let predLen = data.dims[2];
    let line: { text: string; mean: number }[] = [];
    let ml = data.dims[0] - 1;
    for (let l = 0; l < data.data.length; l += predLen * data.dims[1]) {
        const predsIdx: number[] = [];
        const predsProb: number[] = [];

        for (let i = l; i < l + predLen * data.dims[1]; i += predLen) {
            const tmpArr = data.data.slice(i, i + predLen) as Float32Array;
            const tmpMax = tmpArr.reduce((a, b) => Math.max(a, b), -Infinity);
            const tmpIdx = tmpArr.indexOf(tmpMax);
            predsProb.push(tmpMax);
            predsIdx.push(tmpIdx);
        }
        line[ml] = decode(predsIdx, predsProb, true);
        ml--;
    }
    function decode(textIndex: number[], textProb: number[], isRemoveDuplicate: boolean) {
        const ignoredTokens = [0];
        const charList = [];
        const confList = [];
        for (let idx = 0; idx < textIndex.length; idx++) {
            if (textIndex[idx] in ignoredTokens) {
                continue;
            }
            if (isRemoveDuplicate) {
                if (idx > 0 && textIndex[idx - 1] === textIndex[idx]) {
                    continue;
                }
            }
            charList.push(character[textIndex[idx] - 1]);
            if (textProb) {
                confList.push(textProb[idx]);
            } else {
                confList.push(1);
            }
        }
        let text = "";
        let mean = 0;
        if (charList.length) {
            text = charList.join("");
            let sum = 0;
            confList.forEach((item) => {
                sum += item;
            });
            mean = sum / confList.length;
        }
        return { text, mean };
    }
    return line;
}

// TODO 使用板式识别代替
/** 组成行 */
function afAfRec(l: resultType) {
    console.log(l);

    let line: resultType = [];
    let ind: Map<BoxType, number> = new Map();
    for (let i in l) {
        ind.set(l[i].box, Number(i));
    }

    function calculateAverageHeight(boxes: BoxType[]): number {
        let totalHeight = 0;
        for (const box of boxes) {
            const [[, y1], , [, y2]] = box;
            const height = y2 - y1;
            totalHeight += height;
        }
        return totalHeight / boxes.length;
    }

    function groupBoxesByMidlineDifference(boxes: BoxType[]): BoxType[][] {
        const averageHeight = calculateAverageHeight(boxes);
        const result: BoxType[][] = [];
        for (const box of boxes) {
            const [[, y1], , [, y2]] = box;
            const midline = (y1 + y2) / 2;
            const group = result.find((b) => {
                const [[, groupY1], , [, groupY2]] = b[0];
                const groupMidline = (groupY1 + groupY2) / 2;
                return Math.abs(groupMidline - midline) < averageHeight / 2;
            });
            if (group) {
                group.push(box);
            } else {
                result.push([box]);
            }
        }

        for (const group of result) {
            group.sort((a, b) => {
                const [ltA] = a;
                const [ltB] = b;
                return ltA[0] - ltB[0];
            });
        }

        result.sort((a, b) => a[0][0][1] - b[0][0][1]);

        return result;
    }

    let boxes = groupBoxesByMidlineDifference([...ind.keys()]);

    for (let i of boxes) {
        let t = [];
        let m = 0;
        for (let j of i) {
            let x = l[ind.get(j)];
            t.push(x.text);
            m += x.mean;
        }
        line.push({ mean: m / i.length, text: t.join(" "), box: [i.at(0)[0], i.at(-1)[1], i.at(-1)[2], i.at(0)[3]] });
    }
    return line;
}
