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
    return new Promise((rs) => rs(true));
}

/** 主要操作 */
async function x(img: ImageData) {
    console.time();
    let h = img.height,
        w = img.width;
    let { transposedData, resizeW: resizeW, image, canvas } = 检测前处理(h, w, img);
    const detResults = await 检测(transposedData, image, det);

    let box = 检测后处理(detResults.data, detResults.dims[3], detResults.dims[2], canvas);

    let mainLine: { text: string; mean: number; box?: number[][] }[] = [];
    for (let i of 识别前处理(resizeW, box)) {
        let { b, imgH, imgW } = i;
        const recResults = await 识别(b, imgH, imgW, rec);
        if (dic.at(-1) == "") {
            // 多出的换行
            dic[dic.length - 1] = " ";
        } else {
            dic.push(" ");
        }
        let line = 识别后处理(recResults, dic);
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
    console.log(mainLine);
    console.timeEnd();
    return mainLine;
}

async function 检测(transposedData: number[][][], image: ImageData, det: SessionType) {
    let x = transposedData.flat(Infinity) as number[];
    const detData = Float32Array.from(x);

    const detTensor = new ort.Tensor("float32", detData, [1, 3, image.height, image.width]);
    let detFeed = {};
    detFeed[det.inputNames[0]] = detTensor;

    const detResults = await det.run(detFeed);
    return detResults[det.outputNames[0]];
}

async function 识别(b: number[][][], imgH: number, imgW: number, rec: SessionType) {
    const recData = Float32Array.from(b.flat(Infinity) as number[]);

    const recTensor = new ort.Tensor("float32", recData, [b.length, 3, imgH, imgW]);
    let recFeed = {};
    recFeed[rec.inputNames[0]] = recTensor;

    const recResults = await rec.run(recFeed);
    return recResults[rec.outputNames[0]];
}

/**
 *
 * @param {ImageData} data 原图
 * @param {number} w 输出宽
 * @param {number} h 输出高
 */
function resizeImg(data: ImageData, w: number, h: number) {
    let x = document.createElement("canvas");
    x.width = data.width;
    x.height = data.height;
    x.getContext("2d").putImageData(data, 0, 0);
    let src = document.createElement("canvas");
    src.width = w;
    src.height = h;
    src.getContext("2d").scale(w / data.width, h / data.height);
    src.getContext("2d").drawImage(x, 0, 0);
    return src.getContext("2d").getImageData(0, 0, w, h);
}

function 检测前处理(h: number, w: number, image: ImageData) {
    let ratio = 1;
    if (Math.max(h, w) > limitSideLen) {
        if (h > w) {
            ratio = limitSideLen / h;
        } else {
            ratio = limitSideLen / w;
        }
    }
    let resizeH = h * ratio;
    let resizeW = w * ratio;

    resizeH = Math.max(Math.round(resizeH / 32) * 32, 32);
    resizeW = Math.max(Math.round(resizeW / 32) * 32, 32);
    image = resizeImg(image, resizeW, resizeH);
    let srcCanvas = document.createElement("canvas");
    srcCanvas.width = resizeW;
    srcCanvas.height = resizeH;
    let id = new ImageData(image.width, image.height);
    for (let i in id.data) id.data[i] = image.data[i];
    srcCanvas.getContext("2d").putImageData(id, 0, 0);

    // src0 = cv.imread(src_canvas);

    const transposedData = toPaddleInput(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    console.log(image);
    if (dev) {
        document.body.append(srcCanvas);
    }
    return { transposedData, resizeW: resizeW, image, canvas: srcCanvas };
}

function 检测后处理(
    data: AsyncType<ReturnType<typeof 检测>>["data"],
    w: number,
    h: number,
    srcCanvas: HTMLCanvasElement
) {
    let canvas = document.createElement("canvas");

    var myImageData = new ImageData(w, h);
    for (let i in data) {
        let n = Number(i) * 4;
        const v = (data[i] as number) > 0.3 ? 255 : 0;
        myImageData.data[n] = myImageData.data[n + 1] = myImageData.data[n + 2] = v;
        myImageData.data[n + 3] = 255;
    }
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").putImageData(myImageData, 0, 0);

    let edgeRect: { box: number[][]; img: ImageData }[] = [];

    let src = cv.imread(canvas);

    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
        let cnt = contours.get(i);
        let bbox = cv.boundingRect(cnt) as { x: number; y: number; width: number; height: number };
        // TODO minAreaRect

        let dx = 8,
            dy = 8;

        let box = [
            [bbox.x - dx, bbox.y - dy],
            [bbox.x + bbox.width + dx * 2, bbox.y - dy],
            [bbox.x + bbox.width + dx * 2, bbox.y + bbox.height + dy * 2],
            [bbox.x - dx, bbox.y + bbox.height + dy * 2],
        ];

        let minSize = 3;
        if (Math.min(bbox.width, bbox.height) >= minSize) {
            let c = document.createElement("canvas");
            c.width = bbox.width + dx * 2;
            c.height = bbox.height + dy * 2;

            let ctx = c.getContext("2d");
            let c0 = srcCanvas;
            ctx.drawImage(c0, -bbox.x + dx, -bbox.y + dy);
            if (dev) document.body.append(c);

            edgeRect.push({ box, img: c.getContext("2d").getImageData(0, 0, c.width, c.height) });
        }
    }

    console.log(edgeRect);

    src.delete();
    contours.delete();
    hierarchy.delete();

    src = contours = hierarchy = null;

    return edgeRect;
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

function 识别前处理(resize_w: any, box: { box: number[][]; img: ImageData }[]) {
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
        let cc = document.createElement("canvas");
        cc.width = imgW;
        cc.height = imgH;
        cc.getContext("2d").putImageData(d, 0, 0);
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

function 识别后处理(data: AsyncType<ReturnType<typeof 识别>>, character: string[]) {
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
