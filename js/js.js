const ort = require("onnxruntime-node");
const jimp = require("jimp");
var cv = require("opencv.js");
const fs = require("fs");
var src0;

start();

async function start() {
    const det = await ort.InferenceSession.create("./m/ch_PP-OCRv2_det_infer.onnx");
    const rec = await ort.InferenceSession.create("./m/ch_PP-OCRv2_rec_infer.onnx");
    let dic = fs.readFileSync("../assets/ppocr_keys_v1.txt").toString().split("\n");
    let img = document.createElement("img");
    img.src = "../a.png";
    img.onload = () => {
        let canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        x(canvas.getContext("2d").getImageData(0, 0, img.width, img.height), det, rec, dic);
    };
}

/**
 *
 * @param {ImageData} img 图片
 * @param {ort.InferenceSession} det 检测器
 * @param {ort.InferenceSession} rec 识别器
 * @param {Array} dic 字典
 */
async function x(img, det, rec, dic) {
    let image = await jimp.read("../a.png");
    let h = image.bitmap.height,
        w = image.bitmap.width;
    let transposedData;
    let resize_w;
    ({ transposedData, resize_w, image } = 检测前处理(h, w, image));
    const det_data = Float32Array.from(transposedData.flat(Infinity));

    const det_tensor = new ort.Tensor("float32", det_data, [1, 3, image.bitmap.height, image.bitmap.width]);
    let det_feed = {};
    det_feed[det.inputNames[0]] = det_tensor;

    const det_results = await det.run(det_feed);

    let box = 检测后处理(
        det_results[det.outputNames[0]].data,
        det_results[det.outputNames[0]].dims[3],
        det_results[det.outputNames[0]].dims[2]
    );

    let { b, imgH, imgW } = 识别前处理(resize_w, box);
    const rec_data = Float32Array.from(b.flat(Infinity));

    const rec_tensor = new ort.Tensor("float32", rec_data, [b.length, 3, imgH, imgW]);
    let rec_feed = {};
    rec_feed[rec.inputNames[0]] = rec_tensor;

    const rec_results = await rec.run(rec_feed);
    let data = rec_results[rec.outputNames[0]];

    let character = dic;
    const pred_len = data.dims[2];

    let line = 识别后处理(data, pred_len, character);
    console.log(line);
}

function 检测前处理(h, w, image) {
    let limit_side_len = 960;
    let ratio = 1;
    if (Math.max(h, w) > limit_side_len) {
        if (h > w) {
            ratio = limit_side_len / h;
        } else {
            ratio = limit_side_len / w;
        }
    }
    let resize_h = h * ratio;
    let resize_w = w * ratio;

    resize_h = Math.max(Math.round(resize_h / 32) * 32, 32);
    resize_w = Math.max(Math.round(resize_w / 32) * 32, 32);
    image = image.resize(resize_w, resize_h);
    document.querySelectorAll("canvas")[0].width = resize_w;
    document.querySelectorAll("canvas")[0].height = resize_h;
    let id = new ImageData(image.bitmap.width, image.bitmap.height);
    for (let i in id.data) id.data[i] = image.bitmap.data[i];
    document.querySelectorAll("canvas")[0].getContext("2d").putImageData(id, 0, 0);

    src0 = cv.imread(document.querySelectorAll("canvas")[0]);

    const transposedData = to_paddle_input(image.bitmap, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    return { transposedData, resize_w, image };
}

function 检测后处理(data, w, h) {
    let canvas = document.createElement("canvas");

    var myImageData = new ImageData(w, h);
    for (let i in data) {
        let n = i * 4;
        myImageData.data[n] = myImageData.data[n + 1] = myImageData.data[n + 2] = data[i] * 255;
        myImageData.data[n + 3] = 255;
    }
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").putImageData(myImageData, 0, 0);

    let edge_rect = [];

    let src = cv.imread(canvas);

    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(src, src, 120, 200, cv.THRESH_BINARY);
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
        let cnt = contours.get(i);
        let bbox = cv.boundingRect(cnt);
        // TODO minAreaRect

        let box = [
            [bbox.x, bbox.y],
            [bbox.x + bbox.width, bbox.y],
            [bbox.x + bbox.width, bbox.y + bbox.height],
            [bbox.x, bbox.y + bbox.height],
        ];

        let min_size = 3;
        if (Math.min(bbox.width, bbox.height) >= min_size) {
            let c = document.createElement("canvas");
            let dx = bbox.width * 0.2,
                dy = bbox.height;
            c.width = bbox.width + dx * 2;
            c.height = bbox.height + dy * 2;

            let ctx = c.getContext("2d");
            let c0 = document.querySelectorAll("canvas")[0];
            ctx.drawImage(c0, -bbox.x + dx, -bbox.y + dy);

            document.body.append(c);

            edge_rect.push({ box, img: c.getContext("2d").getImageData(0, 0, c.width, c.height) });
        }
    }

    console.log(edge_rect);

    src.delete();
    contours.delete();
    hierarchy.delete();

    src = dst = contours = hierarchy = null;

    return edge_rect;
}

function to_paddle_input(image, mean, std) {
    const imagedata = image.data;
    const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());
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

function 识别前处理(resize_w, box) {
    let imgC = 3,
        imgH = 32,
        imgW = 320;
    /**
     *
     * @param {ImageData} img
     */
    function resize_norm_img(img) {
        imgW = Math.floor(32 * max_wh_ratio);
        let h = img.height,
            w = img.width;
        let ratio = w / h;
        let resized_w;
        if (Math.ceil(imgH * ratio) > imgW) {
            resized_w = imgW;
        } else {
            resized_w = Math.floor(Math.ceil(imgH * ratio));
        }
        let c = document.createElement("canvas");
        c.width = resized_w;
        c.height = imgH;
        c.getContext("2d").scale(resize_w / img.width, resize_w / img.width);
        c.getContext("2d").putImageData(img, 0, 0);
        var imageBufferData = c.getContext("2d").getImageData(0, 0, c.width, c.height);
        for (let i = 0; i < imageBufferData.data.length; i++) {
            imageBufferData.data[i] = imageBufferData.data[i];
        }
        let cc = document.createElement("canvas");
        cc.width = imgW;
        cc.height = imgH;
        cc.getContext("2d").putImageData(imageBufferData, 0, 0);
        return cc.getContext("2d").getImageData(0, 0, imgW, imgH);
    }

    let max_wh_ratio = 0;
    for (let r of box) {
        max_wh_ratio = Math.max(r.img.width / r.img.height, max_wh_ratio);
    }
    let b = [];
    for (let r of box) {
        b.push(to_paddle_input(resize_norm_img(r.img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]));
    }
    console.log(b);
    return { b, imgH, imgW };
}

function 识别后处理(data, pred_len, character) {
    let line = [];
    let ml = data.dims[0] - 1;
    for (let l = 0; l < data.data.length; l += pred_len * data.dims[1]) {
        const preds_idx = [];
        const preds_prob = [];

        for (let i = l; i < l + pred_len * data.dims[1]; i += pred_len) {
            const tmpArr = data.data.slice(i, i + pred_len - 1);
            const tmpMax = Math.max(...tmpArr);
            const tmpIdx = tmpArr.indexOf(tmpMax);
            preds_prob.push(tmpMax);
            preds_idx.push(tmpIdx);
        }
        line[ml] = decode(preds_idx, preds_prob, true);
        ml--;
    }
    function decode(text_index, text_prob, is_remove_duplicate) {
        const ignored_tokens = [0];
        const char_list = [];
        const conf_list = [];
        for (let idx = 0; idx < text_index.length; idx++) {
            if (text_index[idx] in ignored_tokens) {
                continue;
            }
            if (is_remove_duplicate) {
                if (idx > 0 && text_index[idx - 1] === text_index[idx]) {
                    continue;
                }
            }
            char_list.push(character[text_index[idx] - 1]);
            if (text_prob) {
                conf_list.push(text_prob[idx]);
            } else {
                conf_list.push(1);
            }
        }
        let text = "";
        let mean = 0;
        if (char_list.length) {
            text = char_list.join("");
            let sum = 0;
            conf_list.forEach((item) => {
                sum += item;
            });
            mean = sum / conf_list.length;
        }
        return { text, mean };
    }
    return line;
}