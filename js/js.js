const ort = require("onnxruntime-node");
const jimp = require("jimp");
var cv = require("opencv.js");
var src0;
async function x() {
    const session = await ort.InferenceSession.create("./m/ch_PP-OCRv2_det_infer.onnx");

    let image = await jimp.read("../a.png");
    let h = image.bitmap.height,
        w = image.bitmap.width;
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
    let img = document.createElement("img");
    img.src = await image.getBase64Async("image/png");
    document.querySelectorAll("canvas")[0].width = resize_w;
    document.querySelectorAll("canvas")[0].height = resize_h;
    function loadimg() {
        return new Promise(function (resolve, reject) {
            img.onload = resolve;
        });
    }
    await loadimg();
    document.querySelectorAll("canvas")[0].getContext("2d").drawImage(img, 0, 0);

    src0 = cv.imread(document.querySelectorAll("canvas")[0]);

    var imageBufferData = image.bitmap.data;
    const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());

    let scale = 1.0 / 255;
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let shapes = [h, w, resize_h / h, resize_w / w];

    let x = 0,
        y = 0;
    for (let i = 0; i < imageBufferData.length; i += 4) {
        if (!blueArray[y]) blueArray[y] = [];
        if (!greenArray[y]) greenArray[y] = [];
        if (!redArray[y]) redArray[y] = [];
        redArray[y][x] = (imageBufferData[i] * scale - mean[0]) / std[0];
        greenArray[y][x] = (imageBufferData[i + 1] * scale - mean[1]) / std[1];
        blueArray[y][x] = (imageBufferData[i + 2] * scale - mean[2]) / std[2];
        x++;
        if (x == w) {
            x = 0;
            y++;
        }
    }

    const transposedData = [blueArray, greenArray, redArray];
    const float32Data = Float32Array.from(transposedData.flat(Infinity));

    const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, image.bitmap.height, image.bitmap.width]);

    const results = await session.run({ x: inputTensor });

    const dataC = results["save_infer_model/scale_0.tmp_1"].data;
    console.log(results);

    let canvas = document.querySelectorAll("canvas")[1];

    var myImageData = new ImageData(resize_w, resize_h);
    for (let i in dataC) {
        let n = i * 4;
        myImageData.data[n] = myImageData.data[n + 1] = myImageData.data[n + 2] = dataC[i] * 255;
        myImageData.data[n + 3] = 255;
    }
    canvas.width = results["save_infer_model/scale_0.tmp_1"].dims[3];
    canvas.height = results["save_infer_model/scale_0.tmp_1"].dims[2];
    canvas.getContext("2d").putImageData(myImageData, 0, 0);

    let box = find_contors();

    const rec = await ort.InferenceSession.create("./m/ch_PP-OCRv2_rec_infer.onnx");

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
        let imageBufferData = resize_norm_img(r.img).data;
        const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array());

        let x = 0,
            y = 0;
        for (let i = 0; i < imageBufferData.length; i += 4) {
            if (!blueArray[y]) blueArray[y] = [];
            if (!greenArray[y]) greenArray[y] = [];
            if (!redArray[y]) redArray[y] = [];
            redArray[y][x] = (imageBufferData[i] / 255 - 0.5) / 0.5;
            greenArray[y][x] = (imageBufferData[i + 1] / 255 - 0.5) / 0.5;
            blueArray[y][x] = (imageBufferData[i + 2] / 255 - 0.5) / 0.5;
            x++;
            if (x == imgW) {
                x = 0;
                y++;
            }
        }
        b.push([blueArray, greenArray, redArray]);
    }
    console.log(b);
    const float32Data1 = Float32Array.from(b.flat(Infinity));

    const inputTensor1 = new ort.Tensor("float32", float32Data1, [b.length, 3, imgH, imgW]);

    const results1 = await rec.run({ x: inputTensor1 });
    let data = results1["save_infer_model/scale_0.tmp_1"];
    console.log();

    const fs = require("fs");
    let character = fs.readFileSync("../assets/ppocr_keys_v1.txt").toString().split("\n");
    console.log(character);
    const pred_len = data.dims[2];

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
    console.log(line);
}
x();

function find_contors() {
    let canvas = document.querySelectorAll("canvas")[1];

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
