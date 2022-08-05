const ort = require("onnxruntime-node");
const jimp = require("jimp");
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

    const dataC = results.data;
    console.log(dataC);
}
x();
