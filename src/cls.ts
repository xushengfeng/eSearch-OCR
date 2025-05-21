import type { OrtOption } from "./main";
import { resizeImg, type SessionType, toPaddleInput } from "./untils";
type ortType = OrtOption["ort"];

export { Cls };

async function Cls<DicI>(img: ImageData, ort: ortType, session: SessionType, dic: DicI[], w: number, h: number) {
    const { transposedData, image } = beforeCls(img, w, h);
    const r = await runS(transposedData, image, ort, session);
    const result = r[0].data as Float32Array;
    const max = result.reduce((a, b) => Math.max(a, b));
    const maxIndex = result.findIndex((item) => item === max);
    return dic[maxIndex];
}

function beforeCls(_image: ImageData, resizeW: number, resizeH: number) {
    const image = resizeImg(_image, resizeW, resizeH);

    const transposedData = toPaddleInput(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    return { transposedData, image };
}

async function runS(transposedData: number[][][], image: ImageData, ort: ortType, layout: SessionType) {
    const x = transposedData.flat(Number.POSITIVE_INFINITY) as number[];
    const detData = Float32Array.from(x);

    const detTensor = new ort.Tensor("float32", detData, [1, 3, image.height, image.width]);
    const detFeed = {};
    detFeed[layout.inputNames[0]] = detTensor;

    const detResults = await layout.run(detFeed);
    const v = Object.values(detResults);
    return v;
}
