// @ts-check
import { init, det, rec, ocr, loadImg, initDet, initRec } from "../";
// @ts-ignore
import ort from "onnxruntime-node";

const lOcr = await init({
    detPath: "",
    recPath: "",
    dic: "",
    ort: ort,
});
{
    // init后，ocr,det,rec全局生效，配置继承自init
    // 由于共享状态，不推荐使用
    await ocr("");

    const d = await det(await loadImg(""));
    const r = await rec(d);
}
{
    // 推荐使用显式的操作，可以有多个OCR且不干扰
    await lOcr.ocr("");

    const d = await lOcr.det(await loadImg(""));
    const r = await lOcr.rec(d);
}

// 单独配置，与init无关
// 在部分场景很有用，如只需要检测文字区域进行隐私保护
{
    const di = await initDet({
        input: "",
        ort: ort,
    });
    const ri = await initRec({
        ort: ort,
        decodeDic: "",
        input: "",
    });
    const d = await di.det(await loadImg(""));
    const r = await ri.rec(d);
}
