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
    await lOcr.ocr("");

    const d = await lOcr.det(await loadImg(""));
    const r = await lOcr.rec(d);
}
{
    await ocr("");

    const d = await det(await loadImg(""));
    const r = await rec(d);
}
{
    const di = await initDet({
        detPath: "",
        ort: ort,
    });
    const ri = await initRec({
        ort: ort,
        dic: "",
        recPath: "",
    });
    const d = await di.det(await loadImg(""));
    const r = await ri.rec(d);
}
