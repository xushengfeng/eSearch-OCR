// @ts-check
import { init, det, rec, ocr, loadImg, initDet, initRec, analyzeLayout, warpDet } from "../";
// @ts-ignore
import ort from "onnxruntime-node";

const modelDir = "./"; // 根据readme下载解压
// 如果报错，要修改为正确的路径
const detPath = `${modelDir}/ppocr_v5_mobile_det.onnx`; // det指识别模型，如果上面提到的文字包没有，那就用中英混合的det（在ch.zip里）
const recPath = `${modelDir}/ppocr_v5_mobile_rec.onnx`;
const recDic = await (await fetch(`${modelDir}/ppocrv5_dict.txt`)).text(); // 在模型压缩包中的txt文件，需要传入里面的内容而不是路径

// 在浏览器中，可以传入 链接 | HTMLImageElement | HTMLCanvasElement | ImageData，在node中需要canvas库加载 ImageData
const imgInput = "";
const oneLineImgInput = ""; // 单行文字的图片

// 配置
// 可以配置多个OCR且不干扰
const lOcr = await init({
    ort: ort,
    det: { input: detPath },
    rec: {
        input: recPath,
        decodeDic: recDic,
        optimize: {
            space: false,
        },
    },
});

async function base() {
    const r = await lOcr.ocr(imgInput);
    console.log(r.parragraphs.map((i) => i.text).join("\n")); // 输出排版识别后的文字
}

async function split() {
    // 流程是 加载图片 det（检测每行文本在图片的位置并取出） rec（识别以行为单位的文字） analyzeLayout（排版分析）
    // 这些功能都可以单独使用
    // 所以上面的操作可以拆分开来
    // 这些操作可以单独运行，或者自定义替换优化
    const d = await lOcr.det(await loadImg(imgInput));
    const rr = await lOcr.rec(d);
    const r = analyzeLayout(rr);
    console.log(r.parragraphs.map((i) => i.text).join("\n"));
}

// 单独配置，与init无关

async function onlyDet() {
    // 单独运行det，检测文字区域
    // 比如要开发文字隐私保护自动打码程序，我不需要识别具体的文字，只需要文本坐标
    const di = await initDet({
        input: detPath,
        ort: ort,
    });
    const d = await di.det(await loadImg(imgInput));
    console.log(d);
}
async function onlyRec() {
    // 单独运行rec，识别文本，可以说是OCR的核心
    // 比如要开发手写输入，通过轨迹点集可以计算文本区域范围，就不需要det了
    const ri = await initRec({
        ort: ort,
        decodeDic: recDic,
        input: recPath,
    });
    // 为了匹配输入格式，使用warpDet
    // 确保输入的图片是单行文字的
    const r = await ri.rec(warpDet(await loadImg(oneLineImgInput)));
    console.log(r);
}

async function recMuti() {
    // rec识别每个字的时候，实际上是把字典里的字进行排序
    // 所以可以输出每个字的候选
    const ri = await initRec({
        ort: ort,
        decodeDic: recDic,
        input: recPath,
    });
    // rawRec给出了候选和置信度，可以自己结合词典等方式优化结果或者用户直接选择候选
    // 如果要组合，生成句子候选，建议使用beam search算法
    // 因为计算所有候选性能太低，我们其实只用专注部分候选，你可以在initRec或者rawRec自定义topK和threshold限制范围
    const r = await ri.rawRec(warpDet(await loadImg(oneLineImgInput)));
    console.log(r);
}

// 你可以更改其他示例函数
await base();
