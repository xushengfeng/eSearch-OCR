import { int, resizeImg, data2canvas, toPaddleInput } from "./untils";
import type { SessionType, AsyncType } from "./untils";
type ortType = typeof import("onnxruntime-common");

async function layout(img: ImageData, ort: ortType, session: SessionType, dic: string[]) {
    const h = img.height;
    const w = img.width;
    const x = beforeS(img, 800, 608);
    const rResults = await runS(x.transposedData, x.image, ort, session);
    console.log(rResults);
    const sr = afterS(rResults, w, h, dic);
    return sr;
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

function beforeS(image: ImageData, shapeH: number, shapeW: number) {
    const ratio = 1;
    const h = image.height;
    const w = image.width;
    let resizeH = shapeH;
    let resizeW = shapeW;

    resizeH = Math.max(Math.round(resizeH / 32) * 32, 32);
    resizeW = Math.max(Math.round(resizeW / 32) * 32, 32);

    // biome-ignore lint: 规范化
    image = resizeImg(image, resizeW, resizeH);

    const transposedData = toPaddleInput(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    return { transposedData, image };
}

function afterS(data: AsyncType<ReturnType<typeof runS>>, w: number, h: number, layoutDic: string[]) {
    const strides = [8, 16, 32, 64];
    const score_threshold = 0.4;
    const nms_threshold = 0.5;
    const nms_top_k = 1000;
    const keep_top_k = 100;
    const scores: AsyncType<ReturnType<typeof runS>> = [];
    const boxes: AsyncType<ReturnType<typeof runS>> = [];
    const n = data.length / 2;
    for (let i = 0; i < n; i++) {
        scores.push(data[i]);
        boxes.push(data[i + n]);
    }
    const reg_max = int(boxes[0].dims.at(-1) / 4 - 1);
    let out_boxes_num: number[] = [];
    const out_boxes_list: number[][][] = [];
    const results: { bbox: number[]; label: string }[] = [];
    const ori_shape = [800, 608];
    const scale_factor = [800 / h, 608 / w];

    const decode_boxes = [];
    const select_scores = [];

    for (let i = 0; i < strides.length; i++) {
        const stride = strides[i];
        const box_distribute = boxes[i];
        const score = scores[i];

        // center
        const fm_h = 800 / stride;
        const fm_w = 608 / stride;
        const h_range: number[] = Array.from({ length: Math.ceil(fm_h) }, (_, index) => index);
        const w_range: number[] = Array.from({ length: Math.ceil(fm_w) }, (_, index) => index);
        const [ww, hh] = meshgrid(w_range, h_range);

        const ct_row: number[] = hh.flat().map((val) => (val + 0.5) * stride);
        const ct_col: number[] = ww.flat().map((val) => (val + 0.5) * stride);

        const center: number[][] = [];
        for (const i in ct_row) {
            center.push([ct_col[i], ct_row[i], ct_col[i], ct_row[i]]);
        }

        // box distribution to distance
        const reg_max1 = reg_max + 1;
        const shape0 = box_distribute.size / reg_max1;

        const box_distance_softmax: number[][] = []; // shape:shape0, reg_max1
        for (let i = 0; i < shape0; i++) {
            const reshape0 = box_distribute.data.slice(i * reg_max1, (i + 1) * reg_max1); // reshape，这是size/regmax1
            const box_distance_axis_softmax = softmax(Array.from(reshape0 as Float32Array)); // softmax
            for (let j = 0; j < reg_max1; j++) {
                box_distance_axis_softmax[j] *= j;
            }
            box_distance_softmax.push(box_distance_axis_softmax); // soft * expand_dims reg_range
        }

        const box_distance_1_sum: number[] = []; // shape 记得转为 shape0/4,4
        const box_distance_shape0 = shape0 / 4;
        for (let i = 0; i < shape0; i++) {
            let x = 0;
            for (let n = 0; n < reg_max1; n++) {
                x += box_distance_softmax[i][n]; // sum
            }
            x *= stride;
            box_distance_1_sum.push(x);
        }

        // top K candidate
        const topKMap: [number, number][] = []; // [index,v]
        const scoreShape0 = score.dims[1];
        const scoreShape1 = score.dims[2];
        for (let i = 0; i < scoreShape0; i++) {
            const slice = score.data.slice(i * scoreShape1, (i + 1) * scoreShape1) as Float32Array;
            topKMap.push([i, Math.max(...(Array.from(slice) as number[]))]);
        }
        topKMap.sort((a, b) => b[1] - a[1]);

        const clipCenter: typeof center = [];
        const clipScore: number[][] = [];
        const clipBoxDistance: number[][] = [];
        for (let i = 0; i < Math.min(center.length, nms_top_k); i++) {
            clipCenter[i] = center[topKMap[i][0]];
        }

        for (let i = 0; i < Math.min(scoreShape0, nms_top_k); i++) {
            const slice = score.data.slice(
                topKMap[i][0] * scoreShape1,
                (topKMap[i][0] + 1) * scoreShape1,
            ) as Float32Array;
            clipScore[i] = Array.from(slice);
        }
        for (let i = 0; i < Math.min(box_distance_shape0, nms_top_k); i++) {
            const slice = box_distance_1_sum.slice(topKMap[i][0] * 4, (topKMap[i][0] + 1) * 4);
            clipBoxDistance[i] = Array.from(slice);
        }

        // decode box
        const decode_box: number[][] = [];
        const boxM = [-1, -1, 1, 1];
        for (const i in clipCenter) {
            const tL: number[] = [];
            for (let n = 0; n < 4; n++) {
                tL.push(clipCenter[i][n] + boxM[n] * clipBoxDistance[i][n]);
            }
            decode_box.push(tL);
        }

        select_scores.push(clipScore);
        decode_boxes.push(decode_box);
    }

    // nms
    // 将解码后的边界框数组和置信度数组拼接起来
    const bboxes: number[][] = decode_boxes.flat(); // ,4
    const confidences: number[][] = select_scores.flat(); // ,10
    const picked_box_probs: number[][][] = []; // 存储经过NMS后的边界框及对应的置信度
    const picked_labels: number[] = []; // 存储被选中的边界框对应的类别标签

    // 遍历每个类别的置信度
    for (let classIndex = 0; classIndex < confidences[0].length; classIndex++) {
        const probs: number[] = confidences.map((conf) => conf[classIndex]);
        const mask: boolean[] = probs.map((prob) => prob > score_threshold); // 根据置信度阈值筛选出符合条件的边界框
        const filteredProbs: number[] = probs.filter((_, i) => mask[i]); // 获取符合条件的置信度
        if (filteredProbs.length === 0) {
            continue;
        }

        const subsetBoxes: number[][] = bboxes.filter((_, i) => mask[i]); // 获取符合条件的边界框

        const boxProbs: number[][] = subsetBoxes.map((box, i) => {
            return [...box, filteredProbs[i]];
        }); // 将边界框和置信度合并成一个数组

        const nmsBoxProbs: number[][] = hard_nms(
            boxProbs,
            nms_threshold, // NMS的IoU阈值
            keep_top_k, // 每个类别保留的最大目标框数
        );

        picked_box_probs.push(nmsBoxProbs); // 将经过NMS筛选后的边界框添加到列表中
        picked_labels.push(...Array(nmsBoxProbs.length).fill(classIndex)); // 将类别标签添加到列表中
    }

    if (picked_box_probs.length === 0) {
        out_boxes_list.push([]); // 如果没有选中的边界框，添加空数组
        out_boxes_num.push(0);
    } else {
        const pickedBoxProbs: number[][] = [];

        const oriShape = ori_shape;
        const scaleFactors = [scale_factor[1], scale_factor[0], scale_factor[1], scale_factor[0]];
        // 调整输出的边界框大小
        const pickedBoxProbsBox: number[][] = [];
        const pickedBoxProbsScore: number[] = [];

        picked_box_probs.flat().forEach((boxProb, i) => {
            pickedBoxProbsBox.push(boxProb.slice(0, 4));
            pickedBoxProbsScore.push(boxProb[4]);
        });
        const w = warp_boxes(pickedBoxProbsBox, oriShape);
        pickedBoxProbsBox.forEach((boxProb, i) => {
            w[i].forEach((val, j) => {
                console.log(val, scaleFactors[Math.floor(j / 2)]);

                w[i][j] = val / scaleFactors[j];
            });
            pickedBoxProbs.push([...w[i], pickedBoxProbsScore[i]]);
        });

        // 将类别标签、置信度和边界框组合成新的数组并添加到输出列表

        out_boxes_list.push(pickedBoxProbs.map((boxProb, i) => [picked_labels[i], boxProb[4], ...boxProb.slice(0, 4)]));
        out_boxes_num.push(picked_labels.length); // 记录选中的边界框数量
    }

    const out_boxes_list1 = out_boxes_list.flat();
    out_boxes_num = out_boxes_num.map((num) => num);

    const labels = layoutDic;

    out_boxes_list1.forEach((dt, i) => {
        const clsid: number = int(dt[0]);
        const bbox: number[] = dt.slice(2);
        const score: number = dt[1];
        const label: string = labels[clsid];
        const result = { bbox: bbox, label: label };
        results.push(result);
    });

    console.log(results);

    return results;
}

function clip(v: number, min: number, max: number) {
    return Math.max(Math.min(v, max), min);
}

function warp_boxes(boxes: number[][], oriShape: number[]): number[][] {
    const width = oriShape[1];
    const height = oriShape[0];
    const n = boxes.length;

    if (n > 0) {
        // warp points
        const xmin: number[] = [];
        const ymin: number[] = [];
        const xmax: number[] = [];
        const ymax: number[] = [];
        for (let i = 0; i < n; i++) {
            const [x1, y1, x2, y2] = boxes[i];
            xmin.push(Math.min(x1, x2));
            ymin.push(Math.min(y1, y2));
            xmax.push(Math.max(x1, x2));
            ymax.push(Math.max(y1, y2));
        }

        const xy1: number[][] = [];
        for (const i in xmin) {
            xy1.push([
                clip(xmin[i], 0, width),
                clip(ymin[i], 0, height),
                clip(xmax[i], 0, width),
                clip(ymax[i], 0, height),
            ]);
        }

        return xy1;
    }
    return boxes;
}

function hard_nms(box_scores: number[][], iou_threshold: number, top_k = -1, candidate_size = 200): number[][] {
    const scores: number[] = box_scores.map((box) => box[4]);
    const boxes: number[][] = box_scores.map((box) => box.slice(0, 4));
    const picked: number[] = [];
    let indexes: number[] = scores.map((_, i) => i);
    indexes = indexes
        .sort((a, b) => scores[b] - scores[a])
        .slice(0, candidate_size)
        .reverse();
    while (indexes.length > 0) {
        const current: number = indexes.at(-1);
        picked.push(current);
        if ((top_k > 0 && picked.length === top_k) || indexes.length === 1) {
            break;
        }
        const current_box: number[] = boxes[current];
        indexes = indexes.slice(0, indexes.length - 1);
        const rest_boxes: number[][] = indexes.map((i) => boxes[i]);
        const iou: number[] = iou_of(rest_boxes, [current_box]);
        indexes = indexes.filter((_, i) => iou[i] <= iou_threshold);
    }

    return picked.map((index) => box_scores[index]);
}

function iou_of(boxes0: number[][], boxes1: number[][], eps = 1e-5): number[] {
    const overlap_left_top: number[][] = boxes0.map((box0, i) => [
        Math.max(box0[0], boxes1[i]?.[0] ?? Number.NEGATIVE_INFINITY),
        Math.max(box0[1], boxes1[i]?.[1] ?? Number.NEGATIVE_INFINITY),
    ]);
    const overlap_right_bottom: number[][] = boxes0.map((box0, i) => [
        Math.min(box0[2], boxes1[i]?.[2] ?? Number.NEGATIVE_INFINITY),
        Math.min(box0[3], boxes1[i]?.[3] ?? Number.NEGATIVE_INFINITY),
    ]);
    const overlap_area: number[] = area_of(overlap_left_top, overlap_right_bottom);
    const area0: number[] = boxes0.map((box) => (box[2] - box[0]) * (box[3] - box[1]));
    const area1: number[] = boxes1.map((box) => (box[2] - box[0]) * (box[3] - box[1]));
    return overlap_area.map((area, i) => area / (area0[i] + area1[i] - area + eps));
}

function area_of(left_top: number[][], right_bottom: number[][]): number[] {
    const hw: number[][] = left_top.map((lt, i) => [
        Math.max(right_bottom[i][0] - lt[0], 0),
        Math.max(right_bottom[i][1] - lt[1], 0),
    ]);
    return hw.map((dims) => dims[0] * dims[1]);
}

function softmax(input: number[]): number[] {
    const maxVal = Math.max(...input);
    const exps = input.map((x) => Math.exp(x - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const result = exps.map((x) => x / sumExps);
    return result;
}

function meshgrid(...args: number[][]): number[][][] {
    const results: number[][][] = [[], [], []];

    for (const y in args[1]) {
        results[0].push(args[0]);
    }

    for (const y in args[1]) {
        const row = [];
        for (const x in args[0]) {
            row.push(args[1][y]);
        }
        results[1].push(row);
    }

    return results;
}

export { layout as runLayout };
