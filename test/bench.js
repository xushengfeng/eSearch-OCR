const x = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");
const diff = require("diff-match-patch");

const dmp = new diff();
start();

async function load(src) {
    const img = document.createElement("img");
    img.src = src;
    await (() => {
        return new Promise((re) => {
            img.onload = () => {
                re();
            };
        });
    })();
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d").drawImage(img, 0, 0);
    return canvas.getContext("2d").getImageData(0, 0, img.width, img.height);
}

async function start() {
    const ocr = await x.init({
        detPath: "./m/v4/ppocr_det.onnx",
        recPath: "./m/v4/ppocr_rec.onnx",
        dic: fs.readFileSync("../assets/ppocr_keys_v1.txt").toString(),
        layoutDic: "text\ntitle\nfigure\nfigure_caption\ntable\ntable_caption\nheader\nfooter\nreference\nequation",
        detShape: [640, 640],
        ort,
        onProgress: (t, a, n) => {
            if (t === "rec") {
                console.log(n / a);
            }
        },
    });

    const r = [];

    for (const i of ["ch", "en", "bg1", "bg2"]) {
        const zqls = [];
        const times = [];
        const srcText = fs.readFileSync(`imgs/${i}.txt`).toString();
        for (let c = 0; c < 9; c++) {
            const data = await load(`imgs/${i}.svg`);
            const text = [];
            const startTime = performance.now();
            for (const i of (await ocr.ocr(data)).parragraphs) {
                text.push(i.text);
            }
            const spendTime = performance.now() - startTime;

            console.table([text.join("\n"), srcText]);

            const diff = dmp.diff_main(text.join("\n"), srcText);

            if (c === 0) console.log(diff);

            let l = 0;
            for (const i of diff) {
                if (i[0] === 0) {
                    l += i[1].length;
                } else {
                    l -= i[1].length * 0.5;
                }
            }
            console.log(l / srcText.length, spendTime);
            zqls.push(l / srcText.length);
            times.push(spendTime);
        }
        const time = times.reduce((a, b) => (a + b) / 2);
        r.push({
            name: i,
            zql: zqls.reduce((a, b) => (a + b) / 2),
            spendTime: time,
            charsP: srcText.length / (time / 1000),
        });
    }

    console.log("bench end");

    const log = JSON.parse(fs.readFileSync("log.json").toString());

    /** @type {typeof r} */
    const lastTest = Object.values(log).at(-1).r;

    for (const i of lastTest) {
        const name = i.name;
        const thistest = r.find((x) => x.name === name) || { zql: 0, charsP: 0 };
        if (Math.abs(i.zql - thistest.zql) > 0.01) {
            console.log("准确率变化：", name, `${i.zql} -> ${thistest.zql}`);
        }
        if (Math.abs(i.charsP - thistest.charsP) / thistest.charsP > 0.1) {
            console.log("字符速度变化：", name, `${i.charsP} -> ${thistest.charsP}`);
        }
    }

    log[new Date().getTime()] = {
        onnx: "1.19.2",
        type: "node",
        provider: "cpu",
        r,
    };
    fs.writeFileSync("log.json", JSON.stringify(log, null, 2));
}
