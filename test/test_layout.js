const { initDet, analyzeLayout, setOCREnv } = require("../");
const fs = require("node:fs");
const path = require("node:path");
const ort = require("onnxruntime-node");

start();

async function start() {
    const modelBasePath = "./m/v4/";
    setOCREnv({
        log: true,
        // dev: true,
    });
    const det = await initDet({
        input: `${modelBasePath}/ppocr_det.onnx`,
        ort,
    });
    document.body.style.display = "flex";
    document.body.style.flexWrap = "wrap";
    const rootPath = "../test/layout_img";
    const dir = fs.readdirSync(rootPath);
    const only = 0 && "7.svg";
    for (const f of dir) {
        if (f !== only && only !== 0) continue;
        const src = path.join(rootPath, f);
        const img = new Image();
        img.src = src;
        img.onload = async () => {
            const c = document.createElement("canvas");
            const ratio = 0.8;
            c.width = Math.floor(img.naturalWidth * ratio);
            c.height = Math.floor(img.naturalHeight * ratio);
            const p = document.createElement("div");
            const h = document.createElement("span");
            h.innerText = f;
            p.append(h, c);
            document.body.append(p);
            const ctx = c.getContext("2d");
            ctx.save();
            ctx.scale(ratio, ratio);
            ctx.drawImage(img, 0, 0);
            ctx.restore();
            const detResult = await det.det(ctx.getImageData(0, 0, c.width, c.height));
            const mainLine = detResult.map((i, n) => ({ ...i, text: n.toString(), mean: 1 }));
            const l = analyzeLayout(mainLine);

            console.log("result", l);

            const color = [];
            for (let h = 10; h < 360; h += Math.floor(360 / l.columns.length)) {
                color.push(`hsl(${h}, 100%, 50%)`);
            }
            for (const [i, c] of color.entries()) {
                const v = 8;
                const x = i * v;
                ctx.fillStyle = c;
                ctx.fillRect(x, 0, v, 8);
            }

            for (const [i, c] of l.columns.entries()) {
                for (const p of c.parragraphs) {
                    drawBox(p.parse.box, p.parse.text, color[i]);
                }
            }

            ctx.strokeStyle = "black";
            ctx.beginPath();
            ctx.moveTo(0, 0);
            for (const [i, c] of l.columns.entries()) {
                ctx.lineTo((c.outerBox[0][0] + c.outerBox[2][0]) / 2, (c.outerBox[0][1] + c.outerBox[2][1]) / 2);
            }
            ctx.stroke();

            function drawBox(box, id = "", color = "red") {
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.moveTo(box[0][0], box[0][1]);
                ctx.lineTo(box[1][0], box[1][1]);
                ctx.lineTo(box[2][0], box[2][1]);
                ctx.lineTo(box[3][0], box[3][1]);
                ctx.lineTo(box[0][0], box[0][1]);
                ctx.stroke();
                ctx.strokeStyle = "black";
                ctx.strokeText(id, box[0][0], box[0][1]);
            }

            window.db = drawBox;

            window.drawPoint = (p) => {
                ctx.fillStyle = "blue";
                ctx.fillRect(p[0], p[1], 4, 4);
            };
        };
    }
}
