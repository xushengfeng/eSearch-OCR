const x = require("../");
const fs = require("node:fs");
const ort = require("onnxruntime-node");

start();

async function start() {
    const modelBasePath = "./m/v4/";
    await x.init({
        detPath: `${modelBasePath}/ppocr_det.onnx`,
        recPath: `${modelBasePath}/ppocr_v4_rec_doc.onnx`,
        dic: fs.readFileSync("../assets/ppocrv4_doc_dict.txt").toString(),
        layoutDic: "text\ntitle\nfigure\nfigure_caption\ntable\ntable_caption\nheader\nfooter\nreference\nequation",
        log: true,
        // dev: true,
        ort,
    });
    document.body.style.display = "flex";
    document.body.style.flexWrap = "wrap";
    const only = 0;
    for (let i = 1; i <= 5; i++) {
        if (i !== only && only !== 0) continue;
        const src = `../test/layout_img/${i}.svg`;
        const img = new Image();
        img.src = src;
        img.onload = async () => {
            const c = document.createElement("canvas");
            const ratio = 0.3;
            c.width = Math.floor(img.naturalWidth * ratio);
            c.height = Math.floor(img.naturalHeight * ratio);
            const p = document.createElement("div");
            const h = document.createElement("span");
            h.innerText = i;
            p.append(h, c);
            document.body.append(p);
            const ctx = c.getContext("2d");
            ctx.save();
            ctx.scale(ratio, ratio);
            ctx.drawImage(img, 0, 0);
            ctx.restore();
            const detResult = await x.det(ctx.getImageData(0, 0, c.width, c.height));
            const mainLine = detResult.map((i, n) => ({ ...i, text: n.toString(), mean: 1 }));
            const l = x.analyzeLayout(mainLine);

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
        };
    }
}
