let canvas = (width: number, height: number) => {
    const c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    return c;
};

export function newCanvas(width: number, height: number) {
    return canvas(width, height);
}

export function setCanvas(x) {
    canvas = x;
}

export function int(num: number) {
    return num > 0 ? Math.floor(num) : Math.ceil(num);
}
export function clip(n: number, min: number, max: number) {
    return Math.max(min, Math.min(n, max));
}
/**
 *
 * @param  data 原图
 * @param  w 输出宽
 * @param  h 输出高
 * @param  fill 小于输出宽高的部分填充还是拉伸
 */
export function resizeImg(data: ImageData, w: number, h: number, fill?: "fill") {
    const ctx = resizeImgC(data, w, h, fill);
    return ctx.getImageData(0, 0, w, h);
}

/**
 *
 * @param  data 原图
 * @param  w 输出宽
 * @param  h 输出高
 * @param  fill 小于输出宽高的部分填充还是拉伸
 */
export function resizeImgC(data: ImageData, w: number, h: number, fill?: "fill") {
    const x = data2canvas(data);
    const src = newCanvas(w, h);
    const ctx = src.getContext("2d");
    if (!ctx) throw new Error("canvas context is null");
    if (fill === "fill") {
        ctx.scale(Math.min(w / data.width, 1), Math.min(h / data.height, 1));
    } else {
        ctx.scale(w / data.width, h / data.height);
    }
    ctx.drawImage(x, 0, 0);
    return ctx;
}
export function data2canvas(data: ImageData, w?: number, h?: number) {
    const x = newCanvas(w || data.width, h || data.height);
    const ctx = x.getContext("2d");
    if (!ctx) throw new Error("canvas context is null");
    ctx.putImageData(data, 0, 0);
    return x;
}
export function toPaddleInput(image: ImageData, mean: number[], std: number[]) {
    const imagedata = image.data;
    const redArray: number[][] = [];
    const greenArray: number[][] = [];
    const blueArray: number[][] = [];
    let x = 0;
    let y = 0;
    for (let i = 0; i < imagedata.length; i += 4) {
        if (!blueArray[y]) blueArray[y] = [];
        if (!greenArray[y]) greenArray[y] = [];
        if (!redArray[y]) redArray[y] = [];
        redArray[y][x] = (imagedata[i] / 255 - mean[0]) / std[0];
        greenArray[y][x] = (imagedata[i + 1] / 255 - mean[1]) / std[1];
        blueArray[y][x] = (imagedata[i + 2] / 255 - mean[2]) / std[2];
        x++;
        if (x === image.width) {
            x = 0;
            y++;
        }
    }

    return [blueArray, greenArray, redArray];
}
export type AsyncType<T> = T extends Promise<infer U> ? U : never;
export type SessionType = AsyncType<ReturnType<typeof import("onnxruntime-common").InferenceSession.create>>;

export class tLog {
    private tl: { t: string; n: number }[] = [];
    private name: string;
    constructor(taskName: string) {
        this.name = taskName;
    }
    l(name: string) {
        const now = performance.now();
        this.tl.push({ t: name, n: now });
        const l: { d: number; n: string; c: number }[] = [];
        for (let i = 1; i < this.tl.length; i++) {
            const d = this.tl[i].n - this.tl[i - 1].n;
            const name = this.tl[i - 1].t;
            const f = l.find((x) => x.n === name);
            if (f) {
                f.c++;
                f.d += d;
            } else l.push({ d: d, n: name, c: 1 });
        }
        const x: string[] = [];
        for (const i of l) {
            const t = i.c > 1 ? `${i.n}x${i.c}` : i.n;
            x.push(`${t} ${i.d}`);
        }
        x.push((this.tl.at(-1) as (typeof this.tl)[0]).t);
        console.log(`${this.name} ${l.map((i) => i.d).reduce((p, c) => p + c, 0)}ms: `, x.join(" "));
    }
}
