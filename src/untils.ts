export function int(num: number) {
    return num > 0 ? Math.floor(num) : Math.ceil(num);
}
export function clip(n: number, min: number, max: number) {
    return Math.max(min, Math.min(n, max));
}
/**
 *
 * @param {ImageData} data 原图
 * @param {number} w 输出宽
 * @param {number} h 输出高
 */
export function resizeImg(data: ImageData, w: number, h: number) {
    let x = data2canvas(data);
    let src = document.createElement("canvas");
    src.width = w;
    src.height = h;
    src.getContext("2d").scale(w / data.width, h / data.height);
    src.getContext("2d").drawImage(x, 0, 0);
    return src.getContext("2d").getImageData(0, 0, w, h);
}
export function data2canvas(data: ImageData, w?: number, h?: number) {
    let x = document.createElement("canvas");
    x.width = w || data.width;
    x.height = h || data.height;
    x.getContext("2d").putImageData(data, 0, 0);
    return x;
}
export function toPaddleInput(image: ImageData, mean: number[], std: number[]) {
    const imagedata = image.data;
    const redArray: number[][] = [];
    const greenArray: number[][] = [];
    const blueArray: number[][] = [];
    let x = 0,
        y = 0;
    for (let i = 0; i < imagedata.length; i += 4) {
        if (!blueArray[y]) blueArray[y] = [];
        if (!greenArray[y]) greenArray[y] = [];
        if (!redArray[y]) redArray[y] = [];
        redArray[y][x] = (imagedata[i] / 255 - mean[0]) / std[0];
        greenArray[y][x] = (imagedata[i + 1] / 255 - mean[1]) / std[1];
        blueArray[y][x] = (imagedata[i + 2] / 255 - mean[2]) / std[2];
        x++;
        if (x == image.width) {
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
        let l: { d: number; n: string; c: number }[] = [];
        for (let i = 1; i < this.tl.length; i++) {
            const d = this.tl[i].n - this.tl[i - 1].n;
            const name = this.tl[i - 1].t;
            let f = l.find((x) => x.n === name);
            if (f) {
                f.c++;
                f.d += d;
            } else l.push({ d: d, n: name, c: 1 });
        }
        const x: string[] = [];
        for (let i of l) {
            const t = i.c > 1 ? `${i.n}x${i.c}` : i.n;
            x.push(`${t} ${i.d}`);
        }
        x.push(this.tl.at(-1).t);
        console.log(`${this.name} ${l.map((i) => i.d).reduce((p, c) => p + c, 0)}ms: `, x.join(" "));
    }
}
