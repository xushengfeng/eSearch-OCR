// @ts-ignore
import { findContours } from "../../src/cv.ts";

const src = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
];

const src2 = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
];

const src3 = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
];

const src4 = [
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
];

const src5 = [
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
];

const src6 = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
];

const s = [src, src2, src3, src4, src5, src6];

for (const i of s) {
    const x: { x: number; y: number }[][] = [];
    findContours(i, x, "");
    console.log(x);
    const r = structuredClone(i);
    for (const y of r) {
        for (let i = 0; i < y.length; i++) {
            y[i] = 0;
        }
    }
    for (const [index, c] of x.entries()) {
        for (const p of c) {
            r[p.y][p.x] = index + 1;
        }
    }
    console.log(r);
}
