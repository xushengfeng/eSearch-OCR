export type Point = { x: number; y: number };
export type Contour = Point[];
type RotatedRect = {
    center: Point;
    size: { width: number; height: number };
    angle: number;
};

/**
 * 计算最小包围旋转矩形（Rotating Calipers 算法简化版）
 * @param contour 轮廓点集
 */
export function minAreaRect(contour: Contour): RotatedRect {
    if (contour.length === 0) throw new Error("Empty contour");

    // 计算凸包（Andrew's monotone chain 算法）
    const hull = convexHull([...contour]);

    let minArea = Number.POSITIVE_INFINITY;
    const result: RotatedRect = {
        center: { x: 0, y: 0 },
        size: { width: 0, height: 0 },
        angle: 0,
    };

    // 遍历凸包每条边计算最小矩形
    for (let i = 0; i < hull.length; i++) {
        const p1 = hull[i];
        const p2 = hull[(i + 1) % hull.length];

        // 计算当前边的方向向量
        const edge = { x: p2.x - p1.x, y: p2.y - p1.y };
        const length = Math.hypot(edge.x, edge.y);
        const [dx, dy] = [edge.x / length, edge.y / length]; // 单位方向向量

        // 计算投影极值
        let minVal = Number.POSITIVE_INFINITY;
        let maxVal = Number.NEGATIVE_INFINITY;
        let minPerp = Number.POSITIVE_INFINITY;
        let maxPerp = Number.NEGATIVE_INFINITY;

        for (const p of hull) {
            // 投影到当前边方向
            const proj = (p.x - p1.x) * dx + (p.y - p1.y) * dy;
            minVal = Math.min(minVal, proj);
            maxVal = Math.max(maxVal, proj);

            // 投影到垂直方向
            const perp = -(p.x - p1.x) * dy + (p.y - p1.y) * dx;
            minPerp = Math.min(minPerp, perp);
            maxPerp = Math.max(maxPerp, perp);
        }

        // 计算当前方向的包围矩形参数
        const currentArea = (maxVal - minVal) * (maxPerp - minPerp);
        if (currentArea < minArea) {
            minArea = currentArea;

            // 计算矩形中心点
            const centerProj = (minVal + maxVal) / 2;
            const centerPerp = (minPerp + maxPerp) / 2;
            result.center = {
                x: p1.x + dx * centerProj - dy * centerPerp,
                y: p1.y + dy * centerProj + dx * centerPerp,
            };

            // 尺寸和角度
            result.size = {
                width: maxVal - minVal,
                height: maxPerp - minPerp,
            };
            result.angle = Math.atan2(dy, dx) * (180 / Math.PI);
        }
    }

    // 确保宽度大于高度（OpenCV 惯例）
    if (result.size.width < result.size.height) {
        [result.size.width, result.size.height] = [result.size.height, result.size.width];
        result.angle += 90;
    }

    // 规范化角度到 [0, 180)
    result.angle = ((result.angle % 180) + 180) % 180;

    return result;
}

// 凸包计算（Andrew's monotone chain 算法）
function convexHull(points: Point[]): Point[] {
    points.sort((a, b) => a.x - b.x || a.y - b.y);

    const lower: Point[] = [];
    for (const p of points) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
            lower.pop();
        }
        lower.push(p);
    }

    const upper: Point[] = [];
    for (let i = points.length - 1; i >= 0; i--) {
        const p = points[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
            upper.pop();
        }
        upper.push(p);
    }

    return lower.slice(0, -1).concat(upper.slice(0, -1));
}

// 向量叉积辅助函数
function cross(o: Point, a: Point, b: Point): number {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function findContours(src: number[][], contours: Point[][], method = "CHAIN_APPROX_SIMPLE"): void {
    const height = src.length;
    const width = height > 0 ? src[0].length : 0;
    const visited: boolean[][] = Array.from({ length: height }, () => new Array(width).fill(false));

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (src[y][x] !== 0 && !visited[y][x] && isContourPoint(src, x, y)) {
                const contour = traceContour(src, visited, x, y, method === "CHAIN_APPROX_SIMPLE");
                contours.push(contour);
            }
        }
    }
}

function isContourPoint(src: number[][], x: number, y: number): boolean {
    // Check 4-neighborhood for background pixels
    return (
        src[y][x] !== 0 &&
        ((y > 0 && src[y - 1][x] === 0) ||
            (y < src.length - 1 && src[y + 1][x] === 0) ||
            (x > 0 && src[y][x - 1] === 0) ||
            (x < src[0].length - 1 && src[y][x + 1] === 0))
    );
}

function traceContour(
    src: number[][],
    visited: boolean[][],
    startX: number,
    startY: number,
    simplify: boolean,
): Point[] {
    const contour: Point[] = [];
    let current: Point = { x: startX, y: startY };
    let prev: Point = { x: startX - 1, y: startY }; // Initial direction: left

    const usedDir = new Map<number, number[]>();
    const toUseDir = new Map<number, number[]>();

    function getPointId(p: Point) {
        return p.x + p.y * src[0].length;
    }

    function getPointById(id: number) {
        const y = Math.floor(id / src[0].length);
        const x = id % src[0].length;
        return { x, y };
    }

    function p2p(p1: Point, p2: Point) {
        const p1Id = getPointId(p1);
        const p2Id = getPointId(p2);

        const p12p2 = getDirection(p2.x - p1.x, p2.y - p1.y);
        const p22p1 = getDirection(p1.x - p2.x, p1.y - p2.y);

        const p1L = usedDir.get(p1Id) ?? [];
        const p2L = usedDir.get(p2Id) ?? [];

        usedDir.set(p1Id, [...p1L, p12p2]);
        usedDir.set(p2Id, [...p2L, p22p1]);
    }

    function jumpNext(useD: number) {
        const pointId = getPointId(current);

        prev = current;
        current = { x: current.x + directions[useD].dx, y: current.y + directions[useD].dy };

        p2p(prev, current);

        const nextD = toUseDir.get(pointId) ?? [];

        const n = nextD.filter((d) => d !== useD);
        if (n.length > 0) toUseDir.set(pointId, n);
        else toUseDir.delete(pointId);
    }

    usedDir.set(getPointId(current), [getDirection(-1, 0)]);

    // console.log("Start", current);

    let count = 0;

    do {
        contour.push(current);
        visited[current.y][current.x] = true;

        const nextD = findNextPoints(src, usedDir, current);
        // console.log(current, nextD);

        if (nextD.length === 0) {
            // console.log(toUseDir);

            if (toUseDir.size === 0) {
                break;
            }
            const [id, dirs] = Array.from(toUseDir.entries()).at(0)!;
            const useD = dirs[0];
            current = getPointById(id);
            jumpNext(useD);
        }
        if (nextD.length >= 1) {
            const pointId = getPointId(current);

            toUseDir.set(pointId, nextD);

            const useD = nextD[0];

            jumpNext(useD);
        }
        count++;
    } while (count < 1000000000);

    return simplify ? simplifyContour(contour) : contour;
}
const directions = [
    { dx: 1, dy: 0 }, // Right
    { dx: 1, dy: -1 }, // Top-Right
    { dx: 0, dy: -1 }, // Top
    { dx: -1, dy: -1 }, // Top-Left
    { dx: -1, dy: 0 }, // Left
    { dx: -1, dy: 1 }, // Bottom-Left
    { dx: 0, dy: 1 }, // Bottom
    { dx: 1, dy: 1 }, // Bottom-Right
];

function findNextPoint(src: number[][], visited: boolean[][], current: Point, prev: Point): Point | null {
    const startDir = getStartDirection(prev, current);
    for (let i = 0; i < 4; i++) {
        const dirIdx = (startDir + i) % 4;
        const { dx, dy } = directions[dirIdx];
        const x = current.x + dx;
        const y = current.y + dy;

        if (x >= 0 && x < src[0].length && y >= 0 && y < src.length) {
            if (src[y][x] !== 0 && !visited[y][x] && isContourPoint(src, x, y)) {
                return { x, y };
            }
        }
    }
    return null;
}

function findNextPoints(src: number[][], usedDir: Map<number, number[]>, current: Point) {
    function getPointId(p: Point) {
        return p.x + p.y * src[0].length;
    }

    const usedDirs = usedDir.get(getPointId(current)) ?? [];

    const d: number[] = [];

    for (const [i, { dx, dy }] of directions.entries()) {
        if (usedDirs.includes(i)) continue;
        const x = current.x + dx;
        const y = current.y + dy;
        if (x >= 0 && x < src[0].length && y >= 0 && y < src.length) {
            if (isContourPoint(src, x, y)) {
                d.push(i);
            }
        }
    }
    return d;
}

function getStartDirection(prev: Point, current: Point): number {
    const dx = current.x - prev.x;
    const dy = current.y - prev.y;

    return getDirection(dx, dy);
}

function getDirection(dx: number, dy: number) {
    const index = directions.findIndex(({ dx: ddx, dy: ddy }) => dx === ddx && dy === ddy);
    return index === -1 ? 0 : index; // Default to Right
}

function simplifyContour(contour: Point[]): Point[] {
    if (contour.length < 3) return [...contour];

    const simplified: Point[] = [contour[0]];
    for (let i = 1; i < contour.length - 1; i++) {
        const prev = simplified[simplified.length - 1];
        const current = contour[i];
        const next = contour[i + 1];

        if (!isCollinear(prev, current, next)) {
            simplified.push(current);
        }
    }
    simplified.push(contour[contour.length - 1]);
    return simplified;
}

function isCollinear(a: Point, b: Point, c: Point): boolean {
    // Check if three points are collinear using cross product
    return (b.x - a.x) * (c.y - b.y) === (b.y - a.y) * (c.x - b.x);
}

export { findContours };
