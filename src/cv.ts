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
