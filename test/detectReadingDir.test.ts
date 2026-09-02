import { describe, expect, it } from "vitest";
import { type ReadingDir, detectReadingDir } from "../src/main";

describe("detectReadingDir", () => {
    it("should return default direction for empty input", () => {
        const result = detectReadingDir([]);
        expect(result.readingDir).toEqual({ block: "tb", inline: "lr" });
        expect(result.angle.reading.inline).toBe(0);
        expect(result.angle.reading.block).toBe(90);
    });

    it("should detect horizontal direction for 0 degree angles", () => {
        // 横排文本 inline 角度接近 0 度
        const angles = [0, 0, 0, 0];
        const result = detectReadingDir(angles);
        expect(result.readingDir.inline).toBe("lr");
        expect(result.readingDir.block).toBe("tb");
    });

    it("should detect vertical direction for 90 degree angles", () => {
        // 竖排文本 inline 角度接近 90 度
        const angles = [90, 90, 90, 90];
        const result = detectReadingDir(angles);
        expect(result.readingDir.inline).toBe("tb");
        expect(result.readingDir.block).toBe("rl");
    });

    it("should detect direction with slight angle variation", () => {
        // 允许轻微角度偏差
        const angles = [85, 88, 92, 90];
        const result = detectReadingDir(angles);
        expect(result.readingDir.inline).toBe("tb");
        expect(result.readingDir.block).toBe("rl");
    });

    it("有其他方向", () => {
        const angles = [0, 88, 92, 90, 91, 2];
        const result = detectReadingDir(angles);
        expect(result.readingDir.inline).toBe("tb");
        expect(result.readingDir.block).toBe("rl");
    });

    it("存在反向", () => {
        const angles = [91, 89, 270, 269, 271];
        // 应该tb而不是bt，因为提示限制
        const result = detectReadingDir(angles);
        expect(result.readingDir.inline).toBe("tb");
        expect(result.readingDir.block).toBe("rl");
    });

    it("should respect custom docDirs", () => {
        const angles = [0, 0, 0];
        const customDirs = [{ block: "tb", inline: "lr" }] as ReadingDir[];
        const result = detectReadingDir(angles, customDirs);
        expect(result.readingDir.inline).toBe("lr");
        expect(result.readingDir.block).toBe("tb");
    });
});
