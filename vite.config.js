import { resolve } from "node:path";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
    // 打包配置
    build: {
        lib: {
            entry: resolve(__dirname, "src/main.ts"),
            name: "esearch-ocr",
        },
    },
    plugins: [dts({ rollupTypes: true })],
});
