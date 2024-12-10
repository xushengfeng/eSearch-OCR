# eSearch-OCR

本仓库是 [eSearch](https://github.com/xushengfeng/eSearch)的 OCR 服务依赖

支持本地 OCR（基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)）

[PaddleOCR License](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/LICENSE)

[paddle 预测库](https://paddle-inference.readthedocs.io/en/latest/user_guides/download_lib.html)

基于[onnxruntime](https://github.com/microsoft/onnxruntime)的 web runtime，使用 wasm 运行，未来可能使用 webgl 甚至是 webgpu。

模型需要转换为 onnx 才能使用：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) 或[在线转换](https://www.paddlepaddle.org.cn/paddle/visualdl/modelconverter/x2paddle)

部分模型已打包：[Releases](https://github.com/xushengfeng/eSearch-OCR/releases/tag/4.0.0)

在 js 文件下可以使用 electron 进行调试

## 使用

```shell
npm i esearch-ocr onnxruntime-web
```

web

```javascript
import * as ocr from "esearch-ocr";
import * as ort from "onnxruntime-web";
```

```javascript
const ocr = require("esearch-ocr");
const ort = require("onnxruntime-node");
```

> [!IMPORTANT]
> 需要手动安装 onnxruntime（onnxruntime-node 或 onnxruntime-web，视平台而定），并在`init`参数中传入`ort`
> 这样设计是因为 web 和 electron 可以使用不同的 ort，很难协调，不如让开发者自己决定

浏览器或 Electron 示例

```javascript
await ocr.init({
    detPath: "ocr/det.onnx",
    recPath: "ocr/rec.onnx",
    dic: "abcdefg...",
    ort,
});

const url = "data:image/png;base64,..."; // 还支持 HTMLImageElement | HTMLCanvasElement | ImageData
ocr.ocr(url)
    .then((result) => {}) // 见下面的解释
    .catch((e) => {});
```

或者

```javascript
const localOCR = await ocr.init({
    detPath: "ocr/det.onnx",
    recPath: "ocr/rec.onnx",
    dic: "abcdefg...",
    ort,
});

localOCR.ocr(/*像上面ocr.ocr一样调用*/);
```

这在需要多次运行 ocr 时非常有用

[node.js 示例](./test/test_node.js)，需要安装`canvas`

[演示项目](https://github.com/xushengfeng/webocr)

init type

```typescript
{
    ort: typeof import("onnxruntime-web");
    detPath: string;
    recPath: string;
    dic: string; // 文件内容，不是路径
    dev?: boolean;
    maxSide?: number;
    imgh?: number;
    imgw?: number;
    detShape?: [number, number]; // ppocr v3 需要指定为[960, 960], v4 为[640, 640]
    canvas?: (w: number, h: number) => any; // 用于node
    imageData?: any; // 用于node
    cv?: any;
}
```

对于返回的值

```ts
type resultType = {
    text: string;
    mean: number;
    box: BoxType; // ↖ ↗ ↘ ↙
    style: { bg: color; text: color }; // rgb数组，表示背景颜色和文字颜色，在简单移除文字时非常有用
}[];

{
    src: resultType; // 每个视觉行，rec输出
    columns: {
        // 分栏，如左右分栏
        src: resultType;
        outerBox: BoxType;
        parragraphs: {
            src: resultType;
            parse: resultType[0];
        }
        [];
    }
    [];
    parragraphs: resultType; // 聚合了columns的每个段落
}
```

合并的文字可以使用

```js
result.parragraphs.map((item) => item.text).join("\n");
```

除了 `ocr` 函数，还有`det`函数，可单独运行，检测文字坐标；`rec`函数，可单独运行，检测文字内容。具体定义可看类型提示。
