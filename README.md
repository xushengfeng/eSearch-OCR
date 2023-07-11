# eSearch-OCR

本仓库是 [eSearch](https://github.com/xushengfeng/eSearch)的 OCR 服务依赖

支持本地 OCR（基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)）

[PaddleOCR License](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/LICENSE)

[paddle 预测库](https://paddle-inference.readthedocs.io/en/latest/user_guides/download_lib.html)

基于[onnxruntime](https://github.com/microsoft/onnxruntime)的 web runtime，使用 wasm 运行，未来可能使用 webgl 甚至是 webgpu。

模型需要转换为 onnx 才能使用：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) 或[在线转换](https://www.paddlepaddle.org.cn/paddle/visualdl/modelconverter/x2paddle)

在 js 文件下使用 electron 进行调试（主要是 require 几个模块和 fs 读取字典，若想纯网页实现，可以自行修改）

## 使用

```shell
npm i esearch-ocr
```

web

```javascript
import * as ocr from "esearch-ocr";
```

nodejs

```javascript
const ocr = require("esearch-ocr");
```

```javascript
await lo.init({
    detPath: "ocr/det.onnx",
    recPath: "ocr/rec.onnx",
    dic: "",
});

let img = document.createElement("img");
img.src = "data:image/png;base64,...";
img.onload = async () => {
    let canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d").drawImage(img, 0, 0);
    ocr.ocr(canvas.getContext("2d").getImageData(0, 0, img.width, img.height))
        .then((l) => {})
        .catch((e) => {});
};
```

init type

```typescript
{
    detPath: string;
    recPath: string;
    dic: string; // raw !string[] && !filePath
    node?: boolean;
    dev?: boolean;
    maxSide?: number;
    imgh?: number;
    imgw?: number;
    ort?: typeof import("onnxruntime-web");
    detShape?: [number, number];
}
```

ocr type

```typescript
type PointType = [number, number]
ocr(img: ImageData): Promise<{
    text: string;
    mean: number;
    box: [PointType, PointType, PointType, PointType]; // lt rt rb lb
}[]>
```
