# eSearch-OCR 开发指南

## 项目概述

eSearch-OCR 是一个基于 PaddleOCR 模型的 ONNX 运行时 OCR 库。

## 开发工具链

- **包管理器**: pnpm
- **构建工具**: Vite
- **类型检查**: TypeScript (strict mode)
- **代码格式化/Lint**: Biome

## 常用命令

```bash
# 安装依赖
pnpm install

# 开发模式
pnpm dev

# 构建
pnpm build

# 类型检查
npx tsc --noEmit

# 代码检查和格式化
npx biome check .
npx biome format .
npx biome lint .
```

## 项目结构

```
src/
├── main.ts    # 主入口，OCR 核心逻辑
├── cls.ts     # 文档方向分类
├── cv.ts      # 计算机视觉工具（轮廓检测、最小矩形）
├── layout.ts  # 版面分析（仅用于版面分析模型，与main的算法分析版面有所区别）
└── untils.ts  # 工具函数
test
    accuracy.test.ts 准确率测试，防止退化，但是耗费时间久，一般在更新算法后最后才执行把关
    det.test.ts det检测流程测试
    rec.test.ts
    layout.test.ts
    docCls.test.ts

    detectReadingDir.test.ts 用于layout的阅读方向检测
    getImgColor.test.ts 用于det
    matchBestBox.test.ts 用于det
```
test下的其他文件用于electron运行视觉测试，不参与自动化开发

test.ts使用vitest测试，需要进入test文件夹才能用，因为测试文件夹下安装了运行时等依赖，与主目录隔离

## 概念

ocr流程：det识别文本区域，提取出来文本框，单行文本，交给rec识别为文本，layout（afAfRec）用传统算法统计分析栏、段落等结构。核心流程就是main initOCR 返回的ocr函数里面所表示的。

det中，模型返回的是每个像素的置信度，以0.3作为阈值，然后通过cv找边缘，匹配框等提取单行文本框，这里，还用了颜色检测来让框尽可能贴近文本边缘。这里只处理了直线文本，不考虑弯折情况。

layout只用det的框来分析，也会分析rec结果判断段落合并方式，如英文要有空格，中文不用。首先根据长宽分布角度判断阅读方向，统一向量处理的基本方向。det会因为空格等把一行文本拆开，这里需要判断间距来合并这些文本框。然后因为文本排版的复杂，可能会分栏，如左右布局，或者ui不同view，漫画对话框等，根据它们临近区分所属的栏或者说是块。在栏中分段，一些纸媒会有空格分段，现代则是换行gap分段，这些通过间距分布来判断，最后实现分段。

## 注意事项

- 使用 `npx tsc --noEmit` 进行类型检查，不要提交有类型错误的代码
- 使用 `npx biome check .` 检查代码风格
- 修改代码后请运行类型检查确保无误
- 如果行为更改也需要更改readme和此agents.md
