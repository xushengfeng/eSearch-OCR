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
├── layout.ts  # 版面分析
└── untils.ts  # 工具函数
```

## 注意事项

- 使用 `npx tsc --noEmit` 进行类型检查，不要提交有类型错误的代码
- 使用 `npx biome check .` 检查代码风格
- 修改代码后请运行类型检查确保无误
