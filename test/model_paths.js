const fs = require("node:fs");
const path = require("node:path");

// 模型版本配置
// 字典文件现在都放在模型目录中，不再使用 assets目录
const MODEL_VERSIONS = {
    v4: {
        det: "ppocr_det.onnx",
        rec: "ppocr_rec.onnx",
        dic: "ppocr_keys_v1.txt", // 字典文件在模型目录中
        basePath: "./m/v4/",
    },
    v5: {
        det: "ppocr_v5_mobile_det.onnx",
        rec: "ppocr_v5_mobile_rec.onnx",
        dic: "ppocrv5_dict.txt", // 字典文件在模型目录中
        basePath: "./m/v5/",
    },
    v5_server: {
        det: "ppocr_v5_server_det.onnx",
        rec: "ppocr_v5_server_rec.onnx",
        dic: "ppocrv5_dict.txt", // 字典文件在模型目录中
        basePath: "./m/v5/",
    },
    v6_tiny: {
        det: "ppocr6_tiny_det.onnx",
        rec: "ppocr6_tiny_rec.onnx",
        dic: "dic.txt", // 字典文件在模型目录中
        basePath: "./m/ppocr_v6_tiny/",
    },
    v6_small: {
        det: "ppocr6_small_det.onnx",
        rec: "ppocr6_small_rec.onnx",
        dic: "dic.txt", // 字典文件在模型目录中
        basePath: "./m/ppocr_v6_small/",
    },
    v6_medium: {
        det: "ppocr6_medium_det.onnx",
        rec: "ppocr6_medium_rec.onnx",
        dic: "dic.txt", // 字典文件在模型目录中
        basePath: "./m/ppocr_v6_medium/",
    },
};

// 获取模型路径
function getModelPath(version = "v5") {
    const config = MODEL_VERSIONS[version];
    if (!config) {
        throw new Error(`未知的模型版本: ${version}。可用版本: ${Object.keys(MODEL_VERSIONS).join(", ")}`);
    }

    return {
        det: path.join(config.basePath, config.det),
        rec: path.join(config.basePath, config.rec),
        dic: path.join(config.basePath, config.dic), // 字典文件现在都在模型目录中
        basePath: config.basePath,
    };
}

// 检查模型文件是否存在
function checkModelExists(version = "v5") {
    const paths = getModelPath(version);
    const results = {
        det: fs.existsSync(paths.det),
        rec: fs.existsSync(paths.rec),
        dic: fs.existsSync(paths.dic),
        allExist: false,
    };
    results.allExist = results.det && results.rec && results.dic;
    return results;
}

// 获取模型下载信息
function getDownloadInfo() {
    const baseUrl = "https://github.com/xushengfeng/eSearch-OCR/releases/download/4.0.0";
    return {
        v4: `${baseUrl}/ch.zip`,
        v5_mobile: `${baseUrl}/ppocr_v5_mobile.zip`,
        v5_server: `${baseUrl}/ppocr_v5_server.zip`,
        v6_tiny: `${baseUrl}/ppocr_v6_tiny.zip`,
        v6_small: `${baseUrl}/ppocr_v6_small.zip`,
        v6_medium: `${baseUrl}/ppocr_v6_medium.zip`,
        doc_cls: "https://github.com/xushengfeng/eSearch-OCR/releases/download/8.1.0/doc_cls.onnx",
    };
}

// 提醒用户模型可能不存在
function checkAndWarn(version = "v5") {
    const exists = checkModelExists(version);
    if (!exists.allExist) {
        const missing = [];
        if (!exists.det) missing.push("det模型");
        if (!exists.rec) missing.push("rec模型");
        if (!exists.dic) missing.push("字典文件");

        console.warn(`[模型检查] ${version} 版本的以下文件缺失:`);
        for (const item of missing) console.warn(`  - ${item}`);

        const downloadInfo = getDownloadInfo();
        console.warn("[下载地址] 请从以下地址下载模型:");
        if (downloadInfo[version]) {
            console.warn(`  ${downloadInfo[version]}`);
        } else {
            console.warn(`  ${downloadInfo.v5_mobile}`);
        }
        console.warn("[使用说明] 下载后解压到 test/m/ 目录下对应的版本文件夹中");

        return false;
    }
    return true;
}

// 获取所有可用的模型版本
function getAvailableVersions() {
    return Object.keys(MODEL_VERSIONS);
}

// 获取当前可用的本地模型
function getLocalModels() {
    const available = [];
    for (const version of Object.keys(MODEL_VERSIONS)) {
        if (checkModelExists(version).allExist) {
            available.push(version);
        }
    }
    return available;
}

module.exports = {
    MODEL_VERSIONS,
    getModelPath,
    checkModelExists,
    getDownloadInfo,
    checkAndWarn,
    getAvailableVersions,
    getLocalModels,
};

// 如果直接运行此文件，显示模型状态
if (require.main === module) {
    console.log("=== 模型路径配置工具 ===\n");

    console.log("可用的模型版本:");
    for (const v of getAvailableVersions()) {
        const status = checkModelExists(v);
        const statusText = status.allExist ? "✓ 已存在" : "✗ 缺失";
        console.log(`  ${v}: ${statusText}`);
    }

    console.log("\n本地可用的模型:");
    const localModels = getLocalModels();
    if (localModels.length === 0) {
        console.log("  无");
    } else {
        for (const v of localModels) console.log(`  ${v}`);
    }

    console.log("\n模型下载地址:");
    const downloadInfo = getDownloadInfo();
    for (const [key, url] of Object.entries(downloadInfo)) {
        console.log(`  ${key}: ${url}`);
    }

    console.log("\n使用示例:");
    console.log("  const { getModelPath, checkAndWarn } = require('./model_paths');");
    console.log("  const paths = getModelPath('v5');");
    console.log("  checkAndWarn('v5');");
}
