export interface ModelVersion {
    det: string;
    rec: string;
    dic: string;
    basePath: string;
}

export interface ModelPaths {
    det: string;
    rec: string;
    dic: string;
    basePath: string;
}

export interface CheckResult {
    det: boolean;
    rec: boolean;
    dic: boolean;
    allExist: boolean;
}

export const MODEL_VERSIONS: Record<string, ModelVersion>;

export function getModelPath(version?: string): ModelPaths;
export function checkModelExists(version?: string): CheckResult;
export function getDownloadInfo(): Record<string, string>;
export function checkAndWarn(version?: string): boolean;
export function getAvailableVersions(): string[];
export function getLocalModels(): string[];
