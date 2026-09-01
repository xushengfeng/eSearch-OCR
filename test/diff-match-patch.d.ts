declare module "diff-match-patch" {
    class DiffMatchPatch {
        diff_main(text1: string, text2: string): [number, string][];
    }
    export default DiffMatchPatch;
}
