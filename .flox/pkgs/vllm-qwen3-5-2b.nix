# Qwen3.5-2B (unquantized, already under 5GB) for vLLM
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:

mkHfModel {
  pname = "vllm-qwen3.5-2b";
  version = "1.0.0";
  srcPath = /mnt/scratch/models/inferencing/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc;
  slug = "Qwen--Qwen3.5-2B";
  snapshotId = "15852e8c16360a2fea060d615a32b45270f8a8fc";
}
