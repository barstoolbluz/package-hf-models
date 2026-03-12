# Qwen3.5-4B FP8-TORCHAO for vLLM
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:

mkHfModel {
  pname = "vllm-qwen3.5-4b-fp8";
  version = "1.0.0";
  srcPath = /mnt/scratch/models/inferencing/hub/models--Qwen--Qwen3.5-4B-FP8-TORCHAO/snapshots/cbd334b0c03d4ef0e42ba39772bad7dd86ddfb3d;
  slug = "Qwen--Qwen3.5-4B-FP8-TORCHAO";
  snapshotId = "cbd334b0c03d4ef0e42ba39772bad7dd86ddfb3d";
}
