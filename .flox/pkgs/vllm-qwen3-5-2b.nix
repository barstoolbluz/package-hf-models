# Qwen3.5-2B (SM75+, unquantized, ~4 GB) — dual layout for Triton + vanilla vLLM
#
# HF cache snapshots use symlinks to ../../blobs/, so srcPath must be the
# entire model directory (including blobs/) and we resolve symlinks during copy.
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/vllm-qwen3-5-2b.json);
in
mkHfModel {
  pname = "vllm-qwen3.5-2b";
  baseVersion = "1.0.0";
  inherit buildMeta;
  srcPath = /mnt/scratch/models/inferencing/hub/models--Qwen--Qwen3.5-2B;
  tritonModelName = "qwen3_5_2b";
  slug = "Qwen--Qwen3.5-2B";
  snapshotId = "15852e8c16360a2fea060d615a32b45270f8a8fc";
  vllmDefaults = {
    gpu_memory_utilization = 0.85;
    max_model_len = 4096;
    dtype = "auto";
    enable_log_requests = false;
    trust_remote_code = true;
  };
}
