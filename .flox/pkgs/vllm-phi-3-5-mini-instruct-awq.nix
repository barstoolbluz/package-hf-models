# Phi-3.5-mini-instruct AWQ 4-bit — dual layout for Triton + vanilla vLLM
#
# SM75+ (INT4, all CUDA GPUs), 2.2 GB on disk.
# Quantized from microsoft/Phi-3.5-mini-instruct via model-quantizer (AutoAWQ).
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/phi-3-5-mini-instruct-awq.json);
in
mkHfModel {
  pname = "vllm-phi-3.5-mini-instruct-awq";
  baseVersion = "1.0.1";
  inherit buildMeta;
  srcPath = /mnt/scratch/models/inferencing/hub/models--microsoft--Phi-3.5-mini-instruct-AWQ;
  tritonModelName = "phi3_5_mini_instruct_awq";
  slug = "microsoft--Phi-3.5-mini-instruct-AWQ";
  snapshotId = "d9795a43c4d5249522df7902d274d170c8b7ae6e96eb5c9dfb15f1760b287a17";
  vllmDefaults = {
    gpu_memory_utilization = 0.85;
    max_model_len = 4096;
    dtype = "float16";
    quantization = "awq";
    enable_log_requests = false;
  };
}
