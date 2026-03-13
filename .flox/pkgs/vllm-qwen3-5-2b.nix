# Qwen3.5-2B (unquantized, already under 5GB) for vLLM
#
# HF cache snapshots use symlinks to ../../blobs/, so we need to pass the
# entire model directory (including blobs/) and resolve symlinks during copy.
{ pkgs }:

let
  modelRoot = /mnt/scratch/models/inferencing/hub/models--Qwen--Qwen3.5-2B;
  slug = "Qwen--Qwen3.5-2B";
  snapshotId = "15852e8c16360a2fea060d615a32b45270f8a8fc";
in
pkgs.runCommand "vllm-qwen3.5-2b" {} ''
  _snap="$out/share/models/hub/models--${slug}/snapshots/${snapshotId}"
  mkdir -p "$_snap"
  # Follow symlinks (-L) to resolve blob references
  cp -rL ${modelRoot}/snapshots/${snapshotId}/* "$_snap/"
  # Remove non-essential files to save space
  rm -f "$_snap/.gitattributes" "$_snap/README.md" "$_snap/LICENSE"
  mkdir -p "$out/share/models/hub/models--${slug}/refs"
  echo -n "${snapshotId}" > "$out/share/models/hub/models--${slug}/refs/main"
''
