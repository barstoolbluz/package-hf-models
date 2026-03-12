# mkHfModel.nix — shared builder for HuggingFace model packages (vLLM)
#
# Copies a local model snapshot into the HF cache layout expected by
# vllm-resolve-model's `flox` source:
#   $out/share/models/hub/models--<slug>/snapshots/<snapshotId>/
#
# Uses sandbox = "off" so local paths are accessible.
#
# Usage (from per-model .nix files):
#   { pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:
#   mkHfModel { pname = "..."; version = "..."; srcPath = /path/to/snapshot; ... }
{ runCommand }:
{ pname, version, srcPath, slug, snapshotId }:

runCommand pname {} ''
  _snap="$out/share/models/hub/models--${slug}/snapshots/${snapshotId}"
  mkdir -p "$_snap"
  cp -rL ${srcPath}/* "$_snap/"
  mkdir -p "$out/share/models/hub/models--${slug}/refs"
  echo -n "${snapshotId}" > "$out/share/models/hub/models--${slug}/refs/main"
''
