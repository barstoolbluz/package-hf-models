# Phi-4-mini-instruct FP8-TORCHAO for vLLM
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/phi-4-mini-instruct-fp8-torchao.json);
in
mkHfModel {
  pname = "phi-4-mini-instruct-fp8-torchao";
  baseVersion = "1.0.0";
  inherit buildMeta;
  srcPath = /mnt/scratch/models/inferencing/hub/models--microsoft--Phi-4-mini-instruct-FP8-TORCHAO/snapshots/b63ecd840bb9835f35e6d884d47810c4deec89dc;
  slug = "microsoft--Phi-4-mini-instruct-FP8-TORCHAO";
  snapshotId = "b63ecd840bb9835f35e6d884d47810c4deec89dc";
}
