# package-hf-models

Packages HuggingFace model weights into immutable Nix store paths for serving by [vLLM](https://docs.vllm.ai/), [Triton Inference Server](https://github.com/triton-inference-server/server) (vLLM backend), and [SGLang](https://sgl-project.github.io/).

Instead of downloading weights at container or service startup, runtimes mount a pre-built Nix store path. This makes model deployment reproducible, cacheable, and versionable — the same hash always contains the same weights.

## Output layouts

The shared builder [`mkHfModel.nix`](.flox/pkgs/mkHfModel.nix) produces one of two directory layouts depending on whether `slug` and `snapshotId` are provided.

### Single layout (Triton-only)

When `slug` is null (default), weights are copied directly:

```
$out/share/models/<tritonModelName>/
  config.pbtxt              # vLLM Triton backend config
  model-defaults.json       # vLLM engine parameters (no "model" key; resolved at runtime)
  weights/                  # model files copied here (cp -rL)
```

### Dual layout (vLLM + Triton)

When both `slug` and `snapshotId` are provided, the builder creates an HF cache tree alongside the Triton model directory:

```
$out/share/models/hub/models--<org>--<model>/
  refs/main                                     # contains snapshotId
  snapshots/<hash>/                             # model files (single copy, cp -rL)

$out/share/models/<tritonModelName>/
  config.pbtxt
  model-defaults.json
  weights -> ../hub/models--<org>--<model>/snapshots/<hash>   # relative symlink
```

The dual layout mirrors HuggingFace's cache structure so vLLM can load models via `HF_HUB_CACHE`, while Triton discovers the standard `<model>/weights/` directory. Zero file duplication.

## The shared builder — `mkHfModel.nix`

All standard model packages are built by calling `mkHfModel`:

```nix
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:
mkHfModel {
  pname = "vllm-qwen3.5-2b";
  baseVersion = "1.0.0";
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/vllm-qwen3-5-2b.json);
  srcPath = /mnt/scratch/models/inferencing/hub/models--Qwen--Qwen3.5-2B;
  tritonModelName = "qwen3_5_2b";
  slug = "Qwen--Qwen3.5-2B";
  snapshotId = "15852e8c16360a2fea060d615a32b45270f8a8fc";
  vllmDefaults = { gpu_memory_utilization = 0.85; max_model_len = 4096; dtype = "auto"; };
};
```

### Parameters

| Parameter | Required | Description |
|---|---|---|
| `pname` | yes | Nix package name |
| `baseVersion` | yes | Semantic version (e.g. `"1.0.0"`) |
| `buildMeta` | yes | Parsed from `build-meta/<name>.json` |
| `srcPath` | yes | Local path to model weights on disk |
| `tritonModelName` | yes | Directory name under `$out/share/models/` |
| `vllmDefaults` | no | vLLM engine params written to `model-defaults.json` |
| `slug` | no | HF-style slug (`org--model`); enables dual layout |
| `snapshotId` | no | HF snapshot hash (required when `slug` is set) |

## Current packages

| Package | Model | Layout | Quant | Notes |
|---|---|---|---|---|
| `phi-4-mini-instruct-fp8-hf` | Phi-4-mini-instruct | single | FP8 (PyTorch) | Triton-only |
| `phi-4-mini-instruct-fp8-torchao` | Phi-4-mini-instruct | single | FP8 (TorchAO) | Triton-only |
| `phi-4-mini-instruct-fp8-sglang` | Phi-4-mini-instruct | HF-only | FP8 (TorchAO) | Custom build (see below) |
| `vllm-phi-3-5-mini-instruct-awq` | Phi-3.5-mini-instruct | dual | AWQ 4-bit | T4-compatible |
| `vllm-qwen3-5-2b` | Qwen3.5-2B | dual | none | T4-compatible |
| `vllm-qwen3-5-4b-fp8` | Qwen3.5-4B | single | FP8 (TorchAO) | |
| `vllm-smollm3-3b-fp8` | SmolLM3-3B | single | FP8 (TorchAO) | |

### Special case: `phi-4-mini-instruct-fp8-sglang`

This package is a **custom derivation** that does not use `mkHfModel`. It produces only the HF cache layout (no Triton `config.pbtxt`) and patches `tokenizer_config.json` — replacing `"TokenizersBackend"` with `"PreTrainedTokenizerFast"` — for compatibility with SGLang < 0.5.10 (which ships `transformers < 4.58`). This package can be removed once SGLang updates its transformers dependency.

## Build metadata versioning

Each package has a corresponding `build-meta/<name>.json`:

```json
{
  "build_version": 2,
  "force_increment": 0,
  "git_rev": "ca88e1655077dab1a0e8bc3583d98bb2575be501",
  "git_rev_short": "ca88e16",
  "changelog": "Dual layout: Triton + vanilla vLLM via mkHfModel with slug/snapshotId."
}
```

- `build_version` increments independently of git — bump it for weight-only changes that don't touch `.nix` files
- Final version string: `<baseVersion>+<git_rev_short>` (e.g. `1.0.0+ca88e16`)
- A version marker is written to `$out/share/<pname>/flox-build-version-<build_version>`

## How runtimes consume packages

| Runtime | Mechanism |
|---|---|
| **vLLM** | Set `HF_HUB_CACHE=$out/share/models/hub` — vLLM resolves the model via the HF cache layout (`refs/main` → snapshot dir) |
| **Triton** | Set `--model-repository=$out/share/models` — Triton discovers `<tritonModelName>/config.pbtxt` and loads weights from the `weights/` subdirectory |
| **SGLang** | Same as vLLM (`HF_HUB_CACHE`) — the HF cache layout is standard |

Dual-layout packages work for both vLLM and Triton simultaneously with zero file duplication.

## How to add a new model

1. Download or quantize weights to `/mnt/scratch/models/inferencing/`. Use the HF cache layout (`hub/models--<org>--<model>/`) for dual-layout packages, or a flat resolved directory for single-layout.

2. Create `build-meta/<name>.json`:
   ```json
   {
     "build_version": 1,
     "force_increment": 0,
     "git_rev": "<current git rev>",
     "git_rev_short": "<short rev>",
     "changelog": "Initial package."
   }
   ```

3. Create `.flox/pkgs/<name>.nix` calling `mkHfModel` with appropriate parameters. Use `slug` + `snapshotId` for dual layout.

4. Build:
   ```bash
   flox build -A <name>
   ```

5. Verify the output layout:
   ```bash
   ls -la result-<name>/share/models/
   ```

6. Publish:
   ```bash
   flox publish
   ```

## Source model locations

Weights live on local scratch storage before packaging:

- **HF cache layout:** `/mnt/scratch/models/inferencing/hub/models--<org>--<model>/` — used for dual-layout packages and packages that need the full HF tree (blobs + snapshots)
- **Pre-resolved:** `/mnt/scratch/models/inferencing/resolved/<org>--<model>/` — flat directory with no blob symlinks, used for single-layout packages

## Building and publishing

```bash
# Build a single package
flox build -A vllm-qwen3-5-2b

# Build all packages
flox build

# Inspect the result
ls -la result-vllm-qwen3-5-2b/share/models/

# Publish to FloxHub
flox publish
```
