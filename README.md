# package-hf-models

Packages HuggingFace model weights into immutable [Nix store](https://nix.dev/manual/nix/stable/store/) paths for serving by [vLLM](https://docs.vllm.ai/), [Triton Inference Server](https://github.com/triton-inference-server/server) (vLLM backend), and [SGLang](https://sgl-project.github.io/).

### Why package weights in Nix?

LLM inference runtimes typically download model weights at startup — from HuggingFace Hub, S3, or a shared filesystem. This is slow, non-reproducible, and hard to version. By packaging weights into the Nix store, every deployment gets an identical, content-addressed directory of model files. The same Nix hash always contains exactly the same weights. Runtimes mount the store path directly — no downloads, no caching surprises, no "works on my machine."

## Key concepts

Before diving into the layouts, here are the HuggingFace and Triton terms used throughout:

- **HuggingFace Hub** — the public registry where model authors publish weights. Each model lives at a path like `microsoft/Phi-4-mini-instruct`.
- **Slug** — a flattened version of the HuggingFace model path with `/` replaced by `--`. For example, `microsoft/Phi-4-mini-instruct` becomes `microsoft--Phi-4-mini-instruct`. HuggingFace's download tools use this format for local cache directories.
- **Snapshot ID** — a commit hash from HuggingFace Hub that pins a specific version of the model weights. Think of it like a git commit SHA for model files.
- **HF cache layout** — the directory structure that HuggingFace's `huggingface_hub` Python library creates when it downloads a model. It stores files under `hub/models--<slug>/snapshots/<hash>/`. Tools like vLLM and SGLang use this library internally, so they know how to find models in this layout.
- **Triton model repository** — Triton Inference Server discovers models by scanning a directory for subdirectories that contain a `config.pbtxt` file. Each subdirectory is one servable model.
- **`config.pbtxt`** — a Protobuf-text configuration file that tells Triton how to serve a model: which backend to use (vLLM in our case), input/output tensor shapes, and streaming configuration.
- **`model-defaults.json`** — a custom file this repo creates alongside `config.pbtxt`. It holds vLLM engine parameters (GPU memory fraction, max sequence length, data type, etc.) that the runtime reads when launching the model.
- **`$out`** — in Nix build expressions, `$out` is the output path in the Nix store (e.g. `/nix/store/abc123-vllm-qwen3.5-2b-1.0.0/`). Everything the build produces goes under this path.

## Output layouts

The shared builder [`mkHfModel.nix`](.flox/pkgs/mkHfModel.nix) produces one of two directory layouts depending on which parameters are provided.

### Single layout (Triton-only)

The simpler layout. Used when only Triton needs to serve the model. Weight files are copied directly into a `weights/` directory:

```
$out/share/models/<tritonModelName>/
  config.pbtxt            # tells Triton to use the vLLM backend
  model-defaults.json     # vLLM engine parameters (GPU memory, max length, dtype, etc.)
  weights/                # model files (tensors, tokenizer, config) copied here
```

Triton scans `$out/share/models/`, finds the `<tritonModelName>/` directory with its `config.pbtxt`, and serves the model.

### Dual layout (vLLM + Triton)

Used when both standalone vLLM *and* Triton need to serve the same packaged weights. The builder creates a full HuggingFace cache tree and points Triton at it via symlink — so there's only one copy of the (often multi-gigabyte) weight files:

```
$out/share/models/hub/models--<slug>/
  refs/main                          # text file containing the snapshot ID
  snapshots/<snapshotId>/            # the actual model files (single copy)

$out/share/models/<tritonModelName>/
  config.pbtxt                       # Triton config (same as single layout)
  model-defaults.json                # vLLM engine parameters
  weights -> ../hub/models--<slug>/snapshots/<snapshotId>   # symlink, not a copy
```

**How this works for each runtime:**

- **vLLM / SGLang:** Set the environment variable `HF_HUB_CACHE=$out/share/models/hub`. These runtimes use HuggingFace's library internally, which follows the `refs/main` → `snapshots/<hash>` chain to find model files — exactly as if the model had been downloaded normally.
- **Triton:** Set `--model-repository=$out/share/models`. Triton sees `<tritonModelName>/config.pbtxt` and follows the `weights/` symlink to reach the same files. It doesn't know or care about the HF cache structure.

Zero file duplication — both runtimes read from the same snapshot directory.

## The shared builder — `mkHfModel.nix`

All standard model packages are built by calling `mkHfModel`. Here's a real example (Qwen3.5-2B, dual layout):

```nix
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:
mkHfModel {
  pname = "vllm-qwen3.5-2b";
  baseVersion = "1.0.0";
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/vllm-qwen3-5-2b.json);
  srcPath = /mnt/scratch/models/inferencing/hub/models--Qwen--Qwen3.5-2B;
  tritonModelName = "qwen3_5_2b";

  # These two enable dual layout (omit both for single layout):
  slug = "Qwen--Qwen3.5-2B";
  snapshotId = "15852e8c16360a2fea060d615a32b45270f8a8fc";

  vllmDefaults = { gpu_memory_utilization = 0.85; max_model_len = 4096; dtype = "auto"; };
};
```

### Parameters

| Parameter | Required | Description |
|---|---|---|
| `pname` | yes | Nix package name (e.g. `"vllm-qwen3.5-2b"`) |
| `baseVersion` | yes | Semantic version (e.g. `"1.0.0"`) |
| `buildMeta` | yes | Parsed JSON from `build-meta/<name>.json` — provides `build_version` and `git_rev_short` |
| `srcPath` | yes | Absolute path to model weights on the local build machine |
| `tritonModelName` | yes | Directory name Triton will see under its model repository (e.g. `"qwen3_5_2b"`) |
| `vllmDefaults` | no | vLLM engine parameters written to `model-defaults.json` (memory budget, dtype, quantization, etc.) |
| `slug` | no | HuggingFace slug with `--` separator (e.g. `"Qwen--Qwen3.5-2B"`). Providing this enables the dual layout |
| `snapshotId` | no | HuggingFace snapshot commit hash. Required when `slug` is set |

**Layout selection:** If both `slug` and `snapshotId` are provided → dual layout. Otherwise → single layout.

## Current packages

| Package | Model | Layout | Quantization | GPU req | Notes |
|---|---|---|---|---|---|
| `phi-4-mini-instruct-fp8-hf` | Phi-4-mini-instruct | single | FP8 (PyTorch native) | SM89+ | Triton-only |
| `phi-4-mini-instruct-fp8-torchao` | Phi-4-mini-instruct | single | FP8 ([TorchAO](https://github.com/pytorch/ao)) | SM89+ | Triton-only |
| `phi-4-mini-instruct-fp8-sglang` | Phi-4-mini-instruct | HF-only | FP8 (TorchAO) | SM89+ | Custom build (see below) |
| `vllm-phi-3-5-mini-instruct-awq` | Phi-3.5-mini-instruct | dual | [AWQ](https://github.com/mit-han-lab/llm-awq) 4-bit | SM75+ | |
| `vllm-qwen3-5-2b` | Qwen3.5-2B | dual | none (full precision) | SM75+ | |
| `vllm-qwen3-5-4b-fp8` | Qwen3.5-4B | single | FP8 (TorchAO) | SM89+ | |
| `vllm-smollm3-3b-fp8` | SmolLM3-3B | single | FP8 (TorchAO) | SM89+ | |

**Quantization types in this repo:**
- **FP8 (W8A8)** — 8-bit float weights and activations. Two toolchains: PyTorch native and [TorchAO](https://github.com/pytorch/ao).
- **AWQ (INT4)** — 4-bit integer weights, FP16 activations. Preserves accuracy via salient-weight protection.
- **none (BF16/FP16)** — unquantized. `dtype = "auto"` resolves to BF16 on SM80+, FP16 on older GPUs.

### GPU compatibility

#### SM architecture reference

| SM | Generation | Representative GPUs |
|---|---|---|
| SM75 | Turing (2018) | T4, RTX 2070/2080 |
| SM80 | Ampere (2020) | A100 |
| SM86 | Ampere (2020) | A10, RTX 3090 |
| SM89 | Ada Lovelace (2022) | L4, L40, L40S, RTX 4090 |
| SM90 | Hopper (2022) | H100, H200 |
| SM100 | Blackwell (2024) | B200 |
| SM120 | Blackwell (2025) | RTX 5090, RTX 5080 |

#### Quantization × SM compatibility matrix

This covers the three methods used in this repo plus common alternatives you may encounter:

| Method | SM75 | SM80/86 | SM89 | SM90 | SM100/120 |
|---|---|---|---|---|---|
| FP8 W8A8 | W8A16 only (Marlin) | W8A16 only (Marlin) | native | native | native |
| AWQ (INT4) | supported | Marlin-optimized | Marlin-optimized | Marlin-optimized | Marlin-optimized |
| GPTQ (INT4) | supported | Marlin-optimized | Marlin-optimized | Marlin-optimized | Marlin-optimized |
| INT8 (W8A8) | supported | supported | supported | supported | supported |
| FP16 / BF16 | FP16 only | native (both) | native (both) | native (both) | native (both) |
| FP4 (NVFP4) | — | — | — | — | native |

**Legend:**
- **native** — hardware tensor-core support, best performance
- **Marlin-optimized** — uses [Marlin](https://github.com/IST-DASLab/marlin) kernels for high-throughput dequantization
- **W8A16 only (Marlin)** — weights stored in FP8, dequantized to FP16 for compute via Marlin; memory savings but no FP8 compute speedup
- **supported** — runs correctly, standard CUDA kernels (native AWQ GEMM on SM75, ExLlamaV2 for GPTQ on SM75)
- **—** — not available on this architecture

#### Choosing a quantization for your GPU

- **SM89+** (L4, L40, RTX 4090, H100): FP8 is the best default — native tensor-core support, half the memory of FP16.
- **SM80/86** (A100, A10, RTX 3090): AWQ or GPTQ for best memory efficiency. FP8 loads but runs as W8A16 via Marlin.
- **SM75** (T4, RTX 2070/2080): AWQ or GPTQ recommended. FP8 loads as W8A16 (memory savings, no FP8 compute speedup). Small unquantized models fit in FP16.

See the [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/index.html) for the full story.

### Special case: `phi-4-mini-instruct-fp8-sglang`

This package is a **custom derivation** that does not use `mkHfModel`. It produces only the HF cache layout (no Triton `config.pbtxt`) and patches `tokenizer_config.json` — replacing `"TokenizersBackend"` with `"PreTrainedTokenizerFast"` — for compatibility with SGLang 0.5.x (which ships `transformers` 4.57.x, before `TokenizersBackend` was added in 4.58). This package can be removed once SGLang ships `transformers >= 4.58`.

## Build metadata versioning

Each package has a corresponding JSON file in `build-meta/` that tracks its version independently of git:

```json
{
  "build_version": 2,
  "force_increment": 0,
  "git_rev": "ca88e1655077dab1a0e8bc3583d98bb2575be501",
  "git_rev_short": "ca88e16",
  "changelog": "Dual layout: Triton + vanilla vLLM via mkHfModel with slug/snapshotId."
}
```

- **`build_version`** increments independently of git commits. Bump it when re-packaging weights without changing `.nix` files (e.g., updated quantization of the same model).
- **Final version string:** `<baseVersion>+<git_rev_short>` (e.g. `1.0.0+ca88e16`). The `baseVersion` comes from the `.nix` file; the git rev is recorded at build time.
- A marker file is written to `$out/share/<pname>/flox-build-version-<build_version>` so you can identify which build produced a given store path.

## How runtimes consume packages

Each runtime needs just one environment variable or flag to find the packaged model:

| Runtime | Configuration | What it does |
|---|---|---|
| **vLLM** | `HF_HUB_CACHE=$out/share/models/hub` | vLLM's HuggingFace integration follows `refs/main` → `snapshots/<hash>` to locate model files |
| **Triton** | `--model-repository=$out/share/models` | Triton scans for `<tritonModelName>/config.pbtxt` and loads weights from the `weights/` subdirectory |
| **SGLang** | `HF_HUB_CACHE=$out/share/models/hub` | Same mechanism as vLLM — both use HuggingFace's cache resolution internally |

Dual-layout packages work for all three runtimes simultaneously from a single store path, with no file duplication.

## How to add a new model

1. **Download or quantize weights** to `/mnt/scratch/models/inferencing/`.
   - For dual layout: use the HF cache structure (`hub/models--<org>--<model>/` with `snapshots/` and optionally `blobs/`)
   - For single layout: use a flat directory (`resolved/<org>--<model>/`) with all symlinks resolved

2. **Create `build-meta/<name>.json`:**
   ```json
   {
     "build_version": 1,
     "force_increment": 0,
     "git_rev": "<current git rev>",
     "git_rev_short": "<short rev>",
     "changelog": "Initial package."
   }
   ```

3. **Create `.flox/pkgs/<name>.nix`** calling `mkHfModel` with appropriate parameters. Include `slug` + `snapshotId` for dual layout, omit both for single layout.

4. **Build:**
   ```bash
   flox build -A <name>
   ```

5. **Verify the output layout:**
   ```bash
   ls -la result-<name>/share/models/
   ```

6. **Publish:**
   ```bash
   flox publish
   ```

## Source model locations

Weights live on local scratch storage before packaging:

- **HF cache layout:** `/mnt/scratch/models/inferencing/hub/models--<org>--<model>/` — contains `blobs/` and `snapshots/` directories as created by `huggingface-cli download`. Used for dual-layout packages.
- **Pre-resolved:** `/mnt/scratch/models/inferencing/resolved/<org>--<model>/` — a flat directory where HF blob symlinks have been resolved to regular files. Used for single-layout packages.

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
