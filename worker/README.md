# GPU Worker

Distributed GPU Inference Worker - Share your idle GPU computing power for LLM inference and image generation.

## Features

- **Easy Setup**: Single command installation with automatic Python environment management
- **Multiple Engines**: Support for native Transformers, vLLM, SGLang backends
- **LLM Inference**: Run Qwen, Llama, GLM, DeepSeek and other popular models
- **Image Generation**: Support FLUX, Stable Diffusion XL and more
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Auto Configuration**: Interactive setup wizard

## Quick Start

### Using npx (Recommended)

```bash
# Interactive menu
npx gpu-worker

# Or step by step
npx gpu-worker configure   # Setup configuration
npx gpu-worker start       # Start worker
npx gpu-worker status      # Check status
```

### Using npm global install

```bash
npm install -g gpu-worker
gpu-worker configure
gpu-worker start
```

## Requirements

- **Node.js**: >= 16.0.0
- **Python**: >= 3.9
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional, for GPU inference)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for model storage

## Configuration

The worker can be configured via:

1. **Interactive wizard**: `gpu-worker configure`
2. **Environment variables**: Copy `.env.example` to `.env`
3. **YAML config file**: Edit `config.yaml`

### Key Configuration Options

| Option | Environment Variable | Description |
|--------|---------------------|-------------|
| Server URL | `GPU_SERVER_URL` | Central server address |
| Worker Name | `GPU_WORKER_NAME` | Display name for this worker |
| Region | `GPU_REGION` | Geographic region (e.g., asia-east) |
| Supported Types | `GPU_SUPPORTED_TYPES` | Task types: llm, image_gen |
| LLM Model | `GPU_LLM_MODEL` | HuggingFace model ID |

## Supported Models

### LLM Models

| Model | VRAM Required | Model ID |
|-------|---------------|----------|
| Qwen2.5-7B | 16GB | `Qwen/Qwen2.5-7B-Instruct` |
| Llama-3.1-8B | 18GB | `meta-llama/Llama-3.1-8B-Instruct` |
| GLM-4-9B | 20GB | `THUDM/glm-4-9b-chat` |

### Image Generation Models

| Model | VRAM Required | Model ID |
|-------|---------------|----------|
| FLUX.1-schnell | 24GB | `black-forest-labs/FLUX.1-schnell` |
| SDXL | 12GB | `stabilityai/stable-diffusion-xl-base-1.0` |

## High-Performance Backends

For production use, install optional high-performance backends:

```bash
# SGLang (recommended for high throughput)
pip install sglang[all]

# vLLM (alternative)
pip install vllm
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Central Server │◄────│   GPU Worker    │
│   (Scheduler)   │     │  (This Package) │
└─────────────────┘     └─────────────────┘
        │                       │
        │                       ▼
        │               ┌───────────────┐
        │               │  GPU/CPU      │
        │               │  Inference    │
        └───────────────┤  Engine       │
                        └───────────────┘
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/Baozhi888/distributed-gpu-inference)
- [Documentation](https://github.com/Baozhi888/distributed-gpu-inference#readme)
- [Issue Tracker](https://github.com/Baozhi888/distributed-gpu-inference/issues)
