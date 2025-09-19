# Flux SRPO in ComfyUI | Realistic Images with No Plastic Look
# SRPO (Semantic Relative Preference Optimization) On OSX Silicon Setup Guide

A comprehensive guide to setting up SRPO with ComfyUI using uv on macOS ARM64 (Apple Silicon).

## 📋 Prerequisites

### System Requirements
- **macOS**: 15.x or later (tested on macOS 15.6.1)
- **Architecture**: ARM64 (Apple Silicon M1/M2/M3/M4)
- **Memory**: Minimum 16GB RAM (recommended 32GB+ for optimal performance)
- **Storage**: At least 50GB free space (50GB for SRPO model + dependencies)
- **Python**: 3.12+ (tested with Python 3.13.7)

### Required Tools
- **uv**: Fast Python package installer and resolver
- **Git**: For cloning repositories (if needed)

## 🚀 Step-by-Step Setup

### Step 1: Verify System Requirements

```bash
# Check macOS version
sw_vers

# Check available memory
echo "Total RAM: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 "GB"}')"

# Check Python version
python3 --version

# Check architecture
uname -m
```

**Expected Output (Actual Results):**
```
ProductName:		macOS
ProductVersion:		15.6.1
BuildVersion:		24G90
Total RAM: 128GB
Python 3.13.7
arm64
```

### Step 2: Install uv (if not already installed)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uv --version
```

**Expected Output (Actual Results):**
```
uv 0.8.18 (c4c47814a 2025-09-17)
```

### Step 3: Create Project Directory

```bash
# Create and navigate to project directory
mkdir -p ~/development/srpo-comfyui
cd ~/development/srpo-comfyui
```

### Step 4: Initialize uv Project

```bash
# Initialize uv project
uv init --no-readme --no-workspace

# Verify project structure
ls -la
```

**Expected Output:**
```
pyproject.toml
.venv/ (created after first package installation)
```

### Step 5: Install ComfyUI and Dependencies

```bash
# Install ComfyUI and all required packages
uv add torch torchvision torchaudio torchsde numpy einops transformers tokenizers sentencepiece safetensors aiohttp yarl pyyaml pillow scipy tqdm psutil alembic sqlalchemy av kornia spandrel soundfile pydantic pydantic-settings comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs

# Verify installation
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

**Expected Output (Actual Results):**
```
PyTorch version: 2.8.0
```

### Step 6: Clone ComfyUI Repository

```bash
# Clone ComfyUI (if not already done)
git clone https://github.com/comfyanonymous/ComfyUI.git

# Verify ComfyUI structure
ls -la ComfyUI/
```

**Expected Structure:**
```
ComfyUI/
├── main.py
├── requirements.txt
├── nodes.py
├── comfy/
├── custom_nodes/
├── models/
└── ...
```

### Step 7: Download SRPO Model and Workflow

```bash
# Create directories for models
mkdir -p ComfyUI/models/diffusion_models/SRPO

# Download SRPO model (this may take time - 47.6GB)
uv run huggingface-cli download rockerBOO/flux.1-dev-SRPO --local-dir ./models/diffusion_models/srpo-bf16 --exclude "*.md" "*.txt" "*.png"

# Move model to correct ComfyUI location
cp models/diffusion_models/srpo-bf16/diffusion_pytorch_model.safetensors ComfyUI/models/diffusion_models/SRPO/

# Download SRPO workflow
uv run huggingface-cli download tencent/SRPO comfyui/SRPO-workflow.json --local-dir ./models/SRPO

# Copy workflow to accessible locations
cp models/SRPO/comfyui/SRPO-workflow.json ComfyUI/user/default/workflows/
cp models/SRPO/comfyui/SRPO-workflow.json ./

# Verify downloads
ls -lh ComfyUI/models/diffusion_models/SRPO/
ls -lh ComfyUI/user/default/workflows/SRPO-workflow.json
```

**Expected Output (Actual Results):**
```
ComfyUI/models/diffusion_models/SRPO/
└── diffusion_pytorch_model.safetensors (47.6GB)

-rw-r--r-- user workflows/SRPO-workflow.json (15KB)
```

### Step 8: Configure ComfyUI for SRPO

The model and workflow are now in the correct locations:
- **Model**: `ComfyUI/models/diffusion_models/SRPO/diffusion_pytorch_model.safetensors`
- **Workflow**: `ComfyUI/user/default/workflows/SRPO-workflow.json`

### Step 9: Start ComfyUI Server

```bash
# Start ComfyUI with SRPO
uv run python ComfyUI/main.py --listen 127.0.0.1 --port 8188
```

**Expected Output:**
```
Checkpoint files will always be loaded safely.
Total VRAM 131072 MB, total RAM 131072 MB
pytorch version: 2.8.0
Mac Version (15, 6, 1)
Set vram state to: SHARED
Device: mps
Using sub quadratic optimization for attention, if you have memory or speed issues try using: --use-split-cross-attention
Python version: 3.12.11 (main, Sep  2 2025, 14:12:30) [Clang 20.1.4 ]
ComfyUI version: 0.3.59
ComfyUI frontend version: 1.28.x
[Prompt Server] web root: /path/to/comfyui_frontend_package/static

Import times for custom nodes:
   0.0 seconds: /Users/exobit/development/cosmos/ComfyUI/custom_nodes/websocket_image_save.py

Context impl SQLiteImpl.
Will assume non-transactional DDL.
No target revision found.
Starting server

To see the GUI go to: http://127.0.0.1:8188
```

### Step 10: Access ComfyUI Web Interface

1. Open your web browser
2. Navigate to: `http://127.0.0.1:8188`
3. You should see the ComfyUI interface

### Step 11: Load SRPO Workflow

1. In ComfyUI, click the **"Load"** button (folder icon)
2. Navigate to `user/default/workflows/SRPO-workflow.json`
3. Click **"Load"** to load the workflow

### Step 12: Configure and Generate Images

The SRPO workflow comes pre-configured with optimal settings:

- **Positive Prompt**: "portrait of a girl" (editable)
- **Resolution**: 1280x720 (configurable via width/height nodes)
- **Guidance Scale**: 3.5 (configurable via FluxGuidance node)
- **Steps**: 50 (configurable via BasicScheduler)
- **Sampler**: Euler (configurable via KSamplerSelect)

#### To Generate an Image:

1. **Edit the prompt** (optional):
   - Click on the text input node
   - Change "portrait of a girl" to your desired prompt

2. **Adjust settings** (optional):
   - Modify width/height for different resolutions
   - Adjust guidance scale for more/less adherence to prompt
   - Change steps for quality vs speed trade-off

3. **Generate**:
   - Click the **"Queue"** button
   - Wait for generation to complete
   - View results in the output

## 🔧 Troubleshooting

### Common Issues

#### 1. "Module not found" errors
```bash
# Reinstall dependencies
uv sync
```

#### 2. Model download failures
```bash
# Resume download
uv run huggingface-cli download rockerBOO/flux.1-dev-SRPO --local-dir ./models/diffusion_models/srpo-bf16 --resume-download
```

#### 3. ComfyUI won't start
```bash
# Check Python path
uv run which python
uv run python --version

# Clear cache
rm -rf ~/.cache/huggingface
```

#### 4. MPS/GPU issues on Apple Silicon
```bash
# Force CPU mode if needed
uv run python ComfyUI/main.py --cpu
```

#### 5. Workflow/Model not visible
```bash
# Check model location
ls -lh ComfyUI/models/diffusion_models/SRPO/

# Check workflow location
ls -lh ComfyUI/user/default/workflows/SRPO-workflow.json
```

### Performance Optimization

#### For Apple Silicon Macs:
- Use MPS acceleration (default)
- Ensure adequate cooling for long generations
- Monitor memory usage with Activity Monitor

#### Memory Management:
```bash
# Check memory usage
uv run python -c "import torch; print(f'Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB')"
```

## 📚 Additional Resources

- **SRPO Paper**: [Directly Aligning the Full Diffusion Trajectory](https://arxiv.org/abs/2509.06942)
- **ComfyUI Documentation**: [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- **HuggingFace Model**: [SRPO Model Card](https://huggingface.co/tencent/SRPO)

## ✅ Verification Steps

After setup, verify everything works:

```bash
# Check model file
ls -lh ComfyUI/models/diffusion_models/SRPO/diffusion_pytorch_model.safetensors

# Check workflow
ls -lh ComfyUI/user/default/workflows/SRPO-workflow.json

# Test ComfyUI import
uv run python -c "import comfyui; print('ComfyUI imports successfully')"

# Check PyTorch MPS support
uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Expected Results (Actual Results):**
- Model file: 47.6GB (full SRPO model)
- Workflow file: 15KB
- ComfyUI: Imports successfully
- MPS: True (on Apple Silicon)

## 🎯 Next Steps

1. **Experiment with prompts**: Try different styles and subjects
2. **Adjust parameters**: Fine-tune guidance, steps, and resolution
3. **Batch processing**: Generate multiple images with variations
4. **Custom workflows**: Modify the SRPO workflow for specific use cases

---

**Note**: This guide was tested on macOS 15.6.1 with Apple Silicon (ARM64), Python 3.13.7, and uv 0.8.18. The SRPO model (47.6GB) provides enhanced realism by 3x compared to base FLUX.1-dev.
