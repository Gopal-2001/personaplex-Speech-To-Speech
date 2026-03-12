# PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models

[![Weights](https://img.shields.io/badge/Weights-HuggingFace-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2602.06053)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE-MIT)

PersonaPlex is a real-time, full-duplex speech-to-speech conversational model that enables persona control through text-based role prompts and audio-based voice conditioning. Trained on a combination of synthetic and real conversations, it produces natural, low-latency spoken interactions with a consistent persona. PersonaPlex is based on the [Moshi](https://github.com/kyutai-labs/moshi) architecture and weights.

> ⚠️ **This is a Windows-native setup guide.** For Linux/Docker setup, refer to the [original repository](https://github.com/nvidia/personaplex).

---

## 📋 Prerequisites

- Windows 10 or Windows 11
- NVIDIA GPU with CUDA support (RTX 3090 recommended, 24GB VRAM)
- [Python 3.11](https://www.python.org/downloads/) installed
- [CUDA 12.4 drivers](https://developer.nvidia.com/cuda-downloads) installed
- [ngrok](https://ngrok.com/download) (optional, for remote access)
- Git

---

## 🚀 Installation

### Step 1: Clone the Repository

```powershell
git clone https://github.com/Gopal-2001/personaplex-Speech-To-Speech.git
cd personaplex-Speech-To-Speech
```

### Step 2: Create a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install PyTorch with CUDA 12.4

> ⚠️ **Important:** Install PyTorch with CUDA **before** installing other packages. Do NOT use `pip install moshi` from PyPI as it will install an incompatible version.

```powershell
pip install torch==2.4.1+cu124 torchaudio==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Compatible Dependencies

```powershell
pip install "aiohttp>=3.10.5,<3.11" "einops==0.7.0" "huggingface-hub>=0.24,<0.25" "numpy>=1.26,<2.2" "safetensors>=0.4.0,<0.5" "sentencepiece==0.2.0" "sphn>=0.1.4,<0.2"
```

### Step 5: Install the Local Moshi-PersonaPlex Package

```powershell
pip install -e moshi/
```

### Step 6: Accept Model License & Set HuggingFace Token

Log in to your [HuggingFace account](https://huggingface.co) and accept the [PersonaPlex model license](https://huggingface.co/nvidia/personaplex-7b-v1).

Then set your token in PowerShell:

```powershell
$env:HF_TOKEN = "<YOUR_HUGGINGFACE_TOKEN>"
```

---

## ▶️ Launch Server

```powershell
$env:TORCHDYNAMO_DISABLE = "1"
python -m moshi.server
```

The server will automatically download model weights on first run (~several GB). This may take a few minutes depending on your internet connection.

Once loaded, access the Web UI at:

```
http://localhost:8998
```

Or look for the LAN IP printed in the terminal:

```
[INFO] Access the Web UI directly at http://192.168.x.x:8998
```

> 💡 **Why `TORCHDYNAMO_DISABLE=1`?**  
> On Windows, `torch.compile` requires `triton` which is not natively supported. This flag disables compilation and reduces latency from ~27ms to ~12ms on RTX 3090.

---

## 🌐 Remote Access via ngrok

To share the server over the internet:

```powershell
ngrok http 8998 --region=in --host-header="localhost:8998"
```

ngrok will provide a public HTTPS URL like:

```
https://xxxx-xxxx.ngrok-free.dev -> http://localhost:8998
```

> **Note:** ngrok free tier URL changes every session.

---

## 💻 CPU Offload (Low VRAM GPUs)

If your GPU has insufficient memory, use the `--cpu-offload` flag (requires `accelerate`):

```powershell
pip install accelerate
$env:TORCHDYNAMO_DISABLE = "1"
python -m moshi.server --cpu-offload
```

---

## 🎤 Voices

PersonaPlex supports a wide range of voices with pre-packaged embeddings:

| Type | Voices |
|------|--------|
| Natural Female | NATF0, NATF1, NATF2, NATF3 |
| Natural Male | NATM0, NATM1, NATM2, NATM3 |
| Variety Female | VARF0, VARF1, VARF2, VARF3, VARF4 |
| Variety Male | VARM0, VARM1, VARM2, VARM3, VARM4 |

---

## 💬 Prompting Guide

### Assistant Role
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

### Customer Service Roles
```
You work for CitySan Services which is a waste management and your name is Ayelen Lucero.
Information: Verify customer name Omar Torres. Current schedule: every other week.
Upcoming pickup: April 12th. Compost bin service available for $8/month add-on.
```

### Casual Conversations
```
You enjoy having a good conversation. Have a reflective conversation about career changes
and feeling of home. You have lived in California for 21 years and consider San Francisco
your home. You work as a teacher and have traveled a lot. You dislike meetings.
```

### Fun / Generalization Example (Astronaut Prompt)
```
You enjoy having a good conversation. Have a technical discussion about fixing a reactor
core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex.
You are already dealing with a reactor core meltdown on a Mars mission.
```

---

## 🔧 Troubleshooting (Windows)

| Error | Fix |
|-------|-----|
| `IndentationError` in `compile.py` | Run `pip uninstall moshi -y` then `pip install -e moshi/` |
| `AssertionError: Torch not compiled with CUDA` | Reinstall: `pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124` |
| `BackendCompilerFailed: triton_key` | Set `$env:TORCHDYNAMO_DISABLE = "1"` before starting |
| Dependency conflict warnings | Non-critical if torch CUDA version is correct; server will still run |
| AI not listening / talking randomly | Allow microphone in Chrome: `chrome://settings/content/microphone` |
| `pip uninstall moshi: WARNING Skipping` | Use `pip install -e moshi/` instead of `pip install moshi` |
| High latency (>20ms) | Ensure `TORCHDYNAMO_DISABLE=1` is set; verify GPU with `python -c "import torch; print(torch.cuda.is_available())"` |

---

## 📁 Project Structure

```
personaplex-Speech-To-Speech/
├── moshi/                  # Core AI server package
│   └── moshi/
│       ├── server.py       # Main WebSocket server
│       ├── models/         # Model loaders and inference
│       ├── modules/        # Transformer, SEANet, Gating
│       └── utils/          # Compile, logging, connection utils
├── client/                 # Frontend client code
├── assets/                 # Static assets and test files
├── Dockerfile              # Docker support (Linux)
├── docker-compose.yaml     # Docker Compose (Linux)
└── README.md
```

---

## 📊 Performance (Windows, RTX 3090)

| Metric | Value |
|--------|-------|
| GPU | NVIDIA GeForce RTX 3090 (24GB) |
| Latency | ~12-14ms |
| Model Load Time | ~60 seconds |
| Framework | PyTorch 2.4.1+cu124 |
| CUDA Version | 12.4 |

---

## 📄 Offline Evaluation

For offline evaluation, stream an input WAV file and produce an output WAV file:

```powershell
# Assistant example
$env:HF_TOKEN = "<TOKEN>"
python -m moshi.offline `
  --voice-prompt "NATF2.pt" `
  --input-wav "assets/test/input_assistant.wav" `
  --seed 42424242 `
  --output-wav "output.wav" `
  --output-text "output.json"
```

```powershell
# Service example
$env:HF_TOKEN = "<TOKEN>"
python -m moshi.offline `
  --voice-prompt "NATM1.pt" `
  --text-prompt (Get-Content "assets/test/prompt_service.txt" -Raw) `
  --input-wav "assets/test/input_service.wav" `
  --seed 42424242 `
  --output-wav "output.wav" `
  --output-text "output.json"
```

---

## 📜 License

The present code is provided under the [MIT license](LICENSE-MIT). The weights for the models are released under the [NVIDIA Open Model license](https://huggingface.co/nvidia/personaplex-7b-v1).

---

## 📚 Citation

```bibtex
@misc{roy2026personaplexvoicerolecontrol,
      title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models}, 
      author={Rajarshi Roy and Jonathan Raiman and Sang-gil Lee and Teodor-Dumitru Ene and Robert Kirby and Sungwon Kim and Jaehyeon Kim and Bryan Catanzaro},
      year={2026},
      eprint={2602.06053},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06053}, 
}
```