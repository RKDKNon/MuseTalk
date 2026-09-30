# 🎭 Lipsync (MuseTalk)

Audio-driven talking head — ใส่เสียง + รูปภาพ/วิดีโอ → ได้วิดีโอ lip-sync หน้าพูดตามเสียง

**Based on:** [MuseTalk](https://github.com/TMElyralab/MuseTalk) by Tencent Lyra Lab

**Features:**
- 🎬 สร้างวิดีโอ lip-sync จากรูปภาพ/วิดีโอ + ไฟล์เสียง
- 🔄 รองรับ 2 เวอร์ชัน: v1.0 (เดิม) และ v1.5 (ปรับปรุง)
- ⚡ Real-time inference — สร้าง avatar แล้วเปลี่ยนเสียงได้เรื่อยๆ
- 🖥️ Gradio Web UI — ลาก drag & drop ง่าย
- 🎯 ปรับ bbox_shift, cheek width, parsing mode ได้ละเอียด
- 🏋️ Training pipeline สำหรับ fine-tune model

---

## 📦 ติดตั้ง

```bash
cd C:\Users\PC\Documents\liveAi\lipsync
uv sync
```

**ต้องมี:**
- Python 3.10 (เท่านั้น ไม่รองรับ 3.11+)
- NVIDIA GPU + CUDA (cu128)
- ffmpeg (ต้องอยู่ใน PATH)
- uv (package manager)

### ติดตั้ง Dependencies เพิ่มเติม (หลัง uv sync)

> ⚠️ **สำคัญ:** มี dependencies ที่ไม่ได้อยู่ใน pyproject.toml ต้องติดตั้งเพิ่มด้วย pip

```bash
cd C:\Users\PC\Documents\liveAi\lipsync

# 1. ติดตั้ง Gradio (Web UI)
.venv\Scripts\python.exe -m pip install gradio -q

# 2. ติดตั้ง OpenMMLab (mmpose ต้องการ)
.venv\Scripts\python.exe -m pip install mmengine -q
.venv\Scripts\python.exe -m pip install mmcv-lite==2.1.0 -q
.venv\Scripts\python.exe -m pip install chumpy --no-build-isolation -q
.venv\Scripts\python.exe -m pip install mmdet mmpose -q

# 3. Fix huggingface-hub version (ถ้ามีปัญหา)
.venv\Scripts\python.exe -m pip install "huggingface-hub>=1.5.0,<2.0" -q
```

### ติดตั้ง ffmpeg (ถ้ายังไม่มี)

```bash
# ดาวน์โหลด ffmpeg สำหรับ Windows
# https://www.gyan.dev/ffmpeg/builds/
# แตกไฟล์แล้วเพิ่ม bin/ เข้า PATH

# ทดสอบ
ffmpeg -version
```

### ดาวน์โหลด Model Weights (~8.8 GB)

```bash
cd C:\Users\PC\Documents\liveAi\lipsync

# ดาวน์โหลดทีละตัวด้วย hf CLI
.venv\Scripts\hf.exe download TMElyralab/MuseTalk --local-dir models
.venv\Scripts\hf.exe download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.bin --local-dir models/sd-vae
.venv\Scripts\hf.exe download openai/whisper-tiny config.json pytorch_model.bin preprocessor_config.json --local-dir models/whisper
.venv\Scripts\hf.exe download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir models/dwpose
.venv\Scripts\hf.exe download ByteDance/LatentSync latentsync_syncnet.pt --local-dir models/syncnet
.venv\Scripts\hf.exe download ManyOtherFunctions/face-parse-bisent 79999_iter.pth resnet18-5c106cde.pth --local-dir models/face-parse-bisent
```

### Model Weights ที่ต้องมี

| Model | ไฟล์ | ขนาด |
|-------|------|------|
| MuseTalk v1 | `models/musetalk/pytorch_model.bin` | 3,242 MB |
| MuseTalk v1.5 | `models/musetalkV15/unet.pth` | 3,242 MB |
| SD VAE | `models/sd-vae/config.json` + `diffusion_pytorch_model.bin` | 319 MB |
| Whisper | `models/whisper/pytorch_model.bin` | 144 MB |
| DWPose | `models/dwpose/dw-ll_ucoco_384.pth` | 388 MB |
| SyncNet | `models/syncnet/latentsync_syncnet.pt` | 1,419 MB |
| Face Parse | `models/face-parse-bisent/79999_iter.pth` + `resnet18` | 95 MB |

---

## 🚀 วิธีใช้งาน

> ⚠️ **สำคัญ:** ต้อง `cd` เข้า folder `lipsync` ก่อนรันทุกคำสั่ง!
> ```bash
> cd C:\Users\PC\Documents\liveAi\lipsync
> ```

### มี 3 โหมดการใช้งาน:

| โหมด | ใช้ทำอะไร | คำสั่ง |
|------|-----------|--------|
| **Gradio Web UI** | ลาก drag & drop ง่าย มี debug | `uv run python app.py` |
| **Normal Inference** | สร้างวิดีโอจาก config file | `uv run python -m scripts.inference` |
| **Real-time Inference** | สร้าง avatar แล้วเปลี่ยนเสียงได้ | `uv run python -m scripts.realtime_inference` |

---

## 1️⃣ Gradio Web UI (แนะนำสำหรับเริ่มต้น)

```bash
cd C:\Users\PC\Documents\liveAi\lipsync

# รันแบบปกติ
uv run python app.py

# รันแบบ float16 (เร็วกว่า ใช้ VRAM น้อยกว่า)
uv run python app.py --use_float16

# เปิดให้เข้าจากเครื่องอื่น
uv run python app.py --share
```

เปิด browser ไปที่ **http://127.0.0.1:7860**

### วิธีใช้ Web UI:
1. **อัพโหลด Reference Video** — วิดีโอหรือรูปภาพใบหน้า
2. **อัพโหลด Driving Audio** — ไฟล์เสียง WAV
3. **กด "1. Test Inpainting"** — ทดสอบพารามิเตอร์กับเฟรมแรก
4. **ปรับค่า** ตามต้องการ (ดูส่วนพารามิเตอร์ด้านล่าง)
5. **กด "2. Generate"** — สร้างวิดีโอ lip-sync

### ตัวเลือก Web UI:

| Flag | ค่า Default | คำอธิบาย |
|------|-------------|----------|
| `--ip` | 127.0.0.1 | IP address |
| `--port` | 7860 | Port number |
| `--share` | false | สร้าง public link |
| `--use_float16` | false | ใช้ float16 (เร็วกว่า) |
| `--ffmpeg_path` | ffmpeg-master... | Path ของ ffmpeg |

---

## 2️⃣ Normal Inference (Batch Mode)

สร้างวิดีโอ lip-sync จาก config file

### ขั้นตอน:

#### 1. สร้างไฟล์ config (YAML)

สร้างไฟล์ เช่น `configs/inference/my_task.yaml`:

```yaml
task_0:
  video_path: "data/video/sun.mp4"     # วิดีโอ/รูปภาพ ที่มีหน้าคน
  audio_path: "data/audio/sun.wav"     # ไฟล์เสียงที่จะ lip-sync

task_1:
  video_path: "data/video/yongen.mp4"
  audio_path: "data/audio/eng.wav"
  bbox_shift: -7                        # ปรับตำแหน่ง bbox (v1 เท่านั้น)
```

#### 2. รัน inference

```bash
# ใช้ v1.5 (แนะนำ)
uv run python -m scripts.inference ^
  --inference_config configs/inference/test.yaml ^
  --result_dir ./results/test ^
  --unet_model_path ./models/musetalkV15/unet.pth ^
  --unet_config ./models/musetalkV15/musetalk.json ^
  --version v15 ^
  --use_float16

# ใช้ v1.0
uv run python -m scripts.inference ^
  --inference_config configs/inference/test.yaml ^
  --result_dir ./results/test ^
  --unet_model_path ./models/musetalk/pytorch_model.bin ^
  --unet_config ./models/musetalk/musetalk.json ^
  --version v1 ^
  --use_float16
```

### ตัวเลือก Normal Inference:

| Flag | ค่า Default | คำอธิบาย |
|------|-------------|----------|
| `--inference_config` | test.yaml | ไฟล์ config กำหนด video/audio pairs |
| `--result_dir` | ./results | โฟลเดอร์เก็บผลลัพธ์ |
| `--unet_model_path` | musetalkV15/unet.pth | Path ของ UNet model |
| `--unet_config` | musetalk.json | Path ของ UNet config |
| `--whisper_dir` | ./models/whisper | โฟลเดอร์ Whisper model |
| `--vae_type` | sd-vae | ประเภท VAE model |
| `--version` | v15 | เวอร์ชัน model: `v1` หรือ `v15` |
| `--gpu_id` | 0 | GPU ID |
| `--fps` | 25 | FPS ของวิดีโอ |
| `--batch_size` | 8 | Batch size |
| `--bbox_shift` | 0 | ขยับ bounding box (px) — v1 เท่านั้น |
| `--extra_margin` | 10 | ขอบเพิ่มสำหรับ face crop — v1.5 |
| `--parsing_mode` | jaw | โหมด face blending: `jaw` หรือ `raw` |
| `--left_cheek_width` | 90 | ความกว้างแก้มซ้าย |
| `--right_cheek_width` | 90 | ความกว้างแก้มขวา |
| `--use_float16` | false | ใช้ float16 (เร็วกว่า) |
| `--use_saved_coord` | false | ใช้ coordinates ที่บันทึกไว้ |
| `--output_vid_name` | auto | ชื่อไฟล์ output |

---

## 3️⃣ Real-time Inference

สร้าง avatar จากวิดีโอ แล้วเปลี่ยนเสียงได้หลายไฟล์โดยไม่ต้อง preprocess ซ้ำ

### ขั้นตอน:

#### 1. สร้างไฟล์ config

สร้าง `configs/inference/realtime.yaml`:

```yaml
avator_1:
  preparation: True        # True = สร้าง avatar ใหม่, False = ใช้ที่มีอยู่
  bbox_shift: 5             # ปรับตำแหน่ง (v1 เท่านั้น, v1.5 ใช้ 0)
  video_path: "data/video/yongen.mp4"
  audio_clips:
    audio_0: "data/audio/yongen.wav"
    audio_1: "data/audio/eng.wav"
```

#### 2. รัน

```bash
# v1.5 (แนะนำ)
uv run python -m scripts.realtime_inference ^
  --inference_config configs/inference/realtime.yaml ^
  --result_dir ./results/realtime ^
  --unet_model_path ./models/musetalkV15/unet.pth ^
  --unet_config ./models/musetalkV15/musetalk.json ^
  --version v15 ^
  --fps 25

# v1.0
uv run python -m scripts.realtime_inference ^
  --inference_config configs/inference/realtime.yaml ^
  --result_dir ./results/realtime ^
  --unet_model_path ./models/musetalk/pytorch_model.bin ^
  --unet_config ./models/musetalk/musetalk.json ^
  --version v1 ^
  --fps 25
```

### ข้อดีของ Real-time Mode:
- **Preparation = True** ครั้งแรก: preprocess วิดีโอ (ช้า)
- **Preparation = False** ครั้งต่อไป: ข้ามการ preprocess (เร็วมาก)
- เปลี่ยนไฟล์เสียงได้หลายไฟล์ในครั้งเดียว

---

## 🎛️ พารามิเตอร์สำคัญ

### v1.0 vs v1.5

| Feature | v1.0 | v1.5 |
|---------|------|------|
| Model | musetalk/pytorch_model.bin (3.2GB) | musetalkV15/unet.pth (3.2GB) |
| bbox_shift | ปรับเองได้ | ใช้ 0 เสมอ |
| extra_margin | ไม่มี | ปรับได้ (default=10) |
| parsing_mode | ไม่มี | jaw / raw |
| cheek_width | ไม่มี | ปรับซ้าย/ขวาได้ |
| คุณภาพ | ดี | **ดีกว่า** (แนะนำ) |

### bbox_shift (v1.0 เท่านั้น)
- ค่าบวก (+) = ขยับ bbox ลง (เห็นคางมากขึ้น)
- ค่าลบ (-) = ขยับ bbox ขึ้น (เห็นคางน้อยลง)
- ปกติใช้ค่า -7 ถึง +7

### extra_margin (v1.5)
- เพิ่มขอบรอบใบหน้า
- ค่ามาก = ขยับขากรรไกรได้กว้างขึ้น
- Default: 10, ช่วง: 0-40

### parsing_mode (v1.5)
- `jaw` — ใช้ jaw parsing (แนะนำ ผลลัพธ์เนียนกว่า)
- `raw` — ใช้ raw face mask

### cheek_width (v1.5)
- `left_cheek_width` / `right_cheek_width`
- กำหนดขอบเขตแก้มที่จะถูกแก้ไข
- Default: 90, ช่วง: 20-160

---

## 📁 โครงสร้างไฟล์

```
lipsync/
├── app.py                              # Gradio Web UI
├── train.py                            # Training pipeline
├── scripts/
│   ├── inference.py                    # Normal batch inference
│   ├── realtime_inference.py           # Real-time inference (avatar)
│   └── preprocess.py                   # Data preprocessing
├── configs/
│   ├── inference/
│   │   ├── test.yaml                   # ตัวอย่าง normal inference config
│   │   └── realtime.yaml               # ตัวอย่าง realtime config
│   └── training/                       # Training configs
├── musetalk/                           # Core MuseTalk library
│   ├── models/                         # Model definitions
│   ├── utils/                          # Utilities (blending, face parsing, etc.)
│   ├── whisper/                        # Whisper integration
│   ├── data/                           # Data loading
│   └── loss/                           # Loss functions
├── models/                             # Downloaded model weights
│   ├── musetalk/                       # v1.0 weights (~3.2GB)
│   ├── musetalkV15/                    # v1.5 weights (~3.2GB)
│   ├── sd-vae/                         # Stable Diffusion VAE (~319MB)
│   ├── whisper/                        # Whisper tiny (~144MB)
│   ├── dwpose/                         # DWPose (~388MB)
│   ├── syncnet/                        # SyncNet (~1.4GB)
│   └── face-parse-bisent/              # Face parsing (~95MB)
├── data/                               # ข้อมูลตัวอย่าง
│   ├── video/                          # sun.mp4, yongen.mp4
│   └── audio/                          # eng.wav, sun.wav, yongen.wav
├── results/                            # (auto-created) ผลลัพธ์
├── download_weights.bat                # Script ดาวน์โหลด weights (Windows)
├── download_weights.sh                 # Script ดาวน์โหลด weights (Linux)
├── inference.sh                        # Script รัน inference (Linux)
├── pyproject.toml
└── README.md
```

---

## 🧪 ทดสอบด่วน

### ทดสอบด้วย Gradio UI

```bash
cd C:\Users\PC\Documents\liveAi\lipsync
uv run python app.py --use_float16
# เปิด http://127.0.0.1:7860
# อัพโหลด data/video/sun.mp4 + data/audio/sun.wav → Generate
```

### ทดสอบด้วย CLI

```bash
cd C:\Users\PC\Documents\liveAi\lipsync

# v1.5 normal inference กับข้อมูลตัวอย่าง
uv run python -m scripts.inference ^
  --inference_config configs/inference/test.yaml ^
  --result_dir ./results/test ^
  --unet_model_path ./models/musetalkV15/unet.pth ^
  --unet_config ./models/musetalkV15/musetalk.json ^
  --version v15 ^
  --use_float16
```

ผลลัพธ์จะอยู่ที่ `./results/test/v15/`

---

## 🔗 เชื่อมต่อกับ OmniVoice TTS

ใช้ omnivoice-tts สร้างเสียงภาษาไทย → ใช้ lipsync สร้างวิดีโอหน้าพูด:

```bash
# 1. สร้างเสียง
cd C:\Users\PC\Documents\liveAi\omnivoice-tts
uv run omnivoice-rt --text "สวัสดีครับ" --instruct "male, low pitch" --output ../lipsync/data/audio/thai_tts.wav

# 2. สร้างวิดีโอ lip-sync
cd C:\Users\PC\Documents\liveAi\lipsync
# แก้ config ให้ชี้ไปที่ thai_tts.wav แล้วรัน inference
# หรือใช้ Gradio UI อัพโหลดไฟล์
```

---

## 🔧 Tips

- **ใช้ v1.5** — คุณภาพดีกว่า v1.0 มาก
- **ใช้ `--use_float16`** — เร็วกว่าและใช้ VRAM น้อยกว่า
- **ทดสอบ Inpainting ก่อน** — กด "Test Inpainting" ใน Gradio เพื่อดูผลก่อน Generate
- **ปรับ extra_margin** — ถ้าขากรรไกรขยับไม่พอ ให้เพิ่มค่า
- **ปรับ cheek_width** — ถ้าแก้มดูผิดปกติ ให้ลดค่า
- **วิดีโอควรเป็น 25 fps** — Web UI จะแปลงให้อัตโนมัติ
- **ต้องมี ffmpeg** — ทุกโหมดต้องใช้ ffmpeg ในการรวมเสียง+วิดีโอ
- **Realtime mode ครั้งแรกช้า** — ตั้ง `preparation: True` ครั้งแรก, ครั้งต่อไปตั้ง `False`
- **VRAM** — ใช้ประมาณ 4-6 GB (float16)

---

## ⚠️ Compatibility Patches (app.py)

`app.py` มี patches ที่จำเป็นสำหรับ PyTorch 2.6+ และ mmcv-lite:

### 1. mmcv-lite Stub
เนื่องจากไม่มี CUDA Toolkit ติดตั้ง จึงใช้ `mmcv-lite` แทน `mmcv` (full)
`app.py` มี stub ที่ mock `mmcv._ext` module ให้ mmpose import ได้โดยไม่ต้อง compile CUDA ops

> MuseTalk ใช้แค่ DWPose ไม่ต้องการ CUDA ops ของ mmcv จริงๆ

### 2. torch.load Patch
PyTorch 2.6+ เปลี่ยน default `weights_only=True` ซึ่งทำให้โหลด checkpoint เก่าไม่ได้
`app.py` patch ให้ default กลับเป็น `weights_only=False`

### 3. moviepy v2 Compatibility
moviepy v2 ลบ `moviepy.editor` ไปแล้ว `app.py` มี try/except fallback

---

## 🐛 Troubleshooting

| ปัญหา | วิธีแก้ |
|-------|--------|
| `No module named 'gradio'` | `.venv\Scripts\python.exe -m pip install gradio` |
| `No module named 'mmpose'` | ดู "ติดตั้ง Dependencies เพิ่มเติม" ด้านบน |
| `MMCV incompatible` | ใช้ `mmcv-lite==2.1.0` (ไม่ใช่ mmcv) |
| `weights_only` error | Patch อยู่ใน app.py แล้ว (torch.load) |
| `moviepy.editor` not found | Patch อยู่ใน app.py แล้ว (try/except) |
| `omnivoice-rt` not found | ต้อง `cd omnivoice-tts` ก่อนรัน |
| ดาวน์โหลด weights ช้า | ตั้ง `HF_TOKEN` สำหรับ HuggingFace |
| `CUDA_HOME not set` | ใช้ mmcv-lite แทน (ไม่ต้องติดตั้ง CUDA Toolkit) |
