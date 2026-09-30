# Patch mmcv-lite: stub mmcv._ext so mmpose can import without compiled CUDA ops
import importlib, importlib.machinery, types, sys

class _StubExt(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = '<mmcv_lite_stub>'
        self.__loader__ = importlib.machinery.SourceFileLoader(name, '<mmcv_lite_stub>')
        self.__spec__ = importlib.machinery.ModuleSpec(name, self.__loader__)
        self.__package__ = 'mmcv'
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        def _stub(*args, **kwargs):
            raise NotImplementedError(f"mmcv._ext.{name} requires full mmcv with CUDA ops")
        return _stub

_orig_import = importlib.import_module
def _patched_import(name, *args, **kwargs):
    if name == 'mmcv._ext':
        if name not in sys.modules:
            sys.modules[name] = _StubExt(name)
        return sys.modules[name]
    return _orig_import(name, *args, **kwargs)
importlib.import_module = _patched_import

# Patch torch.load for PyTorch 2.6+
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ═══════════════════════════════════════════════════════════════
# Live AI Avatar Studio
# ═══════════════════════════════════════════════════════════════
import os
import time
import json
import re
import argparse
import threading
import tempfile
import numpy as np
import cv2
import glob
import pickle
import copy
import shutil

import gradio as gr
import requests
import imageio

try:
    from moviepy.editor import *
except ImportError:
    from moviepy import *

from argparse import Namespace
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import WhisperModel

# ═══════════════════════════════════════════════════════════════
# Load Config & Models
# ═══════════════════════════════════════════════════════════════
ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

# Load API config
API_CONFIG_PATH = os.path.join(os.path.dirname(ProjectDir), "api.json")
if os.path.exists(API_CONFIG_PATH):
    with open(API_CONFIG_PATH, 'r', encoding='utf-8') as f:
        API_CONFIG = json.load(f)
else:
    API_CONFIG = {"providers": {}}

def get_llm_choices():
    """Extract model choices from api.json"""
    choices = []
    for provider_id, provider in API_CONFIG.get("providers", {}).items():
        for model_id, model_info in provider.get("models", {}).items():
            label = f"{provider['name']} / {model_info['name']}"
            choices.append(label)
    return choices

def get_llm_config(choice_label):
    """Get API config for a selected model"""
    for provider_id, provider in API_CONFIG.get("providers", {}).items():
        for model_id, model_info in provider.get("models", {}).items():
            label = f"{provider['name']} / {model_info['name']}"
            if label == choice_label:
                return {
                    "base_url": provider["settings"]["baseURL"],
                    "api_key": provider["settings"].get("apiKey", os.environ.get(provider.get("env", [""])[0] if provider.get("env") else "", "")),
                    "model_id": model_info["modelID"],
                    "provider": provider["name"],
                    "model_name": model_info["name"]
                }
    return None

# Voice style presets (Thai labels → English instruct for OmniVoice)
VOICE_STYLE_MAP = {
    "หญิง - เสียงต่ำ": "female, low pitch",
    "หญิง - เสียงปกติ": "female, moderate pitch",
    "หญิง - เสียงสูง": "female, high pitch",
    "หญิง - กระซิบ": "female, whisper",
    "หญิง - วัยรุ่น เสียงสูง": "female, young adult, high pitch",
    "ชาย - เสียงต่ำ": "male, low pitch",
    "ชาย - เสียงทุ้ม": "male, very low pitch",
    "ชาย - เสียงปกติ": "male, moderate pitch",
    "ชาย - กระซิบ": "male, whisper",
    "ชาย - ผู้ใหญ่ เสียงทุ้ม": "male, elderly, low pitch",
    "โคลนเสียง (อัพโหลดไฟล์)": "__clone__",
}
VOICE_STYLES = list(VOICE_STYLE_MAP.keys())

# ═══════════════════════════════════════════════════════════════
# Load MuseTalk Models (same as app.py)
# ═══════════════════════════════════════════════════════════════
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor

print("Loading MuseTalk models...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path=os.path.join(CheckpointsDir, "musetalkV15", "unet.pth"),
    vae_type="sd-vae",
    unet_config=os.path.join(CheckpointsDir, "musetalkV15", "musetalk.json"),
    device=device
)
pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)
weight_dtype = unet.model.dtype
timesteps = torch.tensor([0], device=device)

audio_processor = AudioProcessor(feature_extractor_path=os.path.join(CheckpointsDir, "whisper"))
whisper = WhisperModel.from_pretrained(os.path.join(CheckpointsDir, "whisper"))
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
print("MuseTalk models loaded!")

# ═══════════════════════════════════════════════════════════════
# LLM Script Generation
# ═══════════════════════════════════════════════════════════════
def generate_script_llm(model_choice, system_prompt, product_info, progress=gr.Progress()):
    """Generate script using selected LLM"""
    config = get_llm_config(model_choice)
    if not config:
        return "❌ ไม่พบ model ที่เลือก"
    
    progress(0.1, desc="🧠 กำลังสร้างสคริปต์...")
    
    base_url = config["base_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        if "/v1/" not in base_url:
            base_url = base_url + "/v1" if not base_url.endswith("/v1") else base_url
    
    url = f"{base_url}/chat/completions"
    
    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    
    if not system_prompt.strip():
        system_prompt = """คุณคือนักขายสินค้าออนไลน์มืออาชีพที่กำลังไลฟ์สดขายสินค้า 
คุณต้องพูดเป็นธรรมชาติ กระตุ้นให้ลูกค้าสนใจและอยากซื้อ
ใช้ภาษาไทยเท่านั้น พูดสั้นกระชับแต่ละย่อหน้า ประมาณ 2-3 ประโยค
สร้าง 3-5 ย่อหน้าสำหรับไลฟ์สด"""

    user_msg = f"สร้างสคริปต์ไลฟ์สดสำหรับขายสินค้านี้:\n\n{product_info}"
    
    payload = {
        "model": config["model_id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.8,
        "max_tokens": 2048,
        "stream": False
    }
    
    try:
        progress(0.3, desc=f"📡 เชื่อมต่อ {config['provider']}...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        script = data["choices"][0]["message"]["content"]
        progress(1.0, desc="✅ สร้างสคริปต์เสร็จ!")
        return script
        
    except requests.exceptions.Timeout:
        return "❌ หมดเวลาเชื่อมต่อ (timeout) ลองใหม่อีกครั้ง"
    except requests.exceptions.RequestException as e:
        return f"❌ เชื่อมต่อ API ไม่ได้: {str(e)}"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {str(e)}"

# ═══════════════════════════════════════════════════════════════
# TTS Generation (OmniVoice)
# ═══════════════════════════════════════════════════════════════
def generate_tts(text, voice_style, progress=gr.Progress()):
    """Generate TTS audio using OmniVoice TTS server or CLI"""
    progress(0.1, desc="Generating audio...")
    
    # Translate Thai label to English instruct
    instruct = VOICE_STYLE_MAP.get(voice_style, voice_style)
    if instruct == "__clone__":
        instruct = "female, moderate pitch"  # fallback for clone mode
    
    # Try OmniVoice TTS server first
    try:
        response = requests.post(
            "http://localhost:8765/tts/generate",
            data={"text": text, "instruct": instruct, "num_step": 16},
            timeout=30
        )
        if response.status_code == 200:
            audio_path = os.path.join(ProjectDir, "results", "tts_output.wav")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            with open(audio_path, 'wb') as f:
                f.write(response.content)
            progress(1.0, desc="✅ สร้างเสียงเสร็จ!")
            return audio_path
    except:
        pass
    
    # Fallback: use CLI
    try:
        import subprocess
        audio_path = os.path.join(ProjectDir, "results", "tts_output.wav")
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        tts_dir = os.path.join(os.path.dirname(ProjectDir), "omnivoice-tts")
        cmd = [
            os.path.join(tts_dir, ".venv", "Scripts", "python.exe"),
            "-m", "omnivoice_tts.realtime",
            "--text", text,
            "--instruct", instruct,
            "--output", audio_path,
            "--mode", "oneshot"
        ]
        progress(0.3, desc="🎙️ เรียก OmniVoice CLI...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=tts_dir)
        if result.returncode == 0 and os.path.exists(audio_path):
            progress(1.0, desc="✅ สร้างเสียงเสร็จ!")
            return audio_path
        else:
            return None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# Lipsync Inference (MuseTalk v1.5)
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def lipsync_inference(video_path, audio_path, progress=gr.Progress()):
    """Run MuseTalk v1.5 lipsync inference"""
    if not video_path or not audio_path:
        return None
    
    progress(0.05, desc="🎭 เตรียมข้อมูล...")
    
    args_dict = {
        "result_dir": os.path.join(ProjectDir, "results", "studio"),
        "fps": 25,
        "batch_size": 8,
        "output_vid_name": None,
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",
        "extra_margin": 10,
        "parsing_mode": "jaw",
        "left_cheek_width": 90,
        "right_cheek_width": 90,
    }
    args = Namespace(**args_dict)
    os.makedirs(args.result_dir, exist_ok=True)
    
    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    
    temp_dir = os.path.join(args.result_dir, "v15")
    os.makedirs(temp_dir, exist_ok=True)
    
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, input_basename + ".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)
    
    output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
    
    # Extract frames
    progress(0.1, desc="🎞️ แยกเฟรมจากวิดีโอ...")
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f'ffmpeg -v fatal -i "{video_path}" -start_number 0 {save_dir_full}/%08d.png'
        os.system(cmd)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    elif get_file_type(video_path) == "image":
        input_img_list = [video_path]
        fps = args.fps
    else:
        input_img_list = [video_path]
        fps = args.fps
    
    # Audio features
    progress(0.2, desc="🔊 วิเคราะห์เสียง...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper, librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
    
    # Face detection
    progress(0.3, desc="👤 ตรวจจับใบหน้า...")
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        with open(crop_coord_save_path, 'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    
    # Process frames
    progress(0.4, desc="🧮 เตรียม latents...")
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    # Generate
    progress(0.5, desc="🎨 สร้างวิดีโอ lip-sync...")
    video_num = len(whisper_chunks)
    gen = datagen(whisper_chunks, input_latent_list_cycle, args.batch_size)
    res_frame_list = []
    
    for i, (whisper_batch, latent_batch) in enumerate(gen):
        audio_feature_batch = pe(whisper_batch.to(device))
        latent_batch = latent_batch.to(dtype=weight_dtype, device=device)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
        
        prog = 0.5 + (0.3 * (i * args.batch_size) / max(video_num, 1))
        progress(min(prog, 0.8), desc=f"🎨 สร้างเฟรม {min((i+1)*args.batch_size, video_num)}/{video_num}...")
    
    # Blend
    progress(0.8, desc="🖼️ ผสานเฟรม...")
    for i, res_frame in enumerate(res_frame_list):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, ori_frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except:
            continue
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
    
    # Encode video
    progress(0.9, desc="📹 เข้ารหัสวิดีโอ...")
    temp_vid_path = os.path.join(temp_dir, f"temp_{output_basename}.mp4")
    cmd = f'ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 "{temp_vid_path}"'
    os.system(cmd)
    
    cmd = f'ffmpeg -y -v warning -i "{audio_path}" -i "{temp_vid_path}" "{output_vid_name}"'
    os.system(cmd)
    
    # Cleanup
    shutil.rmtree(result_img_save_path, ignore_errors=True)
    if os.path.exists(temp_vid_path):
        os.remove(temp_vid_path)
    if get_file_type(video_path) == "video" and os.path.exists(save_dir_full):
        shutil.rmtree(save_dir_full, ignore_errors=True)
    
    progress(1.0, desc="✅ สร้างวิดีโอเสร็จ!")
    return output_vid_name if os.path.exists(output_vid_name) else None

# ═══════════════════════════════════════════════════════════════
# Video check helper
# ═══════════════════════════════════════════════════════════════
def check_video(video):
    if not isinstance(video, str):
        return video
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    output_file_name = "outputxxx_" + file_name
    os.makedirs('./results/input', exist_ok=True)
    output_video = os.path.join('./results/input', output_file_name)
    
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']
    frames = [im for im in reader]
    target_fps = 25
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L + 1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target + 1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video

# ═══════════════════════════════════════════════════════════════
# Full Pipeline: Script → TTS → Lipsync
# ═══════════════════════════════════════════════════════════════
def full_pipeline(video, script_text, voice_style, progress=gr.Progress()):
    """Run the full pipeline: TTS → Lipsync"""
    if not video:
        return None, "❌ กรุณาอัพโหลดวิดีโอ/รูปภาพ reference"
    if not script_text or not script_text.strip():
        return None, "❌ กรุณาสร้างสคริปต์ก่อน"
    
    # Clean script - take first paragraph for demo
    paragraphs = [p.strip() for p in script_text.split('\n') if p.strip() and not p.strip().startswith('#')]
    demo_text = paragraphs[0] if paragraphs else script_text[:200]
    
    # Remove markdown formatting
    demo_text = re.sub(r'[*_#>`\[\]()]', '', demo_text).strip()
    if len(demo_text) > 300:
        demo_text = demo_text[:300]
    
    # Step 1: TTS
    progress(0.1, desc="🎙️ Step 1/2: สร้างเสียง...")
    audio_path = generate_tts(demo_text, voice_style, progress)
    if not audio_path:
        return None, "❌ สร้างเสียงไม่สำเร็จ — ลองเปิด OmniVoice TTS server ก่อน:\ncd omnivoice-tts && uv run omnivoice-tts"
    
    # Step 2: Lipsync
    progress(0.5, desc="🎭 Step 2/2: สร้างวิดีโอ lip-sync...")
    output_video = lipsync_inference(video, audio_path, progress)
    if not output_video:
        return None, "❌ สร้างวิดีโอไม่สำเร็จ"
    
    return output_video, f"✅ สร้างเสร็จ!\n📝 ข้อความ: {demo_text[:100]}...\n🎙️ เสียง: {voice_style}"

# ═══════════════════════════════════════════════════════════════
# Test AI Brain
# ═══════════════════════════════════════════════════════════════
def test_ai_brain(model_choice):
    """Test AI Brain connection and count sentences"""
    config = get_llm_config(model_choice)
    if not config:
        return "[X] ไม่พบ model"
    
    import time as _time
    base_url = config["base_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1" if "/v1/" not in base_url else base_url
    url = f"{base_url}/chat/completions"
    
    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    
    payload = {
        "model": config["model_id"],
        "messages": [
            {"role": "system", "content": "คุณคือนักขายออนไลน์ พูดภาษาไทย สั้นกระชับ"},
            {"role": "user", "content": "แนะนำครีมทาหน้าให้ลูกค้าสั้นๆ 3 ประโยค"}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }
    
    try:
        start = _time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        elapsed = _time.time() - start
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        
        sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip() and len(s.strip()) > 5]
        chars = len(text)
        
        return (f"[OK] เชื่อมต่อสำเร็จ! ({config['provider']} / {config['model_name']})\n"
                f"เวลา: {elapsed:.1f} วินาที | {len(sentences)} ประโยค | {chars} ตัวอักษร\n"
                f"---\n{text[:300]}")
    except requests.exceptions.Timeout:
        return f"[X] หมดเวลา (timeout 30s) - {config['provider']}"
    except Exception as e:
        return f"[X] เชื่อมต่อไม่ได้: {str(e)[:200]}"

# ═══════════════════════════════════════════════════════════════
# Gradio UI — Live AI Avatar Studio (TikTok Live Style)
# ═══════════════════════════════════════════════════════════════
css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700&display=swap');

* { font-family: 'Noto Sans Thai', 'Inter', sans-serif !important; }

.gradio-container {
    background: linear-gradient(160deg, #0a0a0a 0%, #1a0a2e 30%, #16082a 60%, #0d0d0d 100%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 12px 20px;
    background: linear-gradient(90deg, rgba(254,44,85,0.15), rgba(37,244,238,0.12), rgba(254,44,85,0.15));
    border-radius: 14px;
    border: 1px solid rgba(254,44,85,0.25);
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent, rgba(254,44,85,0.08), transparent, rgba(37,244,238,0.08), transparent);
    animation: spin 8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.main-header h1 {
    position: relative;
    background: linear-gradient(90deg, #fe2c55, #25f4ee, #fe2c55);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    animation: gradient-shift 3s ease infinite;
}

@keyframes gradient-shift {
    0%,100% { background-position: 0% center; }
    50% { background-position: 100% center; }
}

.main-header p {
    position: relative;
    color: rgba(255,255,255,0.5);
    margin: 2px 0 0 0;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}

/* Cards */
.gr-group, .gr-box, .gr-panel {
    background: rgba(22, 18, 40, 0.85) !important;
    border: 1px solid rgba(254,44,85,0.12) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
}

/* Generate Script Button */
.generate-btn {
    background: linear-gradient(135deg, #fe2c55, #d62b5e) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 20px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(254,44,85,0.3) !important;
}

.generate-btn:hover {
    background: linear-gradient(135deg, #ff4470, #fe2c55) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(254,44,85,0.5) !important;
}

/* Pipeline Button */
.pipeline-btn {
    background: linear-gradient(135deg, #25f4ee, #00c9db) !important;
    border: none !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(37,244,238,0.3) !important;
    transition: all 0.3s ease !important;
}

.pipeline-btn:hover {
    background: linear-gradient(135deg, #5ffffa, #25f4ee) !important;
    box-shadow: 0 6px 30px rgba(37,244,238,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Labels */
label, .gr-input-label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.82rem !important;
}

/* Inputs */
textarea, input[type="text"], .gr-text-input {
    background: rgba(30,25,50,0.9) !important;
    border: 1px solid rgba(254,44,85,0.15) !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
    font-size: 0.88rem !important;
}

textarea:focus, input:focus {
    border-color: rgba(37,244,238,0.5) !important;
    box-shadow: 0 0 12px rgba(37,244,238,0.15) !important;
}

/* Dropdowns */
.gr-dropdown {
    background: rgba(30,25,50,0.9) !important;
    border: 1px solid rgba(254,44,85,0.15) !important;
    color: #e8e8e8 !important;
}

/* 9:16 Video Preview */
.phone-preview video, .phone-preview img {
    aspect-ratio: 9/16 !important;
    max-height: 420px !important;
    object-fit: cover !important;
    border-radius: 16px !important;
    border: 2px solid rgba(254,44,85,0.3) !important;
    box-shadow: 0 8px 32px rgba(254,44,85,0.15), 0 0 60px rgba(37,244,238,0.05) !important;
}

/* Accordion */
.gr-accordion {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}

/* Live badge style */
.live-badge {
    display: inline-block;
    background: #fe2c55;
    color: white;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    animation: pulse-live 2s ease-in-out infinite;
}

@keyframes pulse-live {
    0%,100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #fe2c55; border-radius: 4px; }
"""

llm_choices = get_llm_choices()
default_system_prompt = """คุณคือพิธีกรไลฟ์สดขายสินค้าออนไลน์มืออาชีพ ชื่อ "น้องไอ" 
พูดภาษาไทยเท่านั้น น้ำเสียงเป็นกันเอง กระตุ้นให้ลูกค้าสนใจ
สร้างสคริปต์ 3-5 ย่อหน้า แต่ละย่อหน้าสั้น 2-3 ประโยค
เน้นจุดเด่นสินค้า โปรโมชัน และ call-to-action"""

with gr.Blocks(css=css, title="Live AI Avatar Studio", theme=gr.themes.Base()) as demo:
    
    # Header
    gr.HTML('<div class="main-header"><h1>LIVE AI Avatar Studio</h1><p><span class="live-badge">LIVE</span> &nbsp; สร้างวิดีโอ AI ขายสินค้า สไตล์ไลฟ์สด</p></div>')
    
    with gr.Row(equal_height=False):
        # ═══════ LEFT: Settings (compact) ═══════
        with gr.Column(scale=2, min_width=320):
            with gr.Row():
                with gr.Column(scale=1, min_width=150):
                    avatar_video = gr.Video(label="Reference (วิดีโอ)", sources=['upload'], height=140)
                with gr.Column(scale=1, min_width=150):
                    avatar_image = gr.Image(label="Reference (รูปภาพ)", type="filepath", height=140)
            
            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=llm_choices,
                    value=llm_choices[0] if llm_choices else None,
                    label="สมอง AI",
                    interactive=True, scale=3
                )
                test_brain_btn = gr.Button("ทดสอบ", size="sm", scale=0, min_width=65)
            
            test_result = gr.Textbox(label="ผลทดสอบ AI", lines=4, interactive=False, visible=False)
            
            voice_style = gr.Dropdown(
                choices=VOICE_STYLES,
                value=VOICE_STYLES[0],
                label="เสียง",
                interactive=True
            )
            
            voice_clone_audio = gr.Audio(
                label="ไฟล์เสียงต้นแบบ (โคลนเสียง)",
                type="filepath", visible=False
            )
            
            product_info = gr.Textbox(
                label="สินค้า / คำสั่ง",
                lines=3,
                placeholder="เซรั่ม VitaGlow\n- Hyaluronic Acid, Vit C\n- ราคา 590 (ปกติ 990)\n- ซื้อ 2 แถม 1",
            )
            
            script_btn = gr.Button("สร้างสคริปต์ AI", elem_classes=["generate-btn"])
            
            script_output = gr.Textbox(
                label="สคริปต์ (แก้ไขได้)",
                lines=6,
                placeholder="กดปุ่ม 'สร้างสคริปต์ AI' หรือพิมพ์เอง...",
                interactive=True
            )
            
            with gr.Accordion("System Prompt", open=False):
                system_prompt = gr.Textbox(
                    value=default_system_prompt, label="คำสั่งระบบ", lines=3
                )
        
        # ═══════ RIGHT: 9:16 Preview ═══════
        with gr.Column(scale=1, min_width=280, elem_classes=["phone-preview"]):
            pipeline_btn = gr.Button("สร้างวิดีโอ Avatar", elem_classes=["pipeline-btn"])
            output_video = gr.Video(label="Preview 9:16", height=420)
            status_output = gr.Textbox(label="สถานะ", lines=1, interactive=False, value="พร้อมไลฟ์!")
    
    # ═══════════════════════════════════════
    # Event Handlers
    # ═══════════════════════════════════════
    
    # Convert uploaded image to video format
    def image_to_video_ref(image_path):
        if image_path:
            return image_path  # MuseTalk accepts images directly
        return None
    
    # Toggle voice clone upload visibility
    def toggle_voice_clone(style):
        is_clone = VOICE_STYLE_MAP.get(style, "") == "__clone__"
        return gr.update(visible=is_clone)
    
    avatar_image.change(
        fn=image_to_video_ref,
        inputs=[avatar_image],
        outputs=[avatar_video]
    )
    
    avatar_video.change(
        fn=check_video,
        inputs=[avatar_video],
        outputs=[avatar_video]
    )
    
    voice_style.change(
        fn=toggle_voice_clone,
        inputs=[voice_style],
        outputs=[voice_clone_audio]
    )
    
    # Test AI Brain
    def run_test_brain(model):
        return gr.update(visible=True, value=test_ai_brain(model))
    
    test_brain_btn.click(
        fn=run_test_brain,
        inputs=[model_choice],
        outputs=[test_result]
    )
    
    # Generate script
    script_btn.click(
        fn=generate_script_llm,
        inputs=[model_choice, system_prompt, product_info],
        outputs=[script_output]
    )
    
    # Full pipeline: TTS + Lipsync
    pipeline_btn.click(
        fn=full_pipeline,
        inputs=[avatar_video, script_output, voice_style],
        outputs=[output_video, status_output]
    )

# ═══════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
args = parser.parse_args()

# Check ffmpeg
def fast_check_ffmpeg():
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

if not fast_check_ffmpeg():
    print("Warning: ffmpeg not found in PATH!")

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

print("\n[*] Live AI Avatar Studio starting...")
print(f"[*] Open http://{args.ip}:{args.port}")
demo.queue().launch(
    share=args.share,
    server_name=args.ip,
    server_port=args.port
)
