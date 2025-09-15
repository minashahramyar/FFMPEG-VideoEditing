# # detect.py
# from __future__ import annotations

# import re, subprocess
# from typing import List, Tuple, Optional

# import cv2
# import numpy as np

# from .ffutil import run, ffprobe_json

# # -------------------
# # Helpers
# # -------------------

# _FLOAT = r"([0-9]+(?:\.[0-9]+)?)"

# def _parse_all(pattern: str, text: str) -> List[float]:
#     return [float(m.group(1)) for m in re.finditer(pattern, text)]

# def _dedupe_times(times: List[float], eps: float = 0.35) -> List[float]:
#     if not times: return []
#     times = sorted(times)
#     out = [times[0]]
#     for t in times[1:]:
#         if t - out[-1] >= eps:
#             out.append(t)
#     return out

# def _clamp_to_scan(times: List[float], scan_limit_s: float) -> List[float]:
#     return [t for t in times if 0.0 <= t <= scan_limit_s]


# # -------------------
# # Metadata
# # -------------------

# def first_chapter_from_metadata(path: str) -> Optional[Tuple[float, float]]:
#     data = ffprobe_json(path, "chapters=start_time,end_time,tags")
#     chs = data.get("chapters", [])
#     if not chs: return None
#     c0 = chs[0]
#     try:
#         return float(c0["start_time"]), float(c0["end_time"])
#     except: return None


# # -------------------
# # Visual / audio cues
# # -------------------

# def detect_scene_cuts(path: str, scene_thresh=0.30, scan_limit_s=120.0) -> List[float]:
#     _, _, err = run([
#         "ffmpeg","-hide_banner","-nostdin","-loglevel","info",
#         "-t", str(scan_limit_s), "-i", path,
#         "-vf", f"select='gt(scene\\,{scene_thresh})',showinfo",
#         "-an","-f","null","-"
#     ])
#     times = _parse_all(rf"pts_time:{_FLOAT}", err)
#     return _dedupe_times(_clamp_to_scan(times, scan_limit_s))

# def detect_black_segments(path: str, pic_th=0.98, dur_th=0.2, scan_limit_s=120.0) -> List[Tuple[float,float]]:
#     _, _, err = run([
#         "ffmpeg","-hide_banner","-nostdin","-loglevel","info",
#         "-t", str(scan_limit_s), "-i", path,
#         "-vf", f"blackdetect=d={dur_th}:pic_th={pic_th}",
#         "-an","-f","null","-"
#     ])
#     segs=[]; start=None
#     for line in err.splitlines():
#         ms = re.search(rf"black_start:{_FLOAT}", line)
#         me = re.search(rf"black_end:{_FLOAT}", line)
#         if ms: start=float(ms.group(1))
#         if me and start is not None:
#             segs.append((start,float(me.group(1)))); start=None
#     return segs

# def detect_silence_starts(path: str, noise_db=-30.0, min_dur=0.5, scan_limit_s=120.0) -> List[float]:
#     _, _, err = run([
#         "ffmpeg","-hide_banner","-nostdin","-loglevel","info",
#         "-t", str(scan_limit_s), "-i", path,
#         "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
#         "-f","null","-"
#     ])
#     times=_parse_all(rf"silence_start:\s*{_FLOAT}",err)
#     return _dedupe_times(_clamp_to_scan(times,scan_limit_s))


# # -------------------
# # Speech (VAD)
# # -------------------

# def detect_speech_onset(path: str, scan_limit_s=120.0, aggressiveness=2,
#                         frame_ms=30, min_speech_run_s=1.0) -> List[float]:
#     try:
#         import webrtcvad
#     except: return []
#     cmd=[
#         "ffmpeg","-hide_banner","-nostdin","-loglevel","error",
#         "-t",str(scan_limit_s),"-i",path,
#         "-vn","-ac","1","-ar","16000","-f","s16le","-"
#     ]
#     p=subprocess.Popen(cmd,stdout=subprocess.PIPE)
#     pcm=p.stdout.read() if p.stdout else b""; p.wait()
#     if not pcm: return []
#     vad=webrtcvad.Vad(aggressiveness); sr=16000; bps=2
#     flen=int(sr*frame_ms/1000); step=flen*bps
#     voiced=[]; run_start=None; run_frames=0
#     for i in range(0,len(pcm)-step+1,step):
#         f=pcm[i:i+step]; t=i/(bps*sr)
#         voiced_flag=vad.is_speech(f,sr)
#         if voiced_flag:
#             if run_start is None: run_start=t; run_frames=1
#             else: run_frames+=1
#         else:
#             if run_start is not None and run_frames*(frame_ms/1000)>=min_speech_run_s:
#                 voiced.append(run_start)
#             run_start=None; run_frames=0
#     return _dedupe_times(_clamp_to_scan(voiced,scan_limit_s))


# # -------------------
# # Logo auto-template + detection
# # -------------------

# def _read_frames(path:str,fps:float,scan_limit_s:float):
#     from .ffutil import run
#     _,out,_=run(["ffprobe","-hide_banner","-loglevel","error",
#                  "-select_streams","v:0","-show_entries","stream=width,height",
#                  "-of","csv=s=,:p=0",path])
#     W,H=[int(x) for x in out.strip().split(",")]
#     cmd=["ffmpeg","-hide_banner","-nostdin","-loglevel","error",
#          "-t",str(scan_limit_s),"-i",path,
#          "-vf",f"fps={fps}","-f","rawvideo","-pix_fmt","bgr24","-"]
#     p=subprocess.Popen(cmd,stdout=subprocess.PIPE); fsize=W*H*3
#     frames=[]
#     while True:
#         buf=p.stdout.read(fsize) if p.stdout else None
#         if not buf or len(buf)<fsize: break
#         frames.append(np.frombuffer(buf,dtype=np.uint8).reshape(H,W,3))
#     p.wait(); return frames,W,H

# def auto_generate_logo_template(path:str,out_png:str,scan_limit_s=120.0,fps=10.0,
#                                 roi_rel:Tuple[float,float,float,float]|None=None)->bool:
#     frames,W,H=_read_frames(path,fps,scan_limit_s)
#     if not frames: return False
#     gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]
#     stack=np.stack(gray,0).astype(np.float32)
#     std = np.std(stack, 0)
#     sal = 1.0 - (std - std.min()) / (np.ptp(std) + 1e-6)
#     med=np.median(stack,0).astype(np.uint8)
#     thr=np.percentile(sal,90); mask=(sal>=thr).astype(np.uint8)*255
#     cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts: return False
#     x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
#     crop=med[y:y+h,x:x+w]; return cv2.imwrite(out_png,crop)

# def detect_logo_times(path:str,logo_path:Optional[str],scan_limit_s=120.0,
#                       fps=10.0,match_thresh=0.75)->List[float]:
#     if not logo_path: return []
#     tpl=cv2.imread(logo_path,cv2.IMREAD_GRAYSCALE)
#     if tpl is None: return []
#     th,tw=tpl.shape[:2]
#     frames,W,H=_read_frames(path,fps,scan_limit_s)
#     times=[]
#     for i,f in enumerate(frames):
#         gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
#         res=cv2.matchTemplate(gray,tpl,cv2.TM_CCOEFF_NORMED)
#         _,mv,_,_=cv2.minMaxLoc(res)
#         t=i/fps
#         if mv>=match_thresh: times.append(t)
#     return _dedupe_times(times)


# # -------------------
# # Keyframes
# # -------------------

# def list_keyframes(path:str,scan_limit_s=180.0)->List[float]:
#     _,out,_=run(["ffprobe","-hide_banner","-loglevel","error",
#                  "-read_intervals",f"%+{scan_limit_s}",
#                  "-select_streams","v:0","-show_frames",
#                  "-show_entries","frame=pkt_pts_time,key_frame",
#                  "-of","csv",path])
#     kf=[]
#     for line in (out or "").splitlines():
#         parts=line.split(",")
#         if len(parts)>=3 and parts[1]=="1":
#             try:kf.append(float(parts[2]))
#             except: pass
#     return sorted(kf)

# def snap_to_prev_keyframe(t:float,kf:List[float],max_jump_s=1.0)->float:
#     prev=None
#     for k in kf:
#         if k<=t: prev=k
#         else: break
#     if prev is not None and t-prev<=max_jump_s: return prev
#     return t


# detect.py
from __future__ import annotations

import re
import subprocess
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

from .ffutil import run, ffprobe_json


# ---------------------------
# Helpers
# ---------------------------

_FLOAT = r"([0-9]+(?:\.[0-9]+)?)"

def _parse_all(pattern: str, text: str) -> List[float]:
    return [float(m.group(1)) for m in re.finditer(pattern, text)]

def _dedupe_times(times: List[float], eps: float = 0.35) -> List[float]:
    if not times:
        return []
    times = sorted(times)
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] >= eps:
            out.append(t)
    return out

def _clamp_to_scan(times: List[float], scan_limit_s: float) -> List[float]:
    return [t for t in times if 0.0 <= t <= scan_limit_s]


# ---------------------------
# Metadata (chapters)
# ---------------------------

def first_chapter_from_metadata(path: str) -> Optional[Tuple[float, float]]:
    """Return (start_s, end_s) for first embedded chapter if present, else None."""
    data = ffprobe_json(path, "chapters=start_time,end_time,tags")
    chs = data.get("chapters", [])
    if not chs:
        return None
    c0 = chs[0]
    try:
        return float(c0["start_time"]), float(c0["end_time"])
    except Exception:
        return None


# ---------------------------
# FFmpeg-based visual/audio cues
# ---------------------------

def detect_scene_cuts(path: str, scene_thresh: float = 0.32, scan_limit_s: float = 120.0) -> List[float]:
    """Scene-cut times via FFmpeg scene detector."""
    _, _, err = run([
        "ffmpeg","-hide_banner","-nostdin","-loglevel","info",
        "-t", str(scan_limit_s), "-i", path,
        "-vf", f"select='gt(scene\\,{scene_thresh})',showinfo",
        "-an","-f","null","-"
    ])
    times = _parse_all(rf"pts_time:{_FLOAT}", err)
    return _dedupe_times(_clamp_to_scan(times, scan_limit_s))

def detect_black_segments(path: str, pic_th: float = 0.98, dur_th: float = 0.20, scan_limit_s: float = 120.0) -> List[Tuple[float,float]]:
    """(black_start, black_end) via blackdetect."""
    _, _, err = run([
        "ffmpeg","-hide_banner","-nostdin","-loglevel","info",
        "-t", str(scan_limit_s), "-i", path,
        "-vf", f"blackdetect=d={dur_th}:pic_th={pic_th}",
        "-an","-f","null","-"
    ])
    segs: List[Tuple[float,float]] = []
    start: Optional[float] = None
    for line in err.splitlines():
        ms = re.search(rf"black_start:{_FLOAT}", line)
        me = re.search(rf"black_end:{_FLOAT}", line)
        if ms:
            try: start = float(ms.group(1))
            except: start = None
        if me and start is not None:
            try:
                end = float(me.group(1))
                if 0 <= start <= scan_limit_s and 0 <= end <= scan_limit_s and end > start:
                    segs.append((start, end))
            except:
                pass
            start = None
    if not segs:
        return segs
    segs.sort(key=lambda x: (x[0], x[1]))
    out = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = out[-1]
        if (s - ps) >= 0.35 or (e - pe) >= 0.35:
            out.append((s, e))
    return out

def detect_silence_starts(path: str, noise_db: float = -30.0, min_dur: float = 0.50, scan_limit_s: float = 120.0) -> List[float]:
    """Times where FFmpeg silencedetect reports silence_start."""
    _, _, err = run([
        "ffmpeg","-hide_banner","-nostdin","-loglevel","info",
        "-t", str(scan_limit_s), "-i", path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f","null","-"
    ])
    times = _parse_all(rf"silence_start:\s*{_FLOAT}", err)
    return _dedupe_times(_clamp_to_scan(times, scan_limit_s))


# ---------------------------
# Audio-first cues (VAD, long pause)
# ---------------------------

def _pcm16_mono_16k(path: str, scan_limit_s: float) -> bytes:
    cmd = [
        "ffmpeg","-hide_banner","-nostdin","-loglevel","error",
        "-t", str(scan_limit_s), "-i", path,
        "-vn","-ac","1","-ar","16000","-f","s16le","-"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    pcm = p.stdout.read() if p.stdout else b""
    p.wait()
    return pcm

def detect_speech_onset(path: str, scan_limit_s: float = 120.0, vad_aggr: int = 3, frame_ms: int = 30, min_speech_run_s: float = 1.0) -> List[float]:
    """Candidate speech-start times using WebRTC VAD."""
    try:
        import webrtcvad  # type: ignore
    except Exception:
        return []
    pcm = _pcm16_mono_16k(path, scan_limit_s)
    if not pcm:
        return []
    sr = 16000
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    vad = webrtcvad.Vad(vad_aggr)
    spf = int(sr * frame_ms / 1000)
    n_frames = len(x) // spf
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        frame = (x[i*spf:(i+1)*spf] * 32768.0).astype(np.int16).tobytes()
        try:
            voiced[i] = vad.is_speech(frame, sr)
        except Exception:
            voiced[i] = False
    times = np.arange(n_frames) * (frame_ms / 1000.0)
    need = int(np.ceil(min_speech_run_s / (frame_ms / 1000.0)))
    starts: List[float] = []
    run = 0; start_idx = None
    for i in range(n_frames):
        if voiced[i]:
            if start_idx is None:
                start_idx = i; run = 1
            else:
                run += 1
        else:
            if start_idx is not None and run >= need:
                starts.append(float(times[start_idx]))
            start_idx = None; run = 0
    if start_idx is not None and run >= need:
        starts.append(float(times[start_idx]))
    return _dedupe_times(_clamp_to_scan(starts, scan_limit_s))

def detect_long_pause_after_speech(
    path: str,
    scan_limit_s: float = 120.0,
    vad_aggr: int = 3,
    frame_ms: int = 30,
    min_initial_speech_run_s: float = 3.0,
    long_pause_s: float = 5.0,
    rms_db_floor: float = -45.0,
) -> List[float]:
    """
    Return [t_pause_start] when we see the first sustained QUIET pause (>= long_pause_s)
    right after an initial continuous speech run (>= min_initial_speech_run_s).
    """
    try:
        import webrtcvad  # type: ignore
    except Exception:
        return []
    pcm = _pcm16_mono_16k(path, scan_limit_s)
    if not pcm:
        return []
    sr = 16000
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    vad = webrtcvad.Vad(vad_aggr)
    spf = int(sr * frame_ms / 1000)
    n_frames = len(x) // spf
    voiced = np.zeros(n_frames, dtype=bool)
    rms_db = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = x[i*spf:(i+1)*spf]
        try:
            voiced[i] = vad.is_speech((frame * 32768.0).astype(np.int16).tobytes(), sr)
        except Exception:
            voiced[i] = False
        rms = np.sqrt(np.mean(np.square(frame)) + 1e-12)
        rms_db[i] = 20.0 * np.log10(rms + 1e-12)

    times = np.arange(n_frames) * (frame_ms / 1000.0)

    # end of first proper speech run
    need_speech = int(np.ceil(min_initial_speech_run_s / (frame_ms / 1000.0)))
    run = 0; end_idx = None
    for i in range(n_frames):
        if voiced[i]:
            run += 1
        else:
            if run >= need_speech:
                end_idx = i; break
            run = 0
    if end_idx is None:
        return []

    # long quiet pause
    need_pause = int(np.ceil(long_pause_s / (frame_ms / 1000.0)))
    run = 0; start_idx = None
    for i in range(end_idx, n_frames):
        quiet = (not voiced[i]) and (rms_db[i] <= rms_db_floor)
        if quiet:
            if start_idx is None: start_idx = i; run = 1
            else: run += 1
            if run >= need_pause:
                return [float(times[start_idx])]
        else:
            run = 0; start_idx = None
    return []


# ---------------------------
# Logo (template + auto-template)
# ---------------------------

def _probe_wh(path: str) -> Tuple[int, int]:
    _, out, _ = run([
        "ffprobe","-hide_banner","-loglevel","error",
        "-select_streams","v:0","-show_entries","stream=width,height",
        "-of","csv=s=,:p=0", path
    ])
    try:
        w, h = [int(x) for x in (out or "0,0").strip().split(",")]
    except Exception:
        w, h = 0, 0
    return w, h

def _read_frames_bgr(path: str, fps: float, scan_limit_s: float) -> Tuple[List[np.ndarray], int, int]:
    W, H = _probe_wh(path)
    if W <= 0 or H <= 0:
        return [], 0, 0
    cmd = [
        "ffmpeg","-hide_banner","-nostdin","-loglevel","error",
        "-t", str(scan_limit_s), "-i", path,
        "-vf", f"fps={fps}","-f","rawvideo","-pix_fmt","bgr24","-"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fsz = W * H * 3
    frames: List[np.ndarray] = []
    while True:
        buf = p.stdout.read(fsz) if p.stdout else None
        if not buf or len(buf) < fsz:
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3))
    p.wait()
    return frames, W, H

def auto_generate_logo_template(path: str, out_png: str, scan_limit_s: float = 120.0, fps: float = 10.0) -> bool:
    """Guess a static overlay region and save a template PNG."""
    frames, W, H = _read_frames_bgr(path, fps, scan_limit_s)
    if not frames:
        return False
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], axis=0).astype(np.float32)
    std = np.std(stack, axis=0)
    sal = 1.0 - (std - std.min()) / (np.ptp(std) + 1e-6)  # NumPy 2.x safe
    med = np.median(stack, axis=0).astype(np.uint8)
    thr = float(np.percentile(sal, 90))
    mask = (sal >= thr).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    crop = med[y:y+h, x:x+w]
    return bool(cv2.imwrite(out_png, crop))

def detect_logo_times(path: str, logo_path: Optional[str], scan_limit_s: float = 120.0, fps: float = 10.0, match_thresh: float = 0.78) -> List[float]:
    """Template-match provided logo image and return match times (s)."""
    if not logo_path:
        return []
    tpl = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        return []
    frames, _, _ = _read_frames_bgr(path, fps, scan_limit_s)
    hits: List[float] = []
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, _ = cv2.minMaxLoc(res)
        t = i / fps
        if mv >= match_thresh:
            hits.append(t)
    return _dedupe_times(_clamp_to_scan(hits, scan_limit_s))


# ---------------------------
# Keyframes
# ---------------------------

def list_keyframes(path: str, scan_limit_s: float = 180.0) -> List[float]:
    _, out, _ = run([
        "ffprobe","-hide_banner","-loglevel","error",
        "-read_intervals", f"%+{scan_limit_s}",
        "-select_streams","v:0",
        "-show_frames",
        "-show_entries","frame=pkt_pts_time,key_frame",
        "-of","csv", path
    ])
    kf = []
    for line in (out or "").splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 3 and parts[1] == "1":
            try: kf.append(float(parts[2]))
            except: pass
    return sorted(kf)

def snap_to_nearest_keyframe(t: float, keyframes: List[float], max_jump_s: float = 0.6) -> float:
    """Snap to nearest KF within max_jump_s; else return t."""
    if not keyframes:
        return t
    best = min(keyframes, key=lambda k: abs(k - t))
    return best if abs(best - t) <= max_jump_s else t
