import json, re, subprocess
from typing import Tuple

def run(cmd: list[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def ffprobe_json(path: str, entries: str) -> dict:
    rc, out, err = run([
        "ffprobe","-hide_banner","-loglevel","error",
        "-print_format","json","-show_entries", entries, path
    ])
    if rc != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    return json.loads(out)

def get_duration(path: str) -> float:
    return float(ffprobe_json(path, "format=duration")["format"]["duration"])

def cut_segment(path: str, out_mp4: str, start_s: float, end_s: float, reencode=False) -> None:
    dur = max(0.01, end_s - start_s)
    if not reencode:
        cmd = (["ffmpeg","-hide_banner","-y","-i",path,"-to",f"{dur}",
                "-c","copy","-avoid_negative_ts","make_zero", out_mp4]
               if start_s <= 0.05 else
               ["ffmpeg","-hide_banner","-y","-ss",f"{start_s}","-i",path,"-to",f"{dur}",
                "-c","copy","-avoid_negative_ts","make_zero", out_mp4])
        rc,_,err = run(cmd)
        if rc == 0: return
    rc,_,err = run([
        "ffmpeg","-hide_banner","-y",
        "-ss",f"{start_s}","-i",path,"-to",f"{dur}",
        "-c:v","libx264","-crf","18","-preset","veryfast",
        "-c:a","aac","-b:a","160k","-movflags","+faststart", out_mp4
    ])
    if rc != 0:
        raise RuntimeError(f"Cut failed: {err}")

def extract_thumbnail(path: str, out_jpg: str, search_window_s=5.0, width=1280) -> None:
    rc,_,err = run([
        "ffmpeg","-hide_banner","-y",
        "-ss","0","-t",str(search_window_s),"-i",path,
        "-vf",f"thumbnail,scale={width}:-1","-frames:v","1", out_jpg
    ])
    if rc != 0:
        run(["ffmpeg","-hide_banner","-y","-ss","3","-i",path,
             "-vf",f"scale={width}:-1","-frames:v","1", out_jpg])
