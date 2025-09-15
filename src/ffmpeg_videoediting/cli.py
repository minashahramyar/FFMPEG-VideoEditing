# # cli.py
# from __future__ import annotations

# import argparse, json, sys
# from pathlib import Path
# from typing import List, Tuple, Dict

# from .ffutil import get_duration, extract_thumbnail, cut_segment
# from .detect import (
#     first_chapter_from_metadata,
#     detect_scene_cuts, detect_black_segments, detect_silence_starts,
#     detect_speech_onset, detect_logo_times, auto_generate_logo_template,
#     list_keyframes, snap_to_prev_keyframe
# )

# def cluster_by_time(events:List[Tuple[str,float]],eps:float=1.0)->List[Dict]:
#     if not events: return []
#     ev=sorted(events,key=lambda x:x[1])
#     clusters=[]; cur={"t":ev[0][1],"labels":{ev[0][0]},"times":[ev[0][1]]}
#     for lbl,t in ev[1:]:
#         if t-cur["t"]<=eps:
#             cur["labels"].add(lbl); cur["times"].append(t)
#         else:
#             clusters.append(cur); cur={"t":t,"labels":{lbl},"times":[t]}
#     clusters.append(cur); return clusters

# def score_cluster(c:Dict,pmin:float,pmax:float,hmin:float)->float:
#     if c["t"]<hmin: return float("-inf")
#     score=0; labs=c["labels"]
#     if "logo" in labs: score+=4
#     if "speech" in labs: score+=3
#     if "silence_start" in labs: score+=2
#     if "scene" in labs: score+=1
#     if len(labs)>=2: score+=1
#     if pmin<=c["t"]<=pmax: score+=2
#     return score

# def pick_cluster(clusters:List[Dict],pmin:float,pmax:float,hmin:float)->Dict|None:
#     if not clusters: return None
#     scored=[(score_cluster(c,pmin,pmax,hmin),c) for c in clusters]
#     scored=[s for s in scored if s[0]!=float("-inf")]
#     if not scored: return sorted(clusters,key=lambda c:c["t"])[0]
#     scored.sort(key=lambda s:(-s[0],s[1]["t"])); return scored[0][1]

# def main()->None:
#     ap=argparse.ArgumentParser(description="Auto-detect first chapter and cut before logo motion.")
#     ap.add_argument("input"); ap.add_argument("-o","--outdir",default="output")
#     ap.add_argument("--scan-limit",type=float,default=120.0)
#     ap.add_argument("--min-first-end",type=float,default=35.0)
#     ap.add_argument("--max-first-end",type=float,default=90.0)
#     ap.add_argument("--scene-thresh",type=float,default=0.30)
#     ap.add_argument("--black-pic-th",type=float,default=0.98)
#     ap.add_argument("--black-min-dur",type=float,default=0.20)
#     ap.add_argument("--silence-db",type=float,default=-30.0)
#     ap.add_argument("--silence-min-dur",type=float,default=0.50)
#     ap.add_argument("--vad-aggr",type=int,default=2)
#     ap.add_argument("--vad-min-run",type=float,default=1.0)
#     ap.add_argument("--logo-template",help="Provide logo PNG")
#     ap.add_argument("--auto-logo-template",help="Path to save auto-generated logo PNG")
#     ap.add_argument("--logo-thresh",type=float,default=0.75)
#     ap.add_argument("--logo-margin",type=float,default=0.4)
#     ap.add_argument("--reencode-fallback",action="store_true")
#     ap.add_argument("--debug",action="store_true")
#     args=ap.parse_args()

#     inp=Path(args.input).resolve()
#     if not inp.exists(): print("Input not found",file=sys.stderr); sys.exit(2)
#     outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
#     base=inp.stem; thumb=outdir/f"{base}_first_chapter_thumb.jpg"
#     clip=outdir/f"{base}_first_chapter.mp4"

#     dur=get_duration(str(inp)); start_s=0.0; method="unknown"

#     # maybe auto-generate logo template
#     logo_path=args.logo_template
#     if not logo_path and args.auto_logo_template:
#         if auto_generate_logo_template(str(inp),args.auto_logo_template,scan_limit_s=args.scan_limit):
#             logo_path=args.auto_logo_template
#             if args.debug: print(json.dumps({"auto_logo_template":logo_path},indent=2))

#     # cues
#     scenes=detect_scene_cuts(str(inp),args.scene_thresh,args.scan_limit)
#     blacks=detect_black_segments(str(inp),args.black_pic_th,args.black_min_dur,args.scan_limit)
#     sils=detect_silence_starts(str(inp),args.silence_db,args.silence_min_dur,args.scan_limit)
#     speechs=detect_speech_onset(str(inp),args.scan_limit,args.vad_aggr,min_speech_run_s=args.vad_min_run)
#     logos=detect_logo_times(str(inp),logo_path,scan_limit_s=args.scan_limit,match_thresh=args.logo_thresh)

#     cands=[]
#     for t in scenes: cands.append(("scene",t))
#     for (bs,be) in blacks: cands.append(("black_end",be))
#     for t in sils: cands.append(("silence_start",t))
#     for t in speechs: cands.append(("speech",t))
#     for t in logos: cands.append(("logo",t))

#     clusters=cluster_by_time(cands)
#     if args.debug:
#         print(json.dumps({"clusters":[{"t":c["t"],"labels":list(c["labels"])} for c in clusters]},indent=2))

#     chosen=pick_cluster(clusters,args.min_first_end,args.max_first_end,25.0)
#     if chosen:
#         raw_t=min(chosen["t"],dur)
#         if "logo" in chosen["labels"]:
#             raw_t=max(0,raw_t-args.logo_margin); method="auto-logo"
#         else:
#             method="auto-"+"&".join(sorted(chosen["labels"]))
#         end_s=snap_to_prev_keyframe(raw_t,list_keyframes(str(inp)))
#     else:
#         end_s=min(dur,max(30.0,min(120.0,dur*0.06))); method="heuristic"

#     extract_thumbnail(str(inp),str(thumb),search_window_s=5.0,width=1280)
#     cut_segment(str(inp),str(clip),start_s,end_s,reencode=args.reencode_fallback)

#     print(json.dumps({"input":str(inp),"duration_s":dur,
#                       "method":method,"end_s":round(end_s,2),
#                       "thumbnail":str(thumb),"clip":str(clip)},indent=2))


# cli.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from .ffutil import get_duration, extract_thumbnail, cut_segment
from .detect import (
    first_chapter_from_metadata,
    detect_scene_cuts, detect_black_segments, detect_silence_starts,
    detect_speech_onset, detect_long_pause_after_speech,
    detect_logo_times, auto_generate_logo_template,
    list_keyframes, snap_to_nearest_keyframe,
)

# ---------------------------
# Clustering & scoring
# ---------------------------

def cluster_by_time(events: List[Tuple[str, float]], eps: float = 0.5) -> List[Dict]:
    """Group (label, t) events by proximity (<= eps s)."""
    if not events:
        return []
    ev = sorted(events, key=lambda x: x[1])
    clusters: List[Dict] = []
    cur = {"t": ev[0][1], "labels": {ev[0][0]}, "times": [ev[0][1]]}
    for lbl, t in ev[1:]:
        if t - cur["t"] <= eps:
            cur["labels"].add(lbl)
            cur["times"].append(t)
        else:
            clusters.append(cur)
            cur = {"t": t, "labels": {lbl}, "times": [t]}
    clusters.append(cur)
    return clusters

def score_cluster(c: Dict, pref_min: float, pref_max: float, hard_min: float) -> float:
    """
    Audio-first weighting:
      +6 long_pause   (>= N s, quiet)
      +4 speech       (onset/run)
      +2 silence_start
      +1 scene
      +1 logo
      +1 consensus (>=2 labels)
      +2 inside preferred window
      -inf if t < hard_min
      NEW: early bias inside window (earlier slightly better)
    """
    t = c["t"]
    if t < hard_min:
        return float("-inf")
    labs = c["labels"]
    score = 0.0
    if "long_pause" in labs:   score += 6.0
    if "speech" in labs:       score += 4.0
    if "silence_start" in labs: score += 2.0
    if "scene" in labs:        score += 1.0
    if "logo" in labs:         score += 1.0
    if len(labs) >= 2:         score += 1.0
    if pref_min <= t <= pref_max:
        score += 2.0
        mid = 0.5 * (pref_min + pref_max)
        score -= 0.04 * max(0.0, t - mid)  # earlier bias
    return score

def pick_cluster(clusters: List[Dict], pref_min: float, pref_max: float, hard_min: float = 25.0) -> Dict | None:
    if not clusters:
        return None
    scored = [(score_cluster(c, pref_min, pref_max, hard_min), c) for c in clusters]
    scored = [sc for sc in scored if sc[0] != float("-inf")]
    if not scored:
        return sorted(clusters, key=lambda c: c["t"])[0]
    scored.sort(key=lambda sc: (-sc[0], sc[1]["t"]))
    return scored[0][1]


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-detect first chapter (audio-first) and export clip + thumbnail.")
    ap.add_argument("input", help="Input video path")
    ap.add_argument("-o", "--outdir", default="output", help="Output directory")
    ap.add_argument("--output-name", help="Custom base name for outputs (no extension)")

    # Windows / bounds
    ap.add_argument("--scan-limit", type=float, default=120.0, help="Analyze first N seconds (default 120)")
    ap.add_argument("--min-first-end", type=float, default=40.0, help="Preferred window: min end (default 40s)")
    ap.add_argument("--max-first-end", type=float, default=65.0, help="Preferred window: max end (default 65s)")
    ap.add_argument("--hard-min", type=float, default=25.0, help="Reject candidates earlier than this (default 25s)")

    # Audio detectors
    ap.add_argument("--vad-aggr", type=int, default=3, help="WebRTC VAD aggressiveness 0-3 (default 3)")
    ap.add_argument("--vad-min-run", type=float, default=1.0, help="Min speech run (s) to count as onset (default 1.0)")
    ap.add_argument("--long-pause-seconds", type=float, default=2.0, help="Pause length to mark boundary (default 2s)")
    ap.add_argument("--long-pause-rms-floor", type=float, default=-50.0, help="RMS dBFS must be <= this (default -50)")

    # Visual detectors
    ap.add_argument("--scene-thresh", type=float, default=0.32, help="Scene change threshold (lower = more sensitive)")
    ap.add_argument("--black-pic-th", type=float, default=0.98, help="blackdetect pic_th")
    ap.add_argument("--black-min-dur", type=float, default=0.20, help="blackdetect min duration")

    # Logo (optional)
    ap.add_argument("--logo-template", help="Path to logo image (optional)")
    ap.add_argument("--auto-logo-template", help="If set, auto-generate logo PNG to this path")
    ap.add_argument("--logo-thresh", type=float, default=0.78, help="Logo match threshold")

    # Behavior
    ap.add_argument("--bias-seconds", type=float, default=0.0, help="Subtract N seconds from chosen time before snapping")
    ap.add_argument("--reencode-fallback", action="store_true", help="Re-encode if stream-copy can’t cut exactly")
    ap.add_argument("--debug", action="store_true", help="Print detector outputs and clusters")

    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base = (args.output_name or inp.stem).strip()
    thumb = outdir / f"{base}_first_chapter_thumb.jpg"
    clip  = outdir / f"{base}_first_chapter.mp4"

    duration = get_duration(str(inp))
    start_s = 0.0
    method = "unknown"

    # Use chapters if present
    chap = first_chapter_from_metadata(str(inp))
    if chap:
        end_s = min(chap[1], duration)
        method = "metadata"
    else:
        # Optional auto logo template
        logo_path = args.logo_template
        if not logo_path and args.auto_logo_template:
            if auto_generate_logo_template(str(inp), args.auto_logo_template, scan_limit_s=args.scan_limit):
                logo_path = args.auto_logo_template
                if args.debug:
                    print(json.dumps({"auto_logo_template": logo_path}, indent=2))

        # Collect cues (audio-first)
        long_pauses = detect_long_pause_after_speech(
            str(inp),
            scan_limit_s=args.scan_limit,
            vad_aggr=args.vad_aggr,
            frame_ms=30,
            min_initial_speech_run_s=3.0,
            long_pause_s=args.long_pause_seconds,
            rms_db_floor=args.long_pause_rms_floor,
        )
        speechs = detect_speech_onset(
            str(inp),
            scan_limit_s=args.scan_limit,
            vad_aggr=args.vad_aggr,
            frame_ms=30,
            min_speech_run_s=args.vad_min_run,
        )
        sils   = detect_silence_starts(str(inp), noise_db=-30.0, min_dur=0.50, scan_limit_s=args.scan_limit)
        scenes = detect_scene_cuts(str(inp), scene_thresh=args.scene_thresh, scan_limit_s=args.scan_limit)
        blacks = detect_black_segments(str(inp), pic_th=args.black_pic_th, dur_th=args.black_min_dur, scan_limit_s=args.scan_limit)
        logos  = detect_logo_times(str(inp), logo_path, scan_limit_s=args.scan_limit, match_thresh=args.logo_thresh)

        # Candidates
        cands: List[Tuple[str, float]] = []
        for t in long_pauses[:2]:   cands.append(("long_pause", t))
        for t in speechs[:10]:      cands.append(("speech", t))
        for t in sils[:10]:         cands.append(("silence_start", t))
        for t in scenes[:20]:       cands.append(("scene", t))
        for bs, be in blacks[:10]:  cands.append(("black_end", be))
        for t in logos[:10]:        cands.append(("logo", t))

        clusters = cluster_by_time(cands, eps=0.5)

        if args.debug:
            print(json.dumps({
                "long_pause": long_pauses,
                "speech": speechs[:10],
                "silences": sils[:10],
                "scenes": scenes[:20],
                "blacks_end": [be for _, be in blacks[:10]],
                "logos": logos[:10],
                "clusters": [
                    {"t": round(c["t"], 3), "labels": sorted(list(c["labels"])), "n": len(c["times"])}
                    for c in clusters
                ],
                "pref_window": [args.min_first_end, args.max_first_end],
                "hard_min": args.hard_min
            }, indent=2))

        chosen = pick_cluster(clusters, args.min_first_end, args.max_first_end, hard_min=args.hard_min)
        if chosen:
            raw_t = min(chosen["t"], duration)
            raw_t = max(0.0, raw_t - max(0.0, args.bias_seconds))  # optional earlier nudge

            keyframes = list_keyframes(str(inp), scan_limit_s=max(args.scan_limit, 180.0))

            if "long_pause" in chosen["labels"]:
                # NEVER push forward when using long pause: snap to PREVIOUS KF or use raw time
                prev_kf = max([k for k in keyframes if k <= raw_t], default=None)
                if prev_kf is not None and (raw_t - prev_kf) <= 0.8:
                    end_s = prev_kf
                else:
                    end_s = raw_t   # re-encode if needed
                method = "auto-long_pause"
            else:
                nearest = snap_to_nearest_keyframe(raw_t, keyframes, max_jump_s=0.6)
                end_s = nearest if abs(nearest - raw_t) <= 0.6 else raw_t
                method = "auto-" + "&".join(sorted(list(chosen["labels"])))
        else:
            end_s = min(duration, max(30.0, min(120.0, duration * 0.06)))
            method = "heuristic"

    # Outputs
    try:
        extract_thumbnail(str(inp), str(thumb), search_window_s=5.0, width=1280)
    except Exception as e:
        print(f"Warning: thumbnail extraction failed: {e}", file=sys.stderr)

    try:
        cut_segment(str(inp), str(clip), start_s, end_s, reencode=args.reencode_fallback)
    except Exception as e:
        print(f"Cut failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Report
    print(json.dumps({
        "input": str(inp),
        "duration_s": round(duration, 3),
        "first_chapter_detection": method,
        "first_chapter_start_s": round(start_s, 3),
        "first_chapter_end_s": round(end_s, 3),
        "thumbnail": str(thumb),
        "output_clip": str(clip),
    }, indent=2))



    # Sample code for running:

    # ffmpeg-videoediting "/Users/mina.yar/Downloads/SampleVideo/1.mp4" \
    # -o output \
    # --reencode-fallback \
    # --scan-limit 90 \
    # --min-first-end 40 \
    # --max-first-end 55 \
    # --long-pause-seconds 2 \
    # --long-pause-rms-floor -50 \
    # --vad-aggr 3 \
    # --bias-seconds 1.0 \
    # --debug

