#!/usr/bin/env python3
"""
biolum_controller.py — Bioluminescence Imaging Control Panel
Complete GUI for focus, settings, capture, and file browser.

Usage:
    python3 biolum_controller.py

Open in browser:
    http://localhost:5000          (Pi screen)
    http://<pi-hotspot-ip>:5000    (laptop or phone on Pi hotspot)
"""

from flask import Flask, Response, render_template_string, request, jsonify, send_file
from gpiozero import LED
from pathlib import Path
from datetime import datetime
import subprocess
import threading
import time
import os
import signal
import sys
import socket

app = Flask(__name__)

# ── config ────────────────────────────────────────────────────────────────────
LED_GPIO     = 27
BASE_DIR     = Path.home() / "Experiments"
PREVIEW_TMP  = "/tmp/liveview_preview.jpg"
PREVIEW_READ = "/tmp/thumb_liveview_preview.jpg"

# ── shared state ──────────────────────────────────────────────────────────────
_frame_lock      = threading.Lock()
_latest_frame    = b""
_streaming       = False
_light           = LED(LED_GPIO)
_capture_log     = []
_capture_lock    = threading.Lock()
_capture_done    = False
_stop_requested  = False
_last_day_img    = None
_last_biolum_img = None
_shutter_open    = False   # True while bulb shutter is open
_capture_phase   = "idle"  # idle | day | waiting | biolum | done

# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(msg)
    with _capture_lock:
        _capture_log.append(msg)
        if len(_capture_log) > 100:
            _capture_log.pop(0)

def run(cmd):
    log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        log(result.stdout.strip())
    if result.returncode != 0 and result.stderr.strip():
        log("ERR: " + result.stderr.strip())
    return result

def gp(*args):
    return run(["gphoto2"] + list(args))

def check_stop():
    """Returns True if stop was requested — call after each major step."""
    return _stop_requested

def wait_for_file(path, after_time, timeout=120, poll_interval=2):
    """Wait until file exists, is non-empty, and was modified after after_time."""
    elapsed = 0
    while elapsed < timeout:
        if (path.exists()
                and path.stat().st_size > 0
                and path.stat().st_mtime > after_time):
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    return False

def find_newest_file(folder, glob_pattern, after_time):
    """Find the newest file matching pattern that was modified after after_time."""
    matches = list(folder.glob(glob_pattern))
    fresh = [f for f in matches if f.stat().st_mtime > after_time]
    return max(fresh, key=lambda f: f.stat().st_mtime) if fresh else None

def make_folder(experiment, sample):
    today = datetime.now().strftime("%Y.%m.%d")
    name = f"{today}_{experiment}"
    folder = BASE_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def free_camera():
    """Kill any process holding the camera USB device, then wait for it to settle."""
    subprocess.run(["pkill", "-f", "gphoto2"], capture_output=True)
    subprocess.run(["pkill", "-f", "gvfs-gphoto2"], capture_output=True)
    time.sleep(1)

def set_common_settings():
    free_camera()  # ensure no stale gphoto2 or gvfs process holds the device
    gp("--set-config", "imagequality=5")
    gp("--set-config", "highisonr=0")   # disable high ISO NR
    gp("--set-config", "longexpnr=0")   # disable long exposure NR
    gp("--set-config", "capturetarget=0")

def get_pi_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"

# ── preview loop ──────────────────────────────────────────────────────────────

def preview_loop():
    global _latest_frame, _streaming
    while _streaming:
        try:
            result = subprocess.run(
                ["gphoto2", "--capture-preview",
                 "--filename", PREVIEW_TMP, "--force-overwrite"],
                check=False, capture_output=True, timeout=5
            )
            if result.returncode == 0:
                with open(PREVIEW_READ, "rb") as f:
                    data = f.read()
                with _frame_lock:
                    _latest_frame = data
            else:
                time.sleep(1)
        except subprocess.TimeoutExpired:
            time.sleep(1)
        except FileNotFoundError:
            time.sleep(0.5)
        except Exception as e:
            log(f"Preview error: {e}")
            time.sleep(1)

def start_preview():
    global _streaming
    if _streaming:
        return
    _streaming = True
    subprocess.run(["gphoto2", "--set-config", "viewfinder=1"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    threading.Thread(target=preview_loop, daemon=True).start()

def stop_preview():
    global _streaming
    _streaming = False
    time.sleep(0.5)
    subprocess.run(["gphoto2", "--set-config", "viewfinder=0"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_stream():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame and _streaming:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.1)

# ── capture logic ─────────────────────────────────────────────────────────────

def run_day(settings):
    global _last_day_img, _last_biolum_img, _capture_done, _stop_requested
    _last_biolum_img = None  # clear so day preview shows correctly
    _capture_done = False
    _stop_requested = False
    folder = make_folder(settings["experiment"], settings["sample"])
    set_common_settings()
    log("\n▶ DAY IMAGE")
    _light.on()
    log("LED ON")
    time.sleep(2)
    gp("--set-config", f"iso={settings['day_iso']}")
    gp("--set-config", f"shutterspeed={settings['day_shutter']}")
    sample = settings["sample"] or "sample"
    ts = datetime.now().strftime("%H%M%S")   # recorded just before capture
    day_file = str(folder / f"{sample}_day_{ts}.%C")
    run(["gphoto2", "--force-overwrite",
         "--capture-image-and-download", "--filename", day_file])
    _light.off()
    log("LED OFF")
    if check_stop():
        _capture_done = True
        log("⚠ Capture stopped by user.")
        return
    day_jpeg = folder / f"{sample}_day_{ts}.jpg"
    if day_jpeg.exists():
        _last_day_img = str(day_jpeg)
        log(f"✓ Day saved: {day_jpeg.name}")
    else:
        # fallback: find newest day file
        day_jpeg = find_newest_file(folder, f"{sample}_day_*.jpg", 0)
        if day_jpeg:
            _last_day_img = str(day_jpeg)
            log(f"✓ Day saved (fallback): {day_jpeg.name}")
        else:
            log("ERR: Day JPEG not found")
    _capture_done = True
    log("✓ Done!")

def run_biolum(settings):
    global _last_biolum_img, _capture_done, _shutter_open, _stop_requested
    _capture_done = False
    _stop_requested = False
    folder = make_folder(settings["experiment"], settings["sample"])
    set_common_settings()
    biolum_exposure = int(settings["biolum_exposure"])
    sample = settings["sample"] or "sample"
    ts = datetime.now().strftime("%H%M%S")
    log(f"\n▶ BIOLUM IMAGE — {biolum_exposure}s exposure")
    gp("--set-config", f"iso={settings['biolum_iso']}")
    gp("--set-config", "shutterspeed=53")  # bulb
    capture_start_time = time.time()  # record before shutter opens
    gphoto2_wait = biolum_exposure + 10
    poll_timeout = biolum_exposure + 10
    # predict the timestamp when the file will be written
    predicted_write = capture_start_time + biolum_exposure + gphoto2_wait
    ts_biolum = datetime.fromtimestamp(predicted_write).strftime("%H%M%S")
    biolum_file = str(folder / f"{sample}_biolum_{biolum_exposure}sec_{ts_biolum}.%C")
    _shutter_open = True   # signal to client: start countdown NOW
    log("SHUTTER_OPEN")
    run(["gphoto2", "--force-overwrite",
         f"--filename={biolum_file}",
         "--set-config", "bulb=1",
         "--wait-event", f"{biolum_exposure}s",
         "--set-config", "bulb=0",
         f"--wait-event-and-download={gphoto2_wait}s"])
    _shutter_open = False
    log("SHUTTER_CLOSED")
    if check_stop():
        _capture_done = True
        log("⚠ Capture stopped by user.")
        return
    biolum_jpeg = folder / f"{sample}_biolum_{biolum_exposure}sec_{ts_biolum}.jpg"
    biolum_nef  = folder / f"{sample}_biolum_{biolum_exposure}sec_{ts_biolum}.nef"
    log(f"Checking for {biolum_jpeg.name}...")
    if biolum_jpeg.exists():
        _last_biolum_img = str(biolum_jpeg)
        log(f"✓ Biolum JPEG saved: {biolum_jpeg.name}")
    else:
        fallback = find_newest_file(folder, f"{sample}_biolum_{biolum_exposure}sec_*.jpg", capture_start_time)
        if fallback:
            _last_biolum_img = str(fallback)
            log(f"✓ Biolum JPEG saved (fallback): {fallback.name}")
        else:
            log(f"ERR: Biolum JPEG not found")
    if biolum_nef.exists():
        log(f"✓ Biolum NEF saved: {biolum_nef.name}")
    else:
        fallback_nef = find_newest_file(folder, f"{sample}_biolum_{biolum_exposure}sec_*.nef", capture_start_time)
        if fallback_nef:
            log(f"✓ Biolum NEF saved (fallback): {fallback_nef.name}")
        else:
            log(f"ERR: Biolum NEF not found")
    _capture_done = True
    log("✓ Done!")

def run_both(settings):
    global _last_day_img, _last_biolum_img, _capture_done, _capture_phase, _stop_requested
    _capture_phase = "idle"
    _capture_done = False
    _stop_requested = False
    folder = make_folder(settings["experiment"], settings["sample"])
    sample = settings["sample"] or "sample"
    set_common_settings()

    # --- DAY ---
    _capture_phase = "day"
    log("\n▶ DAY IMAGE")
    _light.on()
    log("LED ON")
    time.sleep(2)
    gp("--set-config", f"iso={settings['day_iso']}")
    gp("--set-config", f"shutterspeed={settings['day_shutter']}")
    ts_day = datetime.now().strftime("%H%M%S")
    day_file = str(folder / f"{sample}_day_{ts_day}.%C")
    run(["gphoto2", "--force-overwrite",
         "--capture-image-and-download", "--filename", day_file])
    _light.off()
    log("LED OFF")
    day_jpeg = folder / f"{sample}_day_{ts_day}.jpg"
    if day_jpeg.exists():
        _last_day_img = str(day_jpeg)
        log(f"✓ Day saved: {day_jpeg.name}")
    else:
        fallback = find_newest_file(folder, f"{sample}_day_*.jpg", 0)
        if fallback:
            _last_day_img = str(fallback)
            log(f"✓ Day saved (fallback): {fallback.name}")
        else:
            log("ERR: Day JPEG not found")

    _capture_phase = "waiting"
    if check_stop():
        _capture_done = True
        log("⚠ Capture stopped by user.")
        return
    log("DARK_WAIT_START")
    log("Waiting 5s for LEDs to turn off...")
    time.sleep(5)
    log("DARK_WAIT_DONE")

    if check_stop():
        _capture_done = True
        log("⚠ Capture stopped by user.")
        return
    _capture_phase = "biolum"
    biolum_exposure = int(settings["biolum_exposure"])
    ts_biolum = datetime.now().strftime("%H%M%S")
    log(f"\n▶ BIOLUM IMAGE — {biolum_exposure}s exposure")
    gp("--set-config", f"iso={settings['biolum_iso']}")
    gp("--set-config", "shutterspeed=53")
    capture_start_time = time.time()
    gphoto2_wait = biolum_exposure + 10
    poll_timeout = biolum_exposure + 10
    predicted_write = capture_start_time + biolum_exposure + gphoto2_wait
    ts_biolum = datetime.fromtimestamp(predicted_write).strftime("%H%M%S")
    biolum_file = str(folder / f"{sample}_biolum_{biolum_exposure}sec_{ts_biolum}.%C")
    _shutter_open = True
    log("SHUTTER_OPEN")
    run(["gphoto2", "--force-overwrite",
         f"--filename={biolum_file}",
         "--set-config", "bulb=1",
         "--wait-event", f"{biolum_exposure}s",
         "--set-config", "bulb=0",
         f"--wait-event-and-download={gphoto2_wait}s"])
    _shutter_open = False
    log("SHUTTER_CLOSED")
    if check_stop():
        _capture_done = True
        log("⚠ Capture stopped by user.")
        return
    biolum_jpeg = folder / f"{sample}_biolum_{biolum_exposure}sec_{ts_biolum}.jpg"
    biolum_nef  = folder / f"{sample}_biolum_{biolum_exposure}sec_{ts_biolum}.nef"
    log(f"Checking for {biolum_jpeg.name}...")
    if biolum_jpeg.exists():
        _last_biolum_img = str(biolum_jpeg)
        log(f"✓ Biolum JPEG saved: {biolum_jpeg.name}")
    else:
        fallback = find_newest_file(folder, f"{sample}_biolum_{biolum_exposure}sec_*.jpg", capture_start_time)
        if fallback:
            _last_biolum_img = str(fallback)
            log(f"✓ Biolum JPEG saved (fallback): {fallback.name}")
        else:
            log(f"ERR: Biolum JPEG not found")
    if biolum_nef.exists():
        log(f"✓ Biolum NEF saved: {biolum_nef.name}")
    else:
        fallback_nef = find_newest_file(folder, f"{sample}_biolum_{biolum_exposure}sec_*.nef", capture_start_time)
        if fallback_nef:
            log(f"✓ Biolum NEF saved (fallback): {fallback_nef.name}")
        else:
            log(f"ERR: Biolum NEF not found")

    _capture_phase = "done"
    _capture_done = True
    log("✓ Done!")

# ── file browser helpers ──────────────────────────────────────────────────────

def get_experiments():
    """Return list of experiment folders, files grouped by stem (jpg+nef pairs)."""
    experiments = []
    if not BASE_DIR.exists():
        return experiments
    for folder in sorted(BASE_DIR.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        # group by stem
        stems = {}
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".nef", ".raw"):
                stem = f.stem
                if stem not in stems:
                    stems[stem] = {"stem": stem, "jpg": None, "nef": None,
                                   "jpg_path": None, "nef_path": None,
                                   "jpg_size": None, "nef_size": None}
                if f.suffix.lower() in (".jpg", ".jpeg"):
                    stems[stem]["jpg"] = f.name
                    stems[stem]["jpg_path"] = str(f)
                    stems[stem]["jpg_size"] = f"{f.stat().st_size / 1024 / 1024:.1f} MB"
                else:
                    stems[stem]["nef"] = f.name
                    stems[stem]["nef_path"] = str(f)
                    stems[stem]["nef_size"] = f"{f.stat().st_size / 1024 / 1024:.1f} MB"
        pairs = list(stems.values())
        experiments.append({
            "name": folder.name,
            "path": str(folder),
            "pairs": pairs,
            "count": len(pairs)
        })
    return experiments

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Biolum Controller</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:      #070a0e;
      --panel:   #0d1117;
      --border:  #1a2333;
      --border2: #243044;
      --accent:  #4af0c4;
      --amber:   #f0a84a;
      --warn:    #f05a4a;
      --text:    #b8cce0;
      --muted:   #3d5068;
      --mono:    'DM Mono', monospace;
      --sans:    'DM Sans', sans-serif;
    }

    html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; }

    /* ── tabs ── */
    .tab-bar {
      display: flex;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      padding: 0 20px;
      gap: 0;
      height: 44px;
      align-items: flex-end;
    }

    .tab {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      padding: 0 18px;
      height: 36px;
      display: flex;
      align-items: center;
      cursor: pointer;
      color: var(--text);
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      user-select: none;
    }

    .tab:hover { color: var(--accent); }
    .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

    .tab-content { display: none; height: calc(100vh - 44px); }
    .tab-content.active { display: flex; }

    /* ── CAPTURE TAB layout ── */
    #tab-capture {
      flex-direction: row;
    }

    .sidebar {
      width: 320px;
      flex-shrink: 0;
      border-right: 1px solid var(--border);
      background: var(--panel);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }

    .sidebar-inner {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex: 1;
    }

    .main-panel {
      flex: 1;
      display: grid;
      grid-template-rows: 1fr 110px;
      overflow: hidden;
      min-width: 0;
    }

    /* ── cards ── */
    .card { border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
    .card-header {
      padding: 6px 12px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.18em;
      color: var(--text);
      text-transform: uppercase;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.025);
    }
    .card-body { padding: 10px 12px; }

    .field { margin-bottom: 8px; }
    .field:last-child { margin-bottom: 0; }

    label {
      display: block;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.1em;
      color: var(--text);
      text-transform: uppercase;
      margin-bottom: 3px;
    }

    input[type="text"], input[type="number"] {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border2);
      border-radius: 3px;
      color: var(--text);
      font-family: var(--mono);
      font-size: 13px;
      padding: 6px 10px;
      outline: none;
      transition: border-color 0.2s;
    }
    input:focus { border-color: var(--accent); }

    .row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

    /* LED toggle */
    .led-row { display: flex; align-items: center; justify-content: space-between; padding: 4px 0; }
    .led-label { font-family: var(--mono); font-size: 11px; color: var(--text); }
    .toggle { position: relative; width: 38px; height: 20px; cursor: pointer; }
    .toggle input { display: none; }
    .toggle-track { position: absolute; inset: 0; background: var(--border2); border-radius: 10px; transition: background 0.2s; }
    .toggle-thumb { position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: var(--muted); transition: all 0.2s; }
    .toggle input:checked ~ .toggle-track { background: rgba(74,240,196,0.25); }
    .toggle input:checked ~ .toggle-thumb { left: 20px; background: var(--accent); box-shadow: 0 0 8px var(--accent); }

    /* buttons */
    .btn {
      width: 100%;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 9px;
      border-radius: 3px;
      border: 1px solid;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn + .btn { margin-top: 5px; }
    .btn:disabled { opacity: 0.3; cursor: not-allowed; }

    .btn-primary   { background: var(--accent); border-color: var(--accent); color: #000; font-weight: 600; }
    .btn-primary:hover:not(:disabled) { background: #6ff5d0; }

    .btn-ghost { background: transparent; border-color: var(--border2); color: var(--text); }
    .btn-ghost:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }

    .btn-amber { background: rgba(240,168,74,0.08); border-color: var(--amber); color: var(--amber); }
    .btn-amber:hover:not(:disabled) { background: rgba(240,168,74,0.18); }

    .btn-teal { background: rgba(74,240,196,0.07); border-color: var(--accent); color: var(--accent); }
    .btn-teal:hover:not(:disabled) { background: rgba(74,240,196,0.15); }

    .btn-red { background: transparent; border-color: var(--warn); color: var(--warn); }
    .btn-red:hover:not(:disabled) { background: rgba(240,90,74,0.1); }
    .btn-stop-active {
      background: rgba(240,90,74,0.15) !important;
      border-color: var(--warn) !important;
      color: var(--warn) !important;
    }

    /* phase steps */
    .phase-row { display: flex; gap: 5px; padding: 10px 0 4px; }
    .phase-pip {
      flex: 1;
      font-family: var(--mono);
      font-size: 9px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      text-align: center;
      padding: 4px 2px;
      border: 1px solid var(--border);
      border-radius: 2px;
      color: var(--muted);
      transition: all 0.3s;
    }
    .phase-pip.active { color: var(--accent); border-color: var(--accent); background: rgba(74,240,196,0.05); }
    .phase-pip.done   { color: var(--muted); text-decoration: line-through; }

    /* ── preview ── */
    .preview-area {
      position: relative;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      cursor: zoom-in;
    }

    .preview-area img {
      transition: transform 0.2s ease;
      will-change: transform;
    }

    /* when showing two images side by side */
    .preview-area.dual { gap: 4px; }
    .preview-single, .preview-dual-img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    .preview-dual-img {
      max-width: 50%;
      transition: transform 0.2s ease;
      will-change: transform;
      cursor: zoom-in;
    }

    .corner { position: absolute; width: 18px; height: 18px; opacity: 0.4; }
    .c-tl { top:12px; left:12px; border-top:2px solid var(--accent); border-left:2px solid var(--accent); }
    .c-tr { top:12px; right:12px; border-top:2px solid var(--accent); border-right:2px solid var(--accent); }
    .c-bl { bottom:12px; left:12px; border-bottom:2px solid var(--accent); border-left:2px solid var(--accent); }
    .c-br { bottom:12px; right:12px; border-bottom:2px solid var(--accent); border-right:2px solid var(--accent); }

    .crosshair { position: absolute; pointer-events: none; opacity: 0.15; }
    .crosshair::before { content:''; position:absolute; width:1px; height:40px; background:var(--accent); top:-20px; left:0; }
    .crosshair::after  { content:''; position:absolute; height:1px; width:40px; background:var(--accent); left:-20px; top:0; }

    .preview-tag {
      position: absolute;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.12em;
      color: #ffffff;
      background: rgba(7,10,14,0.92);
      padding: 4px 12px;
      border-radius: 3px;
      text-transform: uppercase;
      pointer-events: none;
      font-weight: 600;
      text-shadow: 0 0 8px rgba(255,255,255,0.3);
    }
    .preview-tag.top-center { top:10px; left:50%; transform:translateX(-50%); }
    .preview-tag.top-left   { top:10px; left:10px; }
    .preview-tag.top-right  { top:10px; right:10px; }

    /* countdown overlay */
    .countdown-wrap {
      position: absolute;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      display: none;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      pointer-events: none;
    }
    .countdown-wrap.visible { display: flex; }

    .countdown-num {
      font-family: var(--mono);
      font-size: 32px;
      font-weight: 500;
      color: var(--accent);
      text-shadow: 0 0 24px var(--accent);
      letter-spacing: 0.05em;
      min-width: 100px;
      text-align: center;
    }

    .countdown-bar-wrap {
      width: 200px;
      height: 3px;
      background: var(--border2);
      border-radius: 2px;
      overflow: hidden;
    }
    .countdown-bar-fill {
      height: 100%;
      background: var(--accent);
      width: 100%;
      transform-origin: left;
      transition: transform 1s linear;
    }

    .countdown-label {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.15em;
      color: #ffffff;
      text-transform: uppercase;
      font-weight: 600;
      text-shadow: 0 0 8px rgba(255,255,255,0.3);
    }

    /* ── log ── */
    .log-panel {
      border-top: 1px solid var(--border);
      background: var(--panel);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .log-header {
      padding: 5px 14px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.15em;
      color: var(--muted);
      text-transform: uppercase;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .log-body {
      flex: 1;
      overflow-y: auto;
      padding: 4px 14px;
      font-family: var(--mono);
      font-size: 11px;
      line-height: 1.6;
    }
    .log-ok   { color: var(--accent); }
    .log-warn { color: var(--amber); }
    .log-err  { color: var(--warn); }
    .log-line { color: var(--text); }

    /* ── BROWSER TAB layout ── */
    #tab-browser {
      flex-direction: column;
      overflow: hidden;
    }

    .browser-layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      flex: 1;
      overflow: hidden;
    }

    .file-tree {
      border-right: 1px solid var(--border);
      background: var(--panel);
      overflow-y: auto;
      padding: 10px;
    }

    .exp-folder {
      margin-bottom: 8px;
      border: 1px solid var(--border);
      border-radius: 3px;
      overflow: hidden;
    }

    .exp-folder-header {
      padding: 7px 10px;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text);
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255,255,255,0.02);
      user-select: none;
    }
    .exp-folder-header:hover { background: rgba(74,240,196,0.05); }

    .exp-folder-count {
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.05em;
    }

    .exp-folder-files { display: none; }
    .exp-folder-files.open { display: block; }

    .file-item {
      padding: 8px 12px 8px 20px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--text);
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-top: 1px solid var(--border);
      transition: all 0.15s;
      gap: 8px;
    }
    .file-item-name {
      flex: 1;
      word-break: break-word;
      line-height: 1.5;
    }
    .file-item:hover { color: var(--text); background: rgba(255,255,255,0.04); }
    .file-item.selected { color: var(--accent); background: rgba(74,240,196,0.07); }

    .file-size { font-size: 11px; color: var(--muted); }

    .file-actions {
      display: none;
      gap: 6px;
      align-items: center;
    }
    .file-item:hover .file-actions { display: flex; }

    .file-action-btn {
      background: transparent;
      border: none;
      cursor: pointer;
      font-size: 16px;
      padding: 3px 5px;
      border-radius: 3px;
      line-height: 1;
      opacity: 0.7;
      transition: opacity 0.15s, background 0.15s;
    }
    .file-action-btn:hover { opacity: 1; background: rgba(255,255,255,0.1); }
    .file-action-btn.rename { color: var(--accent); }
    .file-action-btn.delete { color: var(--warn); }

    .file-pair-sizes {
      font-size: 11px;
      color: var(--muted);
      display: flex;
      gap: 8px;
    }

    .file-preview-panel {
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .file-preview-img-wrap {
      flex: 1;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .file-preview-img-wrap img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      transition: transform 0.2s ease;
      will-change: transform;
      cursor: zoom-in;
    }

    .file-preview-toolbar {
      padding: 10px 16px;
      border-top: 1px solid var(--border);
      background: var(--panel);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .file-info {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
    }

    .btn-download {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 7px 16px;
      border-radius: 3px;
      border: 1px solid var(--accent);
      color: var(--accent);
      background: transparent;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
      transition: all 0.15s;
    }
    .btn-download:hover { background: rgba(74,240,196,0.1); }

    .no-selection {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    /* ── network info badge ── */
    .net-badge {
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.08em;
      padding: 0 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .net-badge span { color: var(--accent); }

    /* glow dot */
    .glow-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--accent);
      animation: glow 2s ease-in-out infinite;
      flex-shrink: 0;
    }
    @keyframes glow {
      0%,100% { box-shadow: 0 0 4px var(--accent); opacity:1; }
      50%      { box-shadow: 0 0 12px var(--accent); opacity:0.5; }
    }
  </style>
</head>
<body>

<!-- tab bar -->
<div class="tab-bar">
  <div class="glow-dot" style="margin-right:8px; margin-bottom:14px;"></div>
  <div style="font-family:var(--mono);font-size:12px;letter-spacing:0.15em;color:var(--accent);text-transform:uppercase;margin-right:20px;display:flex;align-items:center;padding-bottom:2px;">
    Biolum
  </div>
  <div class="tab active" onclick="switchTab('capture')">◎ Capture</div>
  <div class="tab" onclick="switchTab('browser')">⊞ Files</div>
  <div class="tab" onclick="switchTab('settings')">⚙ Settings</div>
  <div class="net-badge" style="margin-left:auto;">
    Access from network: <span id="net-ip">loading...</span>:5000
  </div>
</div>

<!-- ══ CAPTURE TAB ══ -->
<div class="tab-content active" id="tab-capture">

  <aside class="sidebar">
    <div class="sidebar-inner">

      <!-- phase -->
      <div class="phase-row">
        <div class="phase-pip active" id="pip-focus">Focus</div>
        <div class="phase-pip" id="pip-capture">Capture</div>
      </div>

      <!-- experiment -->
      <div class="card">
        <div class="card-body">
          <div class="field">
            <label>Experiment Name</label>
            <input type="text" id="exp-name" value="biolum_test">
          </div>
          <div class="field">
            <label>Sample Name</label>
            <input type="text" id="sample-name" placeholder="e.g. plate_01">
          </div>
        </div>
      </div>

      <!-- day -->
      <div class="card">
        <div class="card-header">Day Image Settings</div>
        <div class="card-body">
          <div class="row-2">
            <div class="field"><label>ISO</label><input type="number" id="day-iso" value="100" min="100" max="6400" step="100"></div>
            <div class="field"><label>Exposure (s)</label><input type="text" id="day-shutter" value="1/80"></div>
          </div>
        </div>
      </div>

      <!-- biolum -->
      <div class="card">
        <div class="card-header">Biolum Image Settings</div>
        <div class="card-body">
          <div class="row-2">
            <div class="field"><label>ISO</label><input type="number" id="biolum-iso" value="6400" min="100" max="6400" step="100"></div>
            <div class="field"><label>Exposure (s)</label><input type="number" id="biolum-exp" value="30"></div>
          </div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:6px;letter-spacing:0.05em;">
            Buffer wait = exposure + 10s &nbsp;·&nbsp; NR disabled
          </div>
        </div>
      </div>

      <!-- LED -->
      <div class="card">
        <div class="card-body">
          <div class="led-row">
            <span class="led-label">LED Power</span>
            <label class="toggle">
              <input type="checkbox" id="led-toggle" onchange="toggleLED()">
              <div class="toggle-track"></div>
              <div class="toggle-thumb"></div>
            </label>
          </div>
        </div>
      </div>

      <!-- actions -->
      <div style="margin-top:auto; display:flex; flex-direction:column; gap:0;">
        <button class="btn btn-ghost" id="btn-focus" onclick="startFocus()">◎ Adjust Focus</button>
        <button class="btn btn-primary" id="btn-focus-ok" onclick="focusOk()" style="display:none">✓ Focus OK</button>
        <div style="height:8px;"></div>
        <button class="btn btn-amber" id="btn-day" onclick="captureDay()" disabled>☀ Take Day Photo</button>
        <button class="btn btn-teal" id="btn-biolum" onclick="captureBiolum()" disabled>◉ Take Biolum Photo</button>
        <button class="btn btn-primary" id="btn-both" onclick="captureBoth()" disabled>▶ Full Sequence</button>
        <button class="btn btn-warn" id="btn-stop" onclick="stopCapture()" style="display:none">■ Stop Capture</button>

      </div>

    </div>
  </aside>

  <div class="main-panel">

    <!-- preview -->
    <div class="preview-area" id="preview-area">
      <img id="preview-img" src="" alt="Preview" class="preview-single">
      <div class="crosshair" id="crosshair"></div>
      <div class="corner c-tl"></div>
      <div class="corner c-tr"></div>
      <div class="corner c-bl"></div>
      <div class="corner c-br"></div>
      <div class="preview-tag top-center" id="preview-label">IDLE</div>

      <div class="countdown-wrap" id="countdown-wrap">
        <div class="countdown-label" id="countdown-label">Exposure</div>
        <div class="countdown-num" id="countdown-num">—</div>
        <div class="countdown-bar-wrap">
          <div class="countdown-bar-fill" id="countdown-bar"></div>
        </div>
      </div>
    </div>

    <!-- log -->
    <div class="log-panel">
      <div class="log-header">System Log</div>
      <div class="log-body" id="log-body"><span class="log-ok">Ready.</span></div>
    </div>

  </div>
</div>

<!-- ══ SETTINGS TAB ══ -->
<div class="tab-content" id="tab-settings" style="flex-direction:column;padding:24px;gap:20px;overflow-y:auto;">

  <div style="max-width:500px;display:flex;flex-direction:column;gap:16px;">

    <!-- Clock -->
    <div class="card">
      <div class="card-header">System Clock</div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:0.08em;margin-bottom:3px;">CURRENT PI TIME</div>
            <div style="font-family:var(--mono);font-size:18px;color:var(--accent);" id="pi-clock">—</div>
          </div>
          <div>
            <div style="font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:0.08em;margin-bottom:3px;">BROWSER TIME</div>
            <div style="font-family:var(--mono);font-size:18px;color:var(--text);" id="browser-clock">—</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.6;">
          If Pi time is wrong, click Sync to set it from your browser's clock. No internet needed.
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="btn btn-primary" style="width:auto;padding:0 20px;" onclick="syncClock()">↻ Sync Pi Clock</button>
          <span style="font-family:var(--mono);font-size:11px;color:var(--accent);" id="clock-sync-msg"></span>
        </div>
      </div>
    </div>

  </div>
</div>

<!-- ══ FILES TAB ══ -->
<div class="tab-content" id="tab-browser">
  <div class="browser-layout">

    <div class="file-tree" id="file-tree">
      <div style="font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
        Experiments
      </div>
      <div id="folder-list">Loading...</div>
    </div>

    <div class="file-preview-panel">
      <div class="file-preview-img-wrap" id="file-preview-wrap">
        <div class="no-selection" id="no-selection">Select a file to preview</div>
        <img id="file-preview-img" src="" style="display:none" alt="File preview">
      </div>
      <div class="file-preview-toolbar">
        <div class="file-info" id="file-info">—</div>
        <a id="download-link" class="btn-download" href="#" download style="display:none">↓ Download</a>
      </div>
    </div>

  </div>
</div>

<script>
  // ── tab switching ──
  function switchTab(name) {
    document.querySelectorAll('.tab').forEach((t,i) => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    const tabs = ['capture','browser','settings'];
    const idx = tabs.indexOf(name);
    document.querySelectorAll('.tab')[idx].classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
    if (name === 'browser') loadFileTree();
    if (name === 'settings') startClockDisplay();
  }

  // ── clock ──
  let clockTimer = null;
  let selectedFilePath = null;
  let selectedFileExt = null;

  async function startClockDisplay() {
    updateClocks();
    if (clockTimer) clearInterval(clockTimer);
    clockTimer = setInterval(updateClocks, 1000);
  }

  async function updateClocks() {
    document.getElementById('browser-clock').textContent =
      new Date().toLocaleTimeString('en-GB', {hour12:false}) + ' ' +
      new Date().toLocaleDateString('en-GB');
    try {
      const r = await fetch('/clock');
      const d = await r.json();
      document.getElementById('pi-clock').textContent = d.time + ' ' + d.date;
    } catch(e) {}
  }

  async function syncClock() {
    const msg = document.getElementById('clock-sync-msg');
    msg.textContent = 'Syncing...';
    const now = new Date();
    const iso = now.toISOString();
    try {
      const r = await fetch('/clock/set', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({iso})
      });
      const d = await r.json();
      msg.style.color = 'var(--accent)';
      msg.textContent = d.ok ? '✓ Synced!' : '✗ Failed: ' + d.error;
      updateClocks();
    } catch(e) {
      msg.style.color = 'var(--warn)';
      msg.textContent = '✗ Error';
    }
    setTimeout(() => { msg.textContent = ''; }, 4000);
  }

  // ── file operations ──
  function setSelectedFile(path, name, ext) {
    selectedFilePath = path;
    selectedFileExt = ext;
    const input = document.getElementById('rename-input');
    if (input) input.value = name.replace(/[.][^.]+$/, '');
  }

  async function renameSelected() {
    if (!selectedFilePath) {
      showFileOpMsg('Select a file in the Files tab first', 'var(--warn)');
      return;
    }
    const newName = document.getElementById('rename-input').value.trim();
    if (!newName) { showFileOpMsg('Enter a new filename', 'var(--warn)'); return; }
    const r = await fetch('/file/rename', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({path: selectedFilePath, new_name: newName})
    });
    const d = await r.json();
    if (d.ok) {
      showFileOpMsg('✓ Renamed to ' + d.new_name, 'var(--accent)');
      selectedFilePath = d.new_path;
      loadFileTree();
    } else {
      showFileOpMsg('✗ ' + d.error, 'var(--warn)');
    }
  }

  async function deleteSelected() {
    if (!selectedFilePath) {
      showFileOpMsg('Select a file in the Files tab first', 'var(--warn)');
      return;
    }
    if (!confirm('Delete ' + selectedFilePath.split('/').pop() + '? This cannot be undone.')) return;
    const r = await fetch('/file/delete', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({path: selectedFilePath})
    });
    const d = await r.json();
    if (d.ok) {
      showFileOpMsg('✓ Deleted', 'var(--accent)');
      selectedFilePath = null;
      loadFileTree();
      document.getElementById('file-preview-img').src = '';
      document.getElementById('no-selection').style.display = 'flex';
      document.getElementById('no-selection').textContent = 'Select a file to preview';
      document.getElementById('file-preview-img').style.display = 'none';
    } else {
      showFileOpMsg('✗ ' + d.error, 'var(--warn)');
    }
  }

  function showFileOpMsg(msg, color) {
    const el = document.getElementById('fileop-msg');
    el.textContent = msg;
    el.style.color = color;
    setTimeout(() => { el.textContent = ''; }, 4000);
  }

  // ── network IP ──
  fetch('/netinfo').then(r=>r.json()).then(d => {
    document.getElementById('net-ip').textContent = d.ip;
  });

  // ── LED ──
  async function toggleLED() {
    const on = document.getElementById('led-toggle').checked;
    await fetch('/led', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({on})
    });
  }

  // ── phase ──
  function setPhase(phase) {
    ['focus','capture'].forEach(p => {
      const el = document.getElementById('pip-' + p);
      if (el) el.classList.remove('active','done');
    });
    const order = ['focus','capture'];
    const idx = order.indexOf(phase);
    order.forEach((p, i) => {
      const el = document.getElementById('pip-' + p);
      if (!el) return;
      if (i < idx) el.classList.add('done');
      else if (i === idx) el.classList.add('active');
    });
  }

  // ── focus ──
  async function startFocus() {
    // clear any previous results from preview
    document.querySelectorAll('.preview-dual-img').forEach(e => e.remove());
    document.querySelectorAll('.preview-tag.top-right').forEach(e => e.remove());
    document.getElementById('preview-area').classList.remove('dual');
    document.getElementById('preview-img').style.display = 'block';
    resetZoom();

    setPhase('focus');
    document.getElementById('btn-focus').style.display = 'none';
    document.getElementById('btn-focus-ok').style.display = 'block';
    document.getElementById('btn-day').disabled = true;
    document.getElementById('btn-biolum').disabled = true;
    document.getElementById('btn-both').disabled = true;
    document.getElementById('preview-label').textContent = 'STARTING...';
    document.getElementById('crosshair').style.display = 'block';
    document.getElementById('led-toggle').checked = true;

    await fetch('/focus/start', {method:'POST'});

    // wait 2s for viewfinder to activate then connect stream directly
    const img = document.getElementById('preview-img');
    img.src = '';
    setTimeout(() => {
      img.src = '/stream?' + Date.now();
      document.getElementById('preview-label').textContent = 'LIVE FOCUS';
    }, 2000);
  }

  async function focusOk() {
    setPhase('capture');
    document.getElementById('btn-focus-ok').style.display = 'none';
    document.getElementById('btn-focus').style.display = 'block';
    document.getElementById('btn-day').disabled = false;
    document.getElementById('btn-biolum').disabled = false;
    document.getElementById('btn-both').disabled = false;
    document.getElementById('preview-label').textContent = 'READY';
    document.getElementById('crosshair').style.display = 'none';
    await fetch('/focus/stop', {method:'POST'});
    document.getElementById('preview-img').src = '/last_preview?' + Date.now();
  }

  // ── settings ──
  function getSettings() {
    return {
      experiment: document.getElementById('exp-name').value || 'biolum',
      sample: document.getElementById('sample-name').value || 'sample',
      day_iso: document.getElementById('day-iso').value,
      day_shutter: document.getElementById('day-shutter').value,
      biolum_iso: document.getElementById('biolum-iso').value,
      biolum_exposure: document.getElementById('biolum-exp').value
    };
  }

  // ── capture ──
  let logTimer = null;

  async function captureDay() {
    setBusy(true);
    resetZoom();
    document.getElementById('preview-label').textContent = 'DAY CAPTURE';
    await fetch('/capture/day', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(getSettings())
    });
    pollLog(false, false, 0);
  }

  async function captureBiolum() {
    setBusy(true);
    resetZoom();
    document.getElementById('preview-label').textContent = 'BIOLUM CAPTURE';
    await fetch('/capture/biolum', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(getSettings())
    });
    // countdown starts when server signals shutter open
    const exp = parseInt(document.getElementById('biolum-exp').value);
    waitForShutterAndCountdown(exp, 'Exposure');
    pollLog(true, false, exp);
  }

  async function captureBoth() {
    setBusy(true);
    resetZoom();
    document.getElementById('preview-label').textContent = 'DAY CAPTURE';
    await fetch('/capture/both', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(getSettings())
    });
    const exp = parseInt(document.getElementById('biolum-exp').value);
    pollLog(true, true, exp);  // isSequence=true
  }

  function setBusy(busy) {
    ['btn-day','btn-biolum','btn-both','btn-focus'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.disabled = busy;
    });
    const stopBtn = document.getElementById('btn-stop');
    if (busy) {
      stopBtn.style.display = 'block';
      stopBtn.classList.add('btn-stop-active');
    } else {
      stopBtn.style.display = 'none';
      stopBtn.classList.remove('btn-stop-active');
    }
    // lock LED off during any capture (especially biolum)
    const ledToggle = document.getElementById('led-toggle');
    if (busy) {
      ledToggle.checked = false;
      ledToggle.disabled = true;
      fetch('/led', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({on: false})});
    } else {
      ledToggle.disabled = false;
    }
  }

  async function stopCapture() {
    const btn = document.getElementById('btn-stop');
    btn.textContent = '⏳ Stopping...';
    btn.disabled = true;
    try {
      await fetch('/capture/stop', {method: 'POST'});
    } catch(e) {}
    // clean up UI
    if (countdownInterval) clearInterval(countdownInterval);
    if (shutterPollTimer) clearInterval(shutterPollTimer);
    if (logTimer) clearInterval(logTimer);
    document.getElementById('countdown-wrap').classList.remove('visible');
    document.getElementById('preview-label').textContent = 'STOPPED';
    document.getElementById('log-body').innerHTML += '<div class="log-warn">⚠ Capture aborted by user</div>';
    // reset stop button before calling setBusy so it ends up hidden and re-enabled
    btn.textContent = '■ Stop Capture';
    btn.disabled = false;
    setBusy(false);
  }

  // ── countdown ──
  let countdownInterval = null;
  let shutterPollTimer  = null;

  // Poll server until shutter opens, then start countdown
  async function waitForShutterAndCountdown(totalSeconds, label) {
    if (shutterPollTimer) clearInterval(shutterPollTimer);
    shutterPollTimer = setInterval(async () => {
      try {
        const r = await fetch('/shutter_status');
        const d = await r.json();
        if (d.open) {
          clearInterval(shutterPollTimer);
          startCountdown(totalSeconds, label);
        }
      } catch(e) {}
    }, 300);
  }

  function startCountdown(totalSeconds, label) {
    if (countdownInterval) clearInterval(countdownInterval);
    const wrap = document.getElementById('countdown-wrap');
    const num  = document.getElementById('countdown-num');
    const bar  = document.getElementById('countdown-bar');
    const lbl  = document.getElementById('countdown-label');

    wrap.classList.add('visible');
    lbl.textContent = label;
    num.textContent = totalSeconds + 's';
    bar.style.transform = 'scaleX(1)';

    // use wall-clock time for accuracy instead of tick counting
    const startTime = Date.now();
    const endTime = startTime + totalSeconds * 1000;

    countdownInterval = setInterval(() => {
      const now = Date.now();
      const remaining = Math.max(0, Math.ceil((endTime - now) / 1000));
      const frac = (endTime - now) / (totalSeconds * 1000);
      bar.style.transform = `scaleX(${Math.max(0, frac)})`;
      num.textContent = remaining + 's';

      if (now >= endTime) {
        clearInterval(countdownInterval);
        lbl.textContent = 'Downloading file at the pace of Nikon Camera, please be patient';
        const bufWait = totalSeconds + 10;
        const bufEnd = Date.now() + bufWait * 1000;
        bar.style.transform = 'scaleX(1)';
        const bufInterval = setInterval(() => {
          const bufNow = Date.now();
          const bufRemaining = Math.max(0, Math.ceil((bufEnd - bufNow) / 1000));
          const bufFrac = (bufEnd - bufNow) / (bufWait * 1000);
          bar.style.transform = `scaleX(${Math.max(0, bufFrac)})`;
          num.textContent = bufRemaining + 's';
          if (bufNow >= bufEnd) {
            clearInterval(bufInterval);
            wrap.classList.remove('visible');
          }
        }, 250);  // update 4x/sec for smooth bar
      }
    }, 250);  // update 4x/sec for smooth bar
  }

  function startDarkWaitCountdown(seconds) {
    const wrap = document.getElementById('countdown-wrap');
    const num  = document.getElementById('countdown-num');
    const bar  = document.getElementById('countdown-bar');
    const lbl  = document.getElementById('countdown-label');
    wrap.classList.add('visible');
    lbl.textContent = 'LEDs cooling down';
    const endTime = Date.now() + seconds * 1000;
    bar.style.transform = 'scaleX(1)';
    const iv = setInterval(() => {
      const now = Date.now();
      const remaining = Math.max(0, Math.ceil((endTime - now) / 1000));
      num.textContent = remaining + 's';
      bar.style.transform = `scaleX(${Math.max(0, (endTime - now) / (seconds * 1000))})`;
      if (now >= endTime) {
        clearInterval(iv);
        wrap.classList.remove('visible');
      }
    }, 250);
  }

  // ── zoom ──
  let zoomLevel = 1;
  let zoomOriginX = 50, zoomOriginY = 50;

  function getZoomTargets() {
    const area = document.getElementById('preview-area');
    if (area.classList.contains('dual')) {
      return Array.from(area.querySelectorAll('.preview-dual-img'));
    }
    return [document.getElementById('preview-img')];
  }

  function applyZoom() {
    getZoomTargets().forEach(img => {
      if (!img) return;
      img.style.transform = `scale(${zoomLevel})`;
      img.style.transformOrigin = `${zoomOriginX}% ${zoomOriginY}%`;
      img.style.cursor = zoomLevel > 1 ? 'zoom-out' : 'zoom-in';
    });
  }

  function resetZoom() {
    zoomLevel = 1;
    zoomOriginX = 50;
    zoomOriginY = 50;
    applyZoom();
  }

  // scroll to zoom
  document.getElementById('preview-area').addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    zoomOriginX = ((e.clientX - rect.left) / rect.width) * 100;
    zoomOriginY = ((e.clientY - rect.top) / rect.height) * 100;
    const delta = e.deltaY < 0 ? 0.2 : -0.2;
    zoomLevel = Math.min(8, Math.max(1, zoomLevel + delta));
    applyZoom();
  }, { passive: false });

  // click to toggle zoom
  document.getElementById('preview-area').addEventListener('click', (e) => {
    const isImg = e.target.id === 'preview-img' || e.target.classList.contains('preview-dual-img');
    if (!isImg) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (zoomLevel > 1) {
      zoomLevel = 1;
    } else {
      zoomLevel = 3;
      zoomOriginX = ((e.clientX - rect.left) / rect.width) * 100;
      zoomOriginY = ((e.clientY - rect.top) / rect.height) * 100;
    }
    applyZoom();
  });

  // touch pinch-to-zoom
  let lastPinchDist = null;
  document.getElementById('preview-area').addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) lastPinchDist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
  });
  document.getElementById('preview-area').addEventListener('touchmove', (e) => {
    if (e.touches.length === 2) {
      e.preventDefault();
      const dist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      if (lastPinchDist) {
        zoomLevel = Math.min(8, Math.max(1, zoomLevel * (dist / lastPinchDist)));
        applyZoom();
      }
      lastPinchDist = dist;
    }
  }, { passive: false });
  document.getElementById('preview-area').addEventListener('touchend', () => {
    lastPinchDist = null;
  });

  // ── log polling ──
  function pollLog(hasBiolum, isSequence, biolumExp) {
    if (logTimer) clearInterval(logTimer);
    let dayShown = false;
    let countdownStarted = false;
    let darkWaitStarted = false;
    logTimer = setInterval(async () => {
      const r = await fetch('/log');
      const data = await r.json();
      const lines = data.lines;
      const body = document.getElementById('log-body');
      body.innerHTML = lines.map(line => {
        if (line.startsWith('✓') || line.startsWith('▶')) return `<div class="log-ok">${line}</div>`;
        if (line.startsWith('ERR'))  return `<div class="log-err">${line}</div>`;
        if (line.startsWith('$'))    return `<div class="log-warn">${line}</div>`;
        return `<div class="log-line">${line}</div>`;
      }).join('');
      body.scrollTop = body.scrollHeight;

      // ── sequence-specific UI updates driven by log content ──
      if (isSequence) {
        const logText = lines.join(' ');

        // day shot complete — show day preview
        if (!dayShown && logText.includes('✓ Day saved')) {
          dayShown = true;
          document.getElementById('preview-label').textContent = 'DAY ✓ — waiting...';
          const res = await fetch('/results');
          const rd = await res.json();
          if (rd.day_img) {
            const img = document.getElementById('preview-img');
            img.src = '/image?path=' + encodeURIComponent(rd.day_img) + '&t=' + Date.now();
            img.style.display = 'block';
          }
        }

        // dark wait
        if (!darkWaitStarted && logText.includes('DARK_WAIT_START')) {
          darkWaitStarted = true;
          document.getElementById('preview-label').textContent = 'WAITING FOR LEDs TO TURN OFF';
          startDarkWaitCountdown(5);
        }

        // biolum phase started
        if (logText.includes('▶ BIOLUM IMAGE')) {
          document.getElementById('preview-label').textContent = 'BIOLUM CAPTURE';
        }

        // shutter open — start countdown
        if (!countdownStarted && logText.includes('SHUTTER_OPEN')) {
          countdownStarted = true;
          startCountdown(biolumExp, 'Biolum Exposure');
        }

        // shutter closed — show loading message
        if (logText.includes('SHUTTER_CLOSED') && countdownStarted) {
          document.getElementById('preview-label').textContent = 'LOADING BIOLUM PHOTO...';
        }
      }

      // show loading message after shutter closes (standalone biolum)
      if (!isSequence && hasBiolum && data.lines &&
          data.lines.some(l => l.trim() === 'SHUTTER_CLOSED')) {
        document.getElementById('preview-label').textContent = 'LOADING BIOLUM PHOTO...';
      }

      if (data.done) {
        clearInterval(logTimer);
        setBusy(false);
        if (countdownInterval) clearInterval(countdownInterval);
        if (shutterPollTimer) clearInterval(shutterPollTimer);
        document.getElementById('countdown-wrap').classList.remove('visible');
        await loadResults(hasBiolum);
      }
    }, 500);
  }

  async function loadResults(hasBiolum) {
    const r = await fetch('/results');
    const data = await r.json();
    const area = document.getElementById('preview-area');
    const img  = document.getElementById('preview-img');

    area.querySelectorAll('.preview-dual-img').forEach(e => e.remove());
    area.querySelectorAll('.preview-tag.top-right').forEach(e => e.remove());
    area.classList.remove('dual');
    img.style.display = 'block';

    // show biolum only if this capture included biolum
    if (hasBiolum && data.biolum_img) {
      img.src = '/image?path=' + encodeURIComponent(data.biolum_img) + '&t=' + Date.now();
      document.getElementById('preview-label').textContent = 'BIOLUM';
    } else if (data.day_img) {
      img.src = '/image?path=' + encodeURIComponent(data.day_img) + '&t=' + Date.now();
      document.getElementById('preview-label').textContent = 'DAY';
    }
  }

  // ── file browser ──
  async function loadFileTree() {
    const r = await fetch('/files');
    const data = await r.json();
    const container = document.getElementById('folder-list');
    if (!data.experiments.length) {
      container.innerHTML = '<div style="font-family:var(--mono);font-size:11px;color:var(--muted);">No experiments found.</div>';
      return;
    }
    container.innerHTML = data.experiments.map((exp, ei) => `
      <div class="exp-folder">
        <div class="exp-folder-header" onclick="toggleFolder(${ei})">
          <span>${exp.name}</span>
          <div style="display:flex;align-items:center;gap:8px;">
            <span class="exp-folder-count">${exp.count} pairs</span>
            <button onclick="event.stopPropagation(); downloadFolder('${exp.path}','${exp.name}', this)"
               style="font-family:var(--mono);font-size:9px;color:var(--accent);letter-spacing:0.08em;background:transparent;border:1px solid var(--border2);padding:1px 6px;border-radius:2px;cursor:pointer;"
               title="Download folder as ZIP">↓ zip</button>
          </div>
        </div>
        <div class="exp-folder-files" id="folder-${ei}">
          ${exp.pairs.map(p => `
            <div class="file-item" onclick="selectPair('${p.stem}','${p.jpg_path || ''}','${p.nef_path || ''}','${exp.path}')"
                 id="pair-${btoa(p.stem).replace(/=/g,'')}">
              <span>${p.stem}</span>
              <div style="display:flex;align-items:center;gap:8px;">
                <div class="file-pair-sizes">
                  ${p.jpg_size ? `<span title="JPEG">jpg ${p.jpg_size}</span>` : ''}
                  ${p.nef_size ? `<span title="NEF">nef ${p.nef_size}</span>` : ''}
                </div>
                <div class="file-actions">
                  <button class="file-action-btn rename" title="Rename pair"
                    onclick="event.stopPropagation(); promptRename('${p.stem}','${p.jpg_path || ''}','${p.nef_path || ''}')">✏</button>
                  <button class="file-action-btn delete" title="Delete both files"
                    onclick="event.stopPropagation(); deletePair('${p.stem}','${p.jpg_path || ''}','${p.nef_path || ''}')">🗑</button>
                </div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `).join('');
  }

  function selectPair(stem, jpgPath, nefPath, folderPath) {
    document.querySelectorAll('.file-item').forEach(e => e.classList.remove('selected'));
    const key = btoa(stem).replace(/=/g,'');
    const el = document.getElementById('pair-' + key);
    if (el) el.classList.add('selected');

    const path = jpgPath || nefPath;
    const name = path.split('/').pop();
    setSelectedFile(path, name, jpgPath ? 'jpg' : 'nef');

    document.getElementById('file-info').textContent = stem;
    const dlLink = document.getElementById('download-link');
    if (jpgPath) {
      dlLink.href = '/download?path=' + encodeURIComponent(jpgPath);
      dlLink.download = stem + '.jpg';
      dlLink.style.display = 'inline-block';
    } else { dlLink.style.display = 'none'; }

    const noSel = document.getElementById('no-selection');
    const img = document.getElementById('file-preview-img');
    if (jpgPath) {
      noSel.style.display = 'none';
      img.style.display = 'block';
      img.src = '/image?path=' + encodeURIComponent(jpgPath) + '&t=' + Date.now();
      fileZoom = 1; applyFileZoom();
    } else {
      noSel.style.display = 'flex';
      noSel.textContent = 'NEF only — download to view';
      img.style.display = 'none';
    }
  }

  async function promptRename(stem, jpgPath, nefPath) {
    const newStem = prompt('Rename "' + stem + '" to:', stem);
    if (!newStem || newStem === stem) return;
    const results = [];
    if (jpgPath) results.push(await renameSingle(jpgPath, newStem));
    if (nefPath) results.push(await renameSingle(nefPath, newStem));
    const allOk = results.every(r => r.ok);
    showToast(allOk ? '✓ Renamed to ' + newStem : '✗ Rename failed');
    loadFileTree();
  }

  async function renameSingle(path, newStem) {
    if (!path) return {ok: true};  // skip if this file doesn't exist in pair
    const r = await fetch('/file/rename', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({path, new_name: newStem})
    });
    return r.json();
  }

  async function deletePair(stem, jpgPath, nefPath) {
    const parts = [jpgPath, nefPath].filter(Boolean);
    const names = parts.map(p => p.split('/').pop()).join(' + ');
    if (!confirm('Delete both files? ' + names + ' - This cannot be undone.')) return;
    const results = [];
    for (const path of parts) {
      const r = await fetch('/file/delete', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({path})
      });
      results.push(await r.json());
    }
    const allOk = results.every(r => r.ok);
    showToast(allOk ? '✓ Deleted ' + stem : '✗ Delete failed');
    document.getElementById('file-preview-img').src = '';
    document.getElementById('no-selection').style.display = 'flex';
    document.getElementById('no-selection').textContent = 'Select a file to preview';
    document.getElementById('file-preview-img').style.display = 'none';
    loadFileTree();
  }

  function toggleFolder(idx) {
    const el = document.getElementById('folder-' + idx);
    el.classList.toggle('open');
  }

  function selectFile(path, name, size, isJpeg) {
    document.querySelectorAll('.file-item').forEach(e => e.classList.remove('selected'));
    const key = btoa(path).replace(/=/g,'');
    const el = document.getElementById('file-' + key);
    if (el) el.classList.add('selected');
    const ext = name.split('.').pop();
    setSelectedFile(path, name, ext);

    document.getElementById('file-info').textContent = name + '  ·  ' + size;

    const dlLink = document.getElementById('download-link');
    dlLink.href = '/download?path=' + encodeURIComponent(path);
    dlLink.download = name;
    dlLink.style.display = 'inline-block';

    const wrap = document.getElementById('file-preview-wrap');
    const noSel = document.getElementById('no-selection');
    const img = document.getElementById('file-preview-img');

    if (isJpeg) {
      noSel.style.display = 'none';
      img.style.display = 'block';
      img.src = '/image?path=' + encodeURIComponent(path) + '&t=' + Date.now();
    } else {
      noSel.style.display = 'flex';
      noSel.textContent = 'RAW/NEF — download to view';
      img.style.display = 'none';
    }
  }

  // ── folder download ──
  async function downloadFolder(path, name, btn) {
    const orig = btn.textContent;
    btn.disabled = true;
    btn.style.color = 'var(--amber)';

    // Step 1: get size estimate
    let estimateSec = 5;
    try {
      const sr = await fetch('/folder_size?path=' + encodeURIComponent(path));
      const sd = await sr.json();
      estimateSec = sd.estimate_sec;
      showToast(`Building ZIP (${sd.size_mb} MB) — ~${estimateSec}s, download starts automatically`);
    } catch(e) {}

    // countdown on button
    let remaining = estimateSec;
    btn.textContent = remaining > 0 ? `~${remaining}s` : '...';
    const countBtn = setInterval(() => {
      remaining--;
      btn.textContent = remaining > 0 ? `~${remaining}s` : '↓';
      if (remaining <= 0) clearInterval(countBtn);
    }, 1000);

    // Step 2: fetch and trigger download
    try {
      const r = await fetch('/download_folder?path=' + encodeURIComponent(path));
      clearInterval(countBtn);
      if (!r.ok) throw new Error('Failed');
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = name + '.zip';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      btn.textContent = '✓';
      btn.style.color = 'var(--accent)';
      showToast('Download started!');
      setTimeout(() => { btn.textContent = orig; btn.disabled = false; btn.style.color = ''; }, 3000);
    } catch(e) {
      clearInterval(countBtn);
      btn.textContent = 'err';
      btn.style.color = 'var(--warn)';
      showToast('ZIP failed — try again');
      setTimeout(() => { btn.textContent = orig; btn.disabled = false; btn.style.color = ''; }, 3000);
    }
  }

  function showToast(msg) {
    let t = document.getElementById('toast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'toast';
      t.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1a2333;border:1px solid var(--border2);color:var(--text);font-family:var(--mono);font-size:11px;letter-spacing:0.08em;padding:8px 18px;border-radius:4px;z-index:9999;transition:opacity 0.4s;pointer-events:none;';
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.style.opacity = '1';
    clearTimeout(t._timer);
    t._timer = setTimeout(() => { t.style.opacity = '0'; }, 4000);
  }

  // ── file browser zoom ──
  let fileZoom = 1;
  let fileZoomOX = 50, fileZoomOY = 50;

  function applyFileZoom() {
    const img = document.getElementById('file-preview-img');
    img.style.transform = `scale(${fileZoom})`;
    img.style.transformOrigin = `${fileZoomOX}% ${fileZoomOY}%`;
    img.style.cursor = fileZoom > 1 ? 'zoom-out' : 'zoom-in';
  }

  document.getElementById('file-preview-wrap').addEventListener('wheel', (e) => {
    if (!document.getElementById('file-preview-img').src) return;
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    fileZoomOX = ((e.clientX - rect.left) / rect.width) * 100;
    fileZoomOY = ((e.clientY - rect.top) / rect.height) * 100;
    const delta = e.deltaY < 0 ? 0.25 : -0.25;
    fileZoom = Math.min(8, Math.max(1, fileZoom + delta));
    applyFileZoom();
  }, { passive: false });

  document.getElementById('file-preview-wrap').addEventListener('click', (e) => {
    if (e.target.id !== 'file-preview-img') return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (fileZoom > 1) { fileZoom = 1; }
    else {
      fileZoom = 3;
      fileZoomOX = ((e.clientX - rect.left) / rect.width) * 100;
      fileZoomOY = ((e.clientY - rect.top) / rect.height) * 100;
    }
    applyFileZoom();
  });

  // pinch for file browser
  let filePinchDist = null;
  document.getElementById('file-preview-wrap').addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) filePinchDist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
  });
  document.getElementById('file-preview-wrap').addEventListener('touchmove', (e) => {
    if (e.touches.length === 2 && filePinchDist) {
      e.preventDefault();
      const dist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      fileZoom = Math.min(8, Math.max(1, fileZoom * (dist / filePinchDist)));
      filePinchDist = dist;
      applyFileZoom();
    }
  }, { passive: false });
  document.getElementById('file-preview-wrap').addEventListener('touchend', () => { filePinchDist = null; });
</script>
</body>
</html>
"""

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/netinfo")
def netinfo():
    ip = get_pi_ip()
    return jsonify({"ip": ip if ip != "unknown" else "10.42.0.1"})

@app.route("/stream")
def stream():
    r = Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return r

@app.route("/last_preview")
def last_preview():
    try:
        with open(PREVIEW_READ, "rb") as f:
            return Response(f.read(), mimetype="image/jpeg")
    except FileNotFoundError:
        return "", 404

@app.route("/led", methods=["POST"])
def led():
    data = request.get_json()
    _light.on() if data.get("on") else _light.off()
    return jsonify({"status": "ok"})

@app.route("/focus/start", methods=["POST"])
def focus_start():
    global _streaming
    _streaming = False  # reset in case previous session left it True
    time.sleep(0.3)
    _light.on()
    start_preview()
    return jsonify({"status": "ok"})

@app.route("/focus/stop", methods=["POST"])
def focus_stop():
    stop_preview()
    _light.off()
    return jsonify({"status": "ok"})

@app.route("/capture/day", methods=["POST"])
def cap_day():
    settings = request.get_json()
    with _capture_lock:
        _capture_log.clear()
    global _capture_done
    _capture_done = False
    threading.Thread(target=run_day, args=(settings,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/capture/biolum", methods=["POST"])
def cap_biolum():
    settings = request.get_json()
    with _capture_lock:
        _capture_log.clear()
    global _capture_done
    _capture_done = False
    threading.Thread(target=run_biolum, args=(settings,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/capture/stop", methods=["POST"])
def cap_stop():
    global _shutter_open, _capture_done, _capture_phase, _stop_requested
    _stop_requested = True  # signal capture threads to exit cleanly
    # if shutter is open, close it before killing gphoto2
    if _shutter_open:
        log("⚠ Stop requested — closing shutter first...")
        subprocess.run(["gphoto2", "--set-config", "bulb=0"],
                       capture_output=True, timeout=5)
        _shutter_open = False
        log("Shutter closed.")
    # kill gphoto2 and gvfs so camera USB is released cleanly
    subprocess.run(["pkill", "-f", "gphoto2"], capture_output=True)
    subprocess.run(["pkill", "-f", "gvfs-gphoto2"], capture_output=True)
    time.sleep(1)
    _light.off()
    _capture_phase = "idle"
    _capture_done = True
    log("⚠ Capture stopped by user.")
    return jsonify({"status": "stopped"})

@app.route("/capture/both", methods=["POST"])
def cap_both():
    settings = request.get_json()
    with _capture_lock:
        _capture_log.clear()
    global _capture_done
    _capture_done = False
    threading.Thread(target=run_both, args=(settings,), daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/shutter_status")
def shutter_status():
    return jsonify({"open": _shutter_open})

@app.route("/capture_phase")
def capture_phase():
    return jsonify({"phase": _capture_phase})

@app.route("/log")
def get_log():
    with _capture_lock:
        lines = list(_capture_log)
    done = bool(lines and lines[-1].startswith("✓ Done"))
    return jsonify({"lines": lines, "done": done})

@app.route("/results")
def results():
    return jsonify({"day_img": _last_day_img, "biolum_img": _last_biolum_img})

@app.route("/reset", methods=["POST"])
def reset_state():
    global _last_day_img, _last_biolum_img, _capture_done
    _last_day_img    = None
    _last_biolum_img = None
    _capture_done    = False
    with _capture_lock:
        _capture_log.clear()
    return jsonify({"status": "ok"})

@app.route("/image")
def serve_image():
    path = request.args.get("path", "")
    try:
        with open(path, "rb") as f:
            return Response(f.read(), mimetype="image/jpeg")
    except Exception:
        return "", 404

@app.route("/files")
def list_files():
    return jsonify({"experiments": get_experiments()})

@app.route("/clock")
def get_clock():
    now = datetime.now()
    return jsonify({"time": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d")})

@app.route("/clock/set", methods=["POST"])
def set_clock():
    data = request.get_json()
    iso = data.get("iso", "")
    try:
        # parse ISO string and format for Linux date command
        from datetime import timezone
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        dt_local = dt.astimezone()
        date_str = dt_local.strftime("%Y-%m-%d %H:%M:%S")
        result = subprocess.run(
            ["sudo", "date", "-s", date_str],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return jsonify({"ok": True})
        else:
            return jsonify({"ok": False, "error": result.stderr.strip()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/file/rename", methods=["POST"])
def rename_file():
    data = request.get_json()
    old_path = Path(data.get("path", ""))
    new_name = data.get("new_name", "").strip()
    if not old_path.exists():
        return jsonify({"ok": False, "error": "File not found"})
    if not new_name:
        return jsonify({"ok": False, "error": "Empty name"})
    # keep original extension
    new_path = old_path.parent / (new_name + old_path.suffix)
    if new_path.exists():
        return jsonify({"ok": False, "error": "File already exists"})
    try:
        old_path.rename(new_path)
        return jsonify({"ok": True, "new_name": new_path.name, "new_path": str(new_path)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/file/delete", methods=["POST"])
def delete_file():
    data = request.get_json()
    path = Path(data.get("path", ""))
    # safety: only allow deletion within BASE_DIR
    try:
        path.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        return jsonify({"ok": False, "error": "Access denied"})
    if not path.exists():
        return jsonify({"ok": False, "error": "File not found"})
    try:
        path.unlink()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/download")
def download_file():
    path = request.args.get("path", "")
    try:
        return send_file(path, as_attachment=True)
    except Exception:
        return "", 404

@app.route("/folder_size")
def folder_size():
    path = request.args.get("path", "")
    folder = Path(path)
    if not folder.is_dir():
        return jsonify({"size_mb": 0, "estimate_sec": 5})
    total = sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
    size_mb = total / 1024 / 1024
    # Empirically measured on Pi 4 with D800 files: ~2.5 MB/s for mixed NEF+JPEG
    # With ZIP_STORED for NEFs, now IO-bound: ~40 MB/s read speed
    estimate_sec = max(3, int(size_mb / 40))
    return jsonify({"size_mb": round(size_mb, 1), "estimate_sec": estimate_sec})

@app.route("/download_folder")
def download_folder():
    import zipfile, io
    path = request.args.get("path", "")
    folder = Path(path)
    if not folder.is_dir():
        return "Not found", 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for f in sorted(folder.iterdir()):
            if f.is_file():
                # NEF/RAW: store uncompressed (fast), JPEG: compress (tiny anyway)
                method = zipfile.ZIP_STORED if f.suffix.lower() in ('.nef', '.raw') else zipfile.ZIP_DEFLATED
                zf.write(f, f.name, compress_type=method)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=folder.name + ".zip",
                     mimetype="application/zip")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    ip = get_pi_ip()
    print()
    print("  Biolum Controller")
    print("  ─────────────────────────────────────────")
    print(f"    Pi screen : http://localhost:5000")
    print(f"    Laptop    : http://{ip}:5000")
    print(f"    Phone     : http://{ip}:5000")
    print("  ─────────────────────────────────────────")
    print()
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        _light.off()
        sys.exit(0)
