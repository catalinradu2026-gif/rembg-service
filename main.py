from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.parse
import time
import random
import string
import numpy as np
import cv2
from PIL import Image
import io
import requests as req_lib

PORT = int(os.environ.get("PORT", 8002))
PROC_DIM = 640

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
print(f"rembg-service ready on port {PORT}")


# ── Background removal ────────────────────────────────────────────────────────

def is_simple_background(img):
    h, w = img.shape[:2]
    m = max(20, int(min(h, w) * 0.06))
    corners = [img[:m, :m], img[:m, w-m:], img[h-m:, :m], img[h-m:, w-m:]]
    samples = np.vstack([c.reshape(-1, 3).astype(np.float32) for c in corners])
    return samples.std(axis=0).mean() < 28, samples.mean(axis=0)


def color_key_mask(img, bg_color, tol=38):
    h, w = img.shape[:2]
    diff = img.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    alpha = np.clip((dist - tol * 0.5) / (tol * 0.5) * 255, 0, 255).astype(np.uint8)
    flood = np.zeros((h + 2, w + 2), np.uint8)
    bg = np.where(dist < tol * 1.2, 0, 255).astype(np.uint8)
    for cy in [0, h // 2, h - 1]:
        for cx in [0, w // 2, w - 1]:
            if bg[cy, cx] == 0:
                cv2.floodFill(bg, flood, (cx, cy), 128)
    flood_bg_inv = 255 - (bg == 128).astype(np.uint8) * 255
    combined = cv2.bitwise_and(alpha, flood_bg_inv)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k1), cv2.MORPH_OPEN, k2)
    _, thresh = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)
    if n > 1:
        thresh = np.where(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    blur = cv2.GaussianBlur(thresh.astype(np.float32), (11, 11), 3)
    border = cv2.dilate(thresh, np.ones((9, 9), np.uint8)) - cv2.erode(thresh, np.ones((9, 9), np.uint8))
    final = thresh.astype(np.float32)
    final[border > 0] = blur[border > 0]
    return np.clip(final, 0, 255).astype(np.uint8)


def grabcut_mask(img):
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enh = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    mx = int(w * 0.04)
    my_top = int(h * 0.04)
    my_bot = int(h * 0.01)  # very small bottom margin to keep wheels
    cv2.grabCut(img_enh, mask, (mx, my_top, w - 2*mx, h - my_top - my_bot), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    alpha = cv2.morphologyEx(cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k1), cv2.MORPH_OPEN, k2)
    blur = cv2.GaussianBlur(alpha.astype(np.float32), (11, 11), 3)
    border = cv2.dilate(alpha, np.ones((9, 9), np.uint8)) - cv2.erode(alpha, np.ones((9, 9), np.uint8))
    final = alpha.astype(np.float32)
    final[border > 0] = blur[border > 0]
    return np.clip(final, 0, 255).astype(np.uint8)


def resize_for_hf(image_data: bytes, max_side: int = 800) -> bytes:
    """Resize image to max_side before sending to HF (faster, less likely to OOM)."""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_data
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return image_data
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()
    except Exception:
        return image_data


def remove_bg_clipdrop(image_data: bytes) -> bytes:
    """Remove background using Clipdrop API (Stability AI) — 100 free/day."""
    api_key = os.environ.get("CLIPDROP_KEY", "")
    if not api_key:
        raise ValueError("no CLIPDROP_KEY")
    r = req_lib.post(
        'https://clipdrop-api.co/remove-background/v1',
        files={'image_file': ('image.jpg', image_data, 'image/jpeg')},
        headers={'x-api-key': api_key},
        timeout=30,
    )
    if not r.ok:
        raise ValueError(f"clipdrop {r.status_code}: {r.text[:150]}")
    if len(r.content) < 1000:
        raise ValueError(f"clipdrop response too small: {len(r.content)} bytes")
    print(f"Clipdrop ok: {len(r.content)} bytes")
    return r.content


def remove_bg_hf(image_data: bytes) -> bytes:
    """Remove background using Hugging Face RMBG (ML quality, free tier)."""
    import urllib.error, time, json as _json
    hf_token = os.environ.get("HF_TOKEN", "")
    small_data = resize_for_hf(image_data, 800)
    models = [
        "https://router.huggingface.co/hf-inference/models/briaai/RMBG-1.4",
        "https://router.huggingface.co/hf-inference/models/briaai/RMBG-2.0",
    ]
    for model_url in models:
        for attempt in range(3):
            try:
                req = urllib.request.Request(model_url, data=small_data, method='POST')
                req.add_header('Content-Type', 'image/jpeg')
                if hf_token:
                    req.add_header('Authorization', f'Bearer {hf_token}')
                with urllib.request.urlopen(req, timeout=45) as r:
                    status = r.status
                    result = r.read()
                if len(result) > 1000:
                    print(f"HF RMBG ok: {model_url} attempt {attempt+1}, {len(result)} bytes")
                    return result
                print(f"HF response too small ({len(result)} bytes), status={status}")
                raise ValueError(f"HF response too small: {len(result)} bytes")
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8', errors='ignore')
                print(f"HF HTTP {e.code} from {model_url}: {body[:200]}")
                if e.code == 503:
                    wait = 20
                    try:
                        wait = min(int(_json.loads(body).get('estimated_time', 20)), 30)
                    except Exception:
                        pass
                    print(f"HF model loading, waiting {wait}s (attempt {attempt+1}/3)...")
                    time.sleep(wait)
                    continue
                break  # non-503 error: try next model
            except Exception as ex:
                print(f"HF attempt {attempt+1} exception: {ex}")
                if attempt < 2:
                    time.sleep(5)
    raise ValueError("HF RMBG all attempts failed")


def remove_bg_from_bytes(image_data: bytes) -> bytes:
    # Try Clipdrop first (ML quality, 100/day free)
    try:
        return remove_bg_clipdrop(image_data)
    except Exception as e:
        print(f"Clipdrop failed ({e}), trying HF...")

    # Try HF RMBG
    try:
        return remove_bg_hf(image_data)
    except Exception as e:
        print(f"HF RMBG failed ({e}), falling back to GrabCut")

    # Fallback: GrabCut
    nparr = np.frombuffer(image_data, np.uint8)
    orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if orig is None:
        raise ValueError("Could not decode image")
    oh, ow = orig.shape[:2]
    scale = min(1.0, PROC_DIM / max(oh, ow))
    proc = cv2.resize(orig, (int(ow * scale), int(oh * scale)), cv2.INTER_AREA) if scale < 1 else orig.copy()
    simple, bg_color = is_simple_background(proc)
    alpha_small = color_key_mask(proc, bg_color) if simple else grabcut_mask(proc)
    if scale < 1:
        alpha = cv2.GaussianBlur(cv2.resize(alpha_small, (ow, oh), cv2.INTER_LINEAR), (5, 5), 1)
    else:
        alpha = alpha_small
    b, g, r = cv2.split(orig)
    _, buf = cv2.imencode('.png', cv2.merge([b, g, r, alpha]))
    return buf.tobytes()


# ── Background compositing ────────────────────────────────────────────────────

def make_showroom(w: int, h: int) -> Image.Image:
    """zyAI WOW showroom — vectorized numpy, low memory."""
    from PIL import ImageDraw, ImageFont

    # Work at half resolution then upscale for speed + memory
    hw, hh = max(w // 2, 1), max(h // 2, 1)
    Y, X = np.mgrid[0:hh, 0:hw].astype(np.float32)
    Xn = (X - hw / 2) / (hw / 2)
    Yn = Y / hh
    wall_frac = 0.58
    wall_h_s = int(hh * wall_frac)

    pix = np.zeros((hh, hw, 3), dtype=np.float32)

    # Spotlight
    spot = np.exp(-(Xn**2 * 10 + (Yn * 0.75)**2 * 2)) * 85 + \
           np.exp(-(Xn**2 * 2   + (Yn - 0.05)**2 * 5)) * 30
    # Side accents
    left  = np.exp(-((Xn + 1.1)**2 * 2 + Yn**2 * 5)) * 35
    right = np.exp(-((Xn - 1.1)**2 * 2 + Yn**2 * 5)) * 35

    wm = (Y < wall_h_s).astype(np.float32)
    pix[:,:,0] += (spot * 0.5 + left * 0.5 + right * 0.5) * wm
    pix[:,:,1] += (spot * 0.4 + left * 0.25 + right * 0.25) * wm
    pix[:,:,2] += (spot * 0.95 + left * 0.9 + right * 0.9) * wm

    fm = 1 - wm
    ft = np.clip((Y - wall_h_s) / max(hh - wall_h_s, 1), 0, 1)
    refl = np.exp(-Xn**2 * 5) * np.exp(-ft * 5) * 60
    pix[:,:,2] += refl * fm
    pix[:,:,0] += refl * 0.3 * fm

    vig = np.clip(1 - (Xn**2 * 0.55 + (Yn - 0.45)**2 * 0.35), 0.07, 1)
    pix *= vig[:,:,np.newaxis]

    # Upscale to full resolution
    small = Image.fromarray(np.clip(pix, 0, 255).astype(np.uint8), 'RGB')
    img = small.resize((w, h), Image.BILINEAR).convert('RGBA')
    draw = ImageDraw.Draw(img)
    wall_h = int(h * wall_frac)

    # Glowing horizon line
    for dy in range(-3, 4):
        a = max(0, 190 - abs(dy) * 55)
        draw.line([(0, wall_h + dy), (w, wall_h + dy)], fill=(60, 110, 255, a))

    # Neon side lines
    for dx in [int(w * 0.04), int(w * 0.96)]:
        draw.line([(dx, int(wall_h * 0.08)), (dx, wall_h)], fill=(70, 50, 240, 90), width=2)

    # Floor grid
    vp = w // 2
    for i in range(1, 8):
        fy = wall_h + int((h - wall_h) * (i / 7)**0.6)
        draw.line([(0, fy), (w, fy)], fill=(45, 75, 195, max(8, 50 - i * 6)))
    for xp in range(-120, 121, 20):
        draw.line([(vp, wall_h), (vp + int(w * xp / 100), h)], fill=(38, 62, 185, 25))

    # zyAI.ro — litere volumetrice 3D pe peretele din spate
    from PIL import ImageFilter
    fs = max(48, int(w * 0.13))
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        fnt = ImageFont.load_default()
    txt = "zyAI.ro"
    bb_tmp = ImageDraw.Draw(Image.new('RGBA', (1, 1))).textbbox((0, 0), txt, font=fnt)
    tw, th = bb_tmp[2] - bb_tmp[0] + 60, bb_tmp[3] - bb_tmp[1] + 60
    pad = 30

    # 1. Glow difuz în spate
    glow = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text((pad, pad), txt, font=fnt, fill=(60, 120, 255, 200))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=14))

    # 2. Extruziune 3D — straturi offset spre dreapta-jos (efect volum)
    vol = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vol)
    depth = max(4, int(fs * 0.06))
    for i in range(depth, 0, -1):
        alpha_vol = int(60 + i * 8)
        vd.text((pad + i, pad + i), txt, font=fnt, fill=(20, 40, 140, alpha_vol))

    # 3. Fața principală a literelor — alb-albastru luminos
    face = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    fd = ImageDraw.Draw(face)
    fd.text((pad, pad), txt, font=fnt, fill=(200, 220, 255, 240))

    # 4. Highlight fin pe marginea de sus (iluminare)
    fd.text((pad, pad - 1), txt, font=fnt, fill=(255, 255, 255, 80))

    # Compozit final
    combined = Image.alpha_composite(glow, vol)
    combined = Image.alpha_composite(combined, face)

    # Poziționare pe perete, centrat, la ~25% din înălțimea peretelui
    px = (w - tw) // 2
    py = int(wall_h * 0.18)
    img.alpha_composite(combined, (px, py))

    return img


def make_studio(w: int, h: int) -> Image.Image:
    """White studio background."""
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    Xn = (X - w / 2) / (w / 2)
    Yn = (Y - h * 0.3) / h
    d = np.clip(np.sqrt(Xn ** 2 + Yn ** 2) / 0.65, 0, 1)
    v = np.clip(255 - d * 28, 220, 255).astype(np.uint8)
    b = np.clip(v.astype(np.int32) - (d * 8).astype(np.int32) + 8, 215, 255).astype(np.uint8)
    pix = np.stack([v, v, b], axis=2)
    return Image.fromarray(pix, 'RGB')


def add_watermark(img: Image.Image) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font_size = max(14, int(h * 0.03))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    text = "zyAI.ro"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = w - tw - int(w * 0.025)
    y = h - th - int(h * 0.025)
    # Shadow
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 100))
    # Main text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 180))
    return img


def composite_image(subject_png: bytes, category: str) -> bytes:
    subject = Image.open(io.BytesIO(subject_png)).convert('RGBA')
    # Cap resolution to avoid OOM on large original images
    MAX_SIDE = 1100
    sw, sh = subject.size
    if max(sw, sh) > MAX_SIDE:
        scale = MAX_SIDE / max(sw, sh)
        sw, sh = int(sw * scale), int(sh * scale)
        subject = subject.resize((sw, sh), Image.LANCZOS)
    cat_lower = (category or '').lower()
    is_auto = not cat_lower or cat_lower == 'general' or 'auto' in cat_lower or cat_lower in ('vehicule', 'masini', 'cars')
    bg = make_showroom(sw, sh) if is_auto else make_studio(sw, sh).convert('RGBA')

    # ── Car reflection on floor ───────────────────────────────────────────────
    if is_auto:
        from PIL import ImageFilter
        wall_h = int(sh * 0.58)
        refl_height = min(int(sh * 0.22), sh - wall_h - 5)
        if refl_height > 10:
            # Flip car vertically for reflection
            refl = subject.transpose(Image.FLIP_TOP_BOTTOM)
            # Fade gradient: opaque at top of reflection, transparent at bottom
            fade = Image.new('L', (sw, sh), 0)
            fade_arr = np.zeros((sh,), dtype=np.uint8)
            for fy in range(refl_height):
                fade_arr[sh - refl_height + fy] = int(55 * (1 - fy / refl_height))
            fade_2d = np.tile(fade_arr[:, np.newaxis], (1, sw))
            fade = Image.fromarray(fade_2d, 'L')
            refl_rgba = refl.copy()
            refl_rgba.putalpha(fade)
            refl_blur = refl_rgba.filter(ImageFilter.GaussianBlur(radius=3))
            bg.alpha_composite(refl_blur, (0, 0))

    bg.paste(subject, (0, 0), subject)
    result = add_watermark(bg)
    out = io.BytesIO()
    result.convert('RGB').save(out, format='WEBP', quality=85)
    return out.getvalue()


# ── Supabase upload ───────────────────────────────────────────────────────────

def upload_to_supabase(data: bytes, content_type: str) -> str:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase not configured on this service")
    from datetime import datetime
    now = datetime.utcnow()
    uid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    path = f"{now.year}/{now.month:02d}/{int(time.time())}-{uid}_pro.webp"
    url = f"{SUPABASE_URL}/storage/v1/object/listings/{path}"
    r = req_lib.post(url, headers={
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'apikey': SUPABASE_KEY,
        'Content-Type': content_type,
        'x-upsert': 'true',
        'cache-control': 'max-age=31536000',
    }, data=data, timeout=30)
    if not r.ok:
        raise ValueError(f"Supabase upload failed {r.status_code}: {r.text[:200]}")
    return f"{SUPABASE_URL}/storage/v1/object/public/listings/{path}"


# ── HTTP handler ──────────────────────────────────────────────────────────────

CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_cors(self):
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            has_clipdrop = bool(os.environ.get("CLIPDROP_KEY", ""))
body = json.dumps({"ok": True, "model": "clipdrop+hf+grabcut" if has_clipdrop else "hf+grabcut"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors()
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/debug-hf":
            import urllib.error as _ue
            hf_token = os.environ.get("HF_TOKEN", "")
            results = {"token_set": bool(hf_token), "token_prefix": hf_token[:8] if hf_token else ""}
            try:
                test_url = "https://www.gstatic.com/webp/gallery/1.jpg"
                with urllib.request.urlopen(test_url, timeout=10) as r:
                    test_data = r.read()
                results["input_bytes"] = len(test_data)
                for model_url in [
                    "https://router.huggingface.co/hf-inference/models/briaai/RMBG-1.4",
                    "https://router.huggingface.co/hf-inference/models/briaai/RMBG-2.0",
                ]:
                    key = model_url.split("/")[-1]
                    try:
                        req = urllib.request.Request(model_url, data=test_data[:50000], method='POST')
                        req.add_header('Content-Type', 'image/jpeg')
                        if hf_token:
                            req.add_header('Authorization', f'Bearer {hf_token}')
                        with urllib.request.urlopen(req, timeout=45) as r:
                            body = r.read()
                        results[key] = {"status": 200, "bytes": len(body)}
                    except _ue.HTTPError as e:
                        body = e.read().decode('utf-8', errors='ignore')
                        results[key] = {"status": e.code, "error": body[:200]}
                    except Exception as ex:
                        results[key] = {"error": str(ex)[:200]}
            except Exception as e:
                results["fetch_error"] = str(e)[:200]
            self._json(200, results)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)
        except Exception as e:
            self._error(400, str(e))
            return

        if self.path == "/remove-bg":
            self._handle_remove_bg(data)
        elif self.path == "/process":
            self._handle_process(data)
        else:
            self._error(404, "not found")

    def _handle_remove_bg(self, data):
        try:
            image_url = ''.join(c for c in data["image_url"] if ord(c) >= 32)
            with urllib.request.urlopen(image_url, timeout=20) as r:
                input_data = r.read()
            output = remove_bg_from_bytes(input_data)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(output)))
            self.send_cors()
            self.end_headers()
            self.wfile.write(output)
        except Exception as e:
            self._error(500, str(e))

    def _handle_process(self, data):
        try:
            image_url = ''.join(c for c in data.get("image_url", "") if ord(c) >= 32)
            category = data.get("category", "general")
            if not image_url:
                self._error(400, "image_url required")
                return
            with urllib.request.urlopen(image_url, timeout=20) as r:
                input_data = r.read()
            no_bg = remove_bg_from_bytes(input_data)
            final_webp = composite_image(no_bg, category)
            final_url = upload_to_supabase(final_webp, 'image/webp')
            self._json(200, {"ok": True, "url": final_url})
        except Exception as e:
            self._error(500, str(e))

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, msg):
        self._json(status, {"error": msg})


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
