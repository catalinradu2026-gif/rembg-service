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

# Try loading rembg with silueta model (44MB, lightweight neural network)
_rembg_session = None
try:
    from rembg import new_session, remove as rembg_remove
    _rembg_session = new_session('silueta')
    print("rembg silueta model loaded OK")
except Exception as e:
    print(f"rembg not available, using GrabCut fallback: {e}")
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
    mx, my = int(w * 0.06), int(h * 0.06)
    cv2.grabCut(img_enh, mask, (mx, my, w - 2*mx, h - 2*my), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    alpha = cv2.morphologyEx(cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k1), cv2.MORPH_OPEN, k2)
    blur = cv2.GaussianBlur(alpha.astype(np.float32), (11, 11), 3)
    border = cv2.dilate(alpha, np.ones((9, 9), np.uint8)) - cv2.erode(alpha, np.ones((9, 9), np.uint8))
    final = alpha.astype(np.float32)
    final[border > 0] = blur[border > 0]
    return np.clip(final, 0, 255).astype(np.uint8)


def remove_bg_from_bytes(image_data: bytes) -> bytes:
    # Use rembg neural model if available
    if _rembg_session is not None:
        try:
            return rembg_remove(image_data, session=_rembg_session)
        except Exception as e:
            print(f"rembg failed, falling back to GrabCut: {e}")

    # GrabCut fallback
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

def make_background(w: int, h: int, is_auto: bool) -> Image.Image:
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        if is_auto:
            r_v = int(13 + t * 5)
            g_v = int(13 + t * 5)
            b_v = max(5, int(26 - t * 21))
            pixels[y, :] = [r_v, g_v, b_v]
        else:
            cx_f = np.arange(w, dtype=np.float32)
            cy_f = y
            dx = (cx_f - w / 2) / (w / 2)
            dy = (cy_f - h * 0.3) / h
            d = np.clip(np.sqrt(dx**2 + dy**2) / 0.65, 0, 1)
            v = np.clip(255 - d * 30, 220, 255).astype(np.uint8)
            b_arr = np.clip(v - d * 10 + 10, 215, 255).astype(np.uint8)
            pixels[y, :, 0] = v
            pixels[y, :, 1] = v
            pixels[y, :, 2] = b_arr
    return Image.fromarray(pixels, 'RGB')


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
    sw, sh = subject.size
    cat_lower = (category or '').lower()
    # Default to auto (dark showroom) when no category or unknown — primary use case
    is_auto = not cat_lower or cat_lower == 'general' or 'auto' in cat_lower or cat_lower in ('vehicule', 'masini', 'cars')
    bg = make_background(sw, sh, is_auto).convert('RGBA')

    if is_auto:
        # Elliptical podium with glowing rim
        podium_y = int(sh * 0.75)
        podium_w = int(sw * 0.85)
        podium_h = int(sh * 0.07)
        pd = np.zeros((podium_h, podium_w, 4), dtype=np.uint8)
        for py in range(podium_h):
            for px in range(podium_w):
                ex = (px - podium_w / 2) / (podium_w / 2)
                ey = (py - podium_h / 2) / (podium_h / 2)
                d2 = ex*ex + ey*ey
                if d2 <= 1:
                    t = py / podium_h
                    rim = max(0, 1 - d2 * 4)  # glow near edge
                    base_b = int(106 + t*(48-106))
                    pd[py, px] = [
                        min(255, int(42 + t*(17-42)) + int(rim * 80)),
                        min(255, int(42 + t*(17-42)) + int(rim * 80)),
                        min(255, base_b + int(rim * 120)),
                        min(255, 180 + int(rim * 75))
                    ]
        podium = Image.fromarray(pd, 'RGBA')
        bg.paste(podium, (int((sw - podium_w) / 2), podium_y - podium_h // 2), podium)

    bg.paste(subject, (0, 0), subject)
    result = add_watermark(bg)
    out = io.BytesIO()
    result.convert('RGB').save(out, format='WEBP', quality=82)
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
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors()
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

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
