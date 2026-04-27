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

def make_showroom(w: int, h: int) -> Image.Image:
    """zyAI WOW showroom: dramatic spotlight, neon floor, reflection glow, perspective grid."""
    from PIL import ImageDraw, ImageFont, ImageFilter

    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    Xn = (X - w / 2) / (w / 2)
    Yn = Y / h

    wall_frac = 0.58
    wall_h = int(h * wall_frac)

    # ── Base: pitch black ─────────────────────────────────────────────────────
    pix = np.zeros((h, w, 3), dtype=np.float32)

    # ── Main spotlight: tight white-blue beam from top center ─────────────────
    spot_tight = np.exp(-(Xn ** 2 * 12 + (Yn * 0.8) ** 2 * 2)) * 90
    spot_wide  = np.exp(-(Xn ** 2 * 2.5 + (Yn - 0.05) ** 2 * 5)) * 35
    spot = spot_tight + spot_wide

    # ── Side accent lights: purple-blue from upper left & right ──────────────
    left  = np.exp(-((Xn + 1.1) ** 2 * 1.8 + (Yn - 0.15) ** 2 * 4)) * 40
    right = np.exp(-((Xn - 1.1) ** 2 * 1.8 + (Yn - 0.15) ** 2 * 4)) * 40

    # ── Wall ─────────────────────────────────────────────────────────────────
    wall_m = (Y < wall_h).astype(np.float32)
    pix[:,:,0] += (spot * 0.55 + left * 0.6 + right * 0.5) * wall_m
    pix[:,:,1] += (spot * 0.45 + left * 0.3 + right * 0.3) * wall_m
    pix[:,:,2] += (spot * 1.0  + left * 1.0 + right * 1.0) * wall_m

    # ── Floor: dark with strong center glow (reflection) ─────────────────────
    floor_m = 1 - wall_m
    ft = np.clip((Y - wall_h) / max(h - wall_h, 1), 0, 1)
    refl = np.exp(-Xn ** 2 * 5) * np.exp(-ft * 6) * 70
    pix[:,:,0] += (refl * 0.4) * floor_m
    pix[:,:,1] += (refl * 0.3) * floor_m
    pix[:,:,2] += (refl * 1.0) * floor_m

    # ── Vignette: dark corners ────────────────────────────────────────────────
    vig = np.clip(1 - (Xn ** 2 * 0.6 + (Yn - 0.45) ** 2 * 0.4), 0.08, 1)
    pix *= vig[:, :, np.newaxis]

    img = Image.fromarray(np.clip(pix, 0, 255).astype(np.uint8), 'RGB').convert('RGBA')
    draw = ImageDraw.Draw(img)

    # ── Light beam cone (soft triangle from top) ──────────────────────────────
    beam_img = Image.new('RGBA', (w, wall_h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(beam_img)
    cx = w // 2
    for rad in range(int(w * 0.45), 0, -3):
        a = max(0, int(8 * (1 - rad / (w * 0.45))))
        bd.polygon([(cx, 0), (cx - rad, wall_h), (cx + rad, wall_h)],
                   fill=(140, 180, 255, a))
    beam_blur = beam_img.filter(ImageFilter.GaussianBlur(radius=8))
    img.alpha_composite(beam_blur, (0, 0))

    # ── Glowing horizon line (wall meets floor) ───────────────────────────────
    for dy in range(-4, 5):
        a = max(0, 200 - abs(dy) * 50)
        b_c = min(255, 200 + abs(dy) * 10)
        draw.line([(0, wall_h + dy), (w, wall_h + dy)], fill=(60, 100, b_c, a))

    # ── Neon side lines on wall ───────────────────────────────────────────────
    for dx in [int(w * 0.05), int(w * 0.95)]:
        for dy2 in range(-1, 2):
            a2 = 80 - abs(dy2) * 30
            draw.line([(dx + dy2, int(wall_h * 0.1)), (dx + dy2, wall_h)],
                      fill=(80, 60, 255, a2), width=1)

    # ── Floor perspective grid ────────────────────────────────────────────────
    vp_x = w // 2
    for i in range(1, 9):
        fy = wall_h + int((h - wall_h) * (i / 8) ** 0.65)
        a3 = max(10, 55 - i * 6)
        draw.line([(0, fy), (w, fy)], fill=(50, 80, 200, a3), width=1)
    for xp in range(-130, 131, 18):
        xe = w // 2 + int(w * xp / 100)
        draw.line([(vp_x, wall_h), (xe, h)], fill=(40, 65, 190, 28), width=1)

    # ── zyAI.ro on floor (large, elegant) ────────────────────────────────────
    fs = max(18, int(w * 0.055))
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        fnt = ImageFont.load_default()
    txt = "zyAI.ro"
    bb = draw.textbbox((0, 0), txt, font=fnt)
    bw2 = bb[2] - bb[0]
    bx2 = (w - bw2) // 2
    by2 = wall_h + int((h - wall_h) * 0.65)
    draw.text((bx2 + 1, by2 + 1), txt, font=fnt, fill=(0, 0, 80, 60))
    draw.text((bx2, by2), txt, font=fnt, fill=(100, 150, 255, 55))

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
    sw, sh = subject.size
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
            model = "rembg-silueta" if _rembg_session else "grabcut-fallback"
            body = json.dumps({"ok": True, "model": model}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors()
            self.end_headers()
            self.wfile.write(body)

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
