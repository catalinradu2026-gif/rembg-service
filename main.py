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


def remove_bg_removebg(image_data: bytes) -> bytes:
    """Remove background using remove.bg API — 50 free/month, full resolution."""
    api_key = os.environ.get("REMOVEBG_KEY", "")
    if not api_key:
        raise ValueError("no REMOVEBG_KEY")
    r = req_lib.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': ('image.jpg', image_data, 'image/jpeg')},
        data={'size': 'auto'},
        headers={'X-Api-Key': api_key},
        timeout=30,
    )
    if not r.ok:
        raise ValueError(f"removebg {r.status_code}: {r.text[:150]}")
    if len(r.content) < 1000:
        raise ValueError(f"removebg response too small: {len(r.content)} bytes")
    print(f"remove.bg ok: {len(r.content)} bytes")
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


def remove_bg_photoroom(image_data: bytes) -> bytes:
    """Remove background using PhotoRoom API — 150 free/month, full resolution."""
    api_key = os.environ.get("PHOTOROOM_KEY", "")
    if not api_key:
        raise ValueError("no PHOTOROOM_KEY")
    r = req_lib.post(
        'https://sdk.photoroom.com/v1/segment',
        files={'image_file': ('image.jpg', image_data, 'image/jpeg')},
        headers={'x-api-key': api_key},
        timeout=30,
    )
    if not r.ok:
        raise ValueError(f"photoroom {r.status_code}: {r.text[:150]}")
    if len(r.content) < 1000:
        raise ValueError(f"photoroom too small: {len(r.content)} bytes")
    print(f"PhotoRoom ok: {len(r.content)} bytes")
    return r.content


def remove_bg_from_bytes(image_data: bytes) -> bytes:
    # Try PhotoRoom first (ML quality, 150/month)
    try:
        return remove_bg_photoroom(image_data)
    except Exception as e:
        print(f"PhotoRoom failed ({e}), trying remove.bg...")

    # Try remove.bg (50/month)
    try:
        return remove_bg_removebg(image_data)
    except Exception as e:
        print(f"remove.bg failed ({e}), trying HF...")

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

def make_showroom(w: int, h: int, wall_frac: float = 0.56) -> Image.Image:
    """zyAI LUXURY showroom — pure black, warm white spotlights, gold accents."""
    from PIL import ImageDraw, ImageFont, ImageFilter

    wall_h = int(h * wall_frac)

    # ── Background at half-res ────────────────────────────────────────────────
    hw, hh = max(w // 2, 1), max(h // 2, 1)
    wh_s = int(hh * wall_frac)
    Y, X = np.mgrid[0:hh, 0:hw].astype(np.float32)
    Xn = (X - hw / 2) / (hw / 2)

    pix = np.zeros((hh, hw, 3), dtype=np.float32)
    wall_m = (Y < wh_s).astype(np.float32)
    floor_m = 1 - wall_m

    # Deep dark navy base
    pix[:,:,2] += 15 * wall_m + 11 * floor_m
    pix[:,:,1] += 8  * wall_m + 7  * floor_m
    pix[:,:,0] += 5  * wall_m + 5  * floor_m

    # ── 3 cool white/blue spotlight cones ────────────────────────────────────
    spots = [
        (hw * 0.50, (200, 220, 255), 1.00),
        (hw * 0.15, (130, 100, 240), 0.60),
        (hw * 0.85, (130, 100, 240), 0.60),
    ]
    for sx, sc, strength in spots:
        dx = X - sx
        dy = np.maximum(Y, 0.5)
        cone = np.exp(-(dx / (dy * 0.30 + 1))**2) * wall_m
        beam = cone * strength * 100
        pix[:,:,0] += beam * sc[0] / 255
        pix[:,:,1] += beam * sc[1] / 255
        pix[:,:,2] += beam * sc[2] / 255

    # Floor center glow
    ft = np.clip((Y - wh_s) / max(hh - wh_s, 1), 0, 1)
    floor_glow = np.exp(-Xn**2 * 3.5) * np.exp(-ft * 7) * 38
    pix[:,:,2] += floor_glow * floor_m
    pix[:,:,0] += floor_glow * 0.35 * floor_m

    small = Image.fromarray(np.clip(pix, 0, 255).astype(np.uint8), 'RGB')
    img = small.resize((w, h), Image.BILINEAR).convert('RGBA')
    draw = ImageDraw.Draw(img)


    # ── Neon blue horizon line ────────────────────────────────────────────────
    for dy in range(-4, 5):
        a = max(0, 210 - abs(dy) * 50)
        draw.line([(0, wall_h+dy), (w, wall_h+dy)], fill=(60, 110, 255, a))

    # ── Neon side lines ───────────────────────────────────────────────────────
    for side_x in [int(w * 0.03), int(w * 0.97)]:
        draw.line([(side_x, int(wall_h*0.05)), (side_x, wall_h)],
                  fill=(70, 50, 240, 85), width=2)

    # ── Floor grid blue ───────────────────────────────────────────────────────
    vp = w // 2
    for i in range(1, 9):
        fy = wall_h + int((h - wall_h) * (i / 8)**0.52)
        draw.line([(0, fy), (w, fy)], fill=(45, 75, 200, max(5, 44 - i*5)))
    for xp in range(-130, 131, 16):
        draw.line([(vp, wall_h), (vp + int(w*xp/100), h)], fill=(38, 62, 185, 18))

    # ── zyAI.ro — LED letters on wall ────────────────────────────────────────
    fs = max(32, int(w * 0.065))
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        fnt = ImageFont.load_default()
    txt = "zyAI.ro"
    tmp_bb = ImageDraw.Draw(Image.new('RGBA', (1,1))).textbbox((0,0), txt, font=fnt)
    tw, th = tmp_bb[2]-tmp_bb[0], tmp_bb[3]-tmp_bb[1]
    pad = 35
    cw, ch = tw + pad*2 + 20, th + pad*2 + 20

    # Outer glow (wide, white-blue)
    g1 = Image.new('RGBA', (cw, ch), (0,0,0,0))
    ImageDraw.Draw(g1).text((pad, pad), txt, font=fnt, fill=(200, 220, 255, 160))
    g1 = g1.filter(ImageFilter.GaussianBlur(radius=18))

    # Mid glow
    g2 = Image.new('RGBA', (cw, ch), (0,0,0,0))
    ImageDraw.Draw(g2).text((pad, pad), txt, font=fnt, fill=(230, 240, 255, 200))
    g2 = g2.filter(ImageFilter.GaussianBlur(radius=7))

    # Tight core glow
    g3 = Image.new('RGBA', (cw, ch), (0,0,0,0))
    ImageDraw.Draw(g3).text((pad, pad), txt, font=fnt, fill=(240, 248, 255, 220))
    g3 = g3.filter(ImageFilter.GaussianBlur(radius=2))

    # Pure white face
    face = Image.new('RGBA', (cw, ch), (0,0,0,0))
    fd = ImageDraw.Draw(face)
    fd.text((pad, pad), txt, font=fnt, fill=(255, 255, 255, 255))

    led = Image.new('RGBA', (cw, ch), (0,0,0,0))
    for layer in [g1, g2, g3, face]:
        led = Image.alpha_composite(led, layer)

    px = max(0, (w - cw) // 2)
    py = max(2, int(wall_h * 0.10))
    img.alpha_composite(led, (px, py))

    return img


def draw_floor_text(bg: Image.Image, canvas_w: int, canvas_h: int, wall_h: int) -> Image.Image:
    """Render 'zyAI.ro' in perspective on the showroom floor using OpenCV warpPerspective."""
    from PIL import ImageDraw, ImageFont, ImageFilter

    fs = max(55, int(canvas_w * 0.11))
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        fnt = ImageFont.load_default()

    txt = "zyAI.ro"
    tmp = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    bb = tmp.textbbox((0, 0), txt, font=fnt)
    tw, th = bb[2] - bb[0] + 24, bb[3] - bb[1] + 24

    # Render text flat on transparent canvas
    flat = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    fd = ImageDraw.Draw(flat)
    for i in range(5, 0, -1):
        fd.text((12 + i, 12 + i), txt, font=fnt, fill=(10, 25, 110, 70 + i * 12))
    fd.text((12, 12), txt, font=fnt, fill=(85, 140, 255, 210))
    fd.text((12, 11), txt, font=fnt, fill=(170, 210, 255, 110))
    glow = flat.filter(ImageFilter.GaussianBlur(radius=6))
    flat = Image.alpha_composite(glow, flat)

    flat_arr = np.array(flat, dtype=np.uint8)

    # Perspective: source rectangle → floor trapezoid
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    cx = canvas_w // 2
    floor_h = canvas_h - wall_h
    far_y  = wall_h + int(floor_h * 0.22)
    near_y = wall_h + int(floor_h * 0.68)
    far_hw  = int(canvas_w * 0.12)
    near_hw = int(canvas_w * 0.33)
    dst = np.float32([
        [cx - far_hw,  far_y],
        [cx + far_hw,  far_y],
        [cx + near_hw, near_y],
        [cx - near_hw, near_y],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(flat_arr, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
    warped_img = Image.fromarray(warped, 'RGBA')
    bg.alpha_composite(warped_img)
    return bg


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
    from PIL import ImageFilter, ImageDraw

    subject = Image.open(io.BytesIO(subject_png)).convert('RGBA')
    cat_lower = (category or '').lower()
    is_auto = not cat_lower or cat_lower == 'general' or 'auto' in cat_lower or cat_lower in ('vehicule', 'masini', 'cars')

    if not is_auto:
        # Non-auto: simple studio, keep original size
        sw, sh = subject.size
        if max(sw, sh) > 1100:
            s = 1100 / max(sw, sh)
            sw, sh = int(sw * s), int(sh * s)
            subject = subject.resize((sw, sh), Image.LANCZOS)
        bg = make_studio(sw, sh).convert('RGBA')
        bg.paste(subject, (0, 0), subject)
        result = add_watermark(bg)
        out = io.BytesIO()
        result.convert('RGB').save(out, format='WEBP', quality=85)
        return out.getvalue()

    # ── Auto showroom ─────────────────────────────────────────────────────────
    sw, sh = subject.size
    if max(sw, sh) > 1100:
        s = 1100 / max(sw, sh)
        sw, sh = int(sw * s), int(sh * s)
        subject = subject.resize((sw, sh), Image.LANCZOS)

    import math

    # Find actual bottom pixel of car (PhotoRoom adds transparent padding)
    _a = np.array(subject)[:, :, 3]
    _rows = np.where(_a.max(axis=1) > 8)[0]
    actual_bottom = int(_rows[-1]) + 1 if len(_rows) > 0 else sh

    # Canvas: actual car height + fixed floor strip below
    floor_extra = int(sw * 0.20)
    canvas_h = actual_bottom + floor_extra
    wall_h = actual_bottom
    wall_frac = wall_h / canvas_h

    bg = make_showroom(sw, canvas_h, wall_frac=wall_frac)
    draw = ImageDraw.Draw(bg)

    # ── Turntable platform at floor line ─────────────────────────────────────
    cx, py = sw // 2, wall_h
    plat_w = int(sw * 0.68)
    plat_h = int(plat_w * 0.14)
    for ring in range(4, 0, -1):
        rw, rh = plat_w + ring*10, plat_h + ring*3
        draw.ellipse([(cx-rw//2, py-rh//2), (cx+rw//2, py+rh//2)],
                     fill=(40, 65, 200, 16*ring))
    draw.ellipse([(cx-plat_w//2, py-plat_h//2), (cx+plat_w//2, py+plat_h//2)],
                 fill=(18, 22, 38, 235))
    draw.arc([(cx-plat_w//2, py-plat_h//2), (cx+plat_w//2, py+plat_h//2)],
             start=185, end=355, fill=(80, 125, 255, 170), width=2)
    for deg in range(0, 360, 24):
        rad = math.radians(deg)
        ix = cx + int((plat_w//2 - 5) * math.cos(rad))
        iy = py + int((plat_h//2 - 2) * math.sin(rad))
        ox = cx + int((plat_w//2 + 1) * math.cos(rad))
        oy = py + int((plat_h//2 + 1) * math.sin(rad))
        draw.line([(ix, iy), (ox, oy)], fill=(60, 100, 210, 100), width=1)

    # ── Reflection ────────────────────────────────────────────────────────────
    refl_h = min(int(sh * 0.18), floor_extra - 4)
    if refl_h > 6:
        refl = subject.crop((0, sh - refl_h, sw, sh)).transpose(Image.FLIP_TOP_BOTTOM)
        refl_canvas = Image.new('RGBA', (sw, canvas_h), (0, 0, 0, 0))
        fade_arr = np.array([int(40 * (1 - i / refl_h)) for i in range(refl_h)], dtype=np.uint8)
        fade_2d = np.tile(fade_arr[:, np.newaxis], (1, sw))
        refl.putalpha(Image.fromarray(fade_2d, 'L'))
        refl_canvas.paste(refl.filter(ImageFilter.GaussianBlur(radius=3)), (0, wall_h))
        bg.alpha_composite(refl_canvas)

    # ── Car at (0,0), bottom of image = floor line ────────────────────────────
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
            has_pr = bool(os.environ.get("PHOTOROOM_KEY", ""))
            has_rbg = bool(os.environ.get("REMOVEBG_KEY", ""))
            model = ("photoroom+" if has_pr else "") + ("removebg+" if has_rbg else "") + "hf+grabcut"
            body = json.dumps({"ok": True, "model": model}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors()
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/debug-clipdrop":
            rbg_key = os.environ.get("REMOVEBG_KEY", "")
            result = {"key_set": bool(rbg_key), "key_prefix": rbg_key[:8] if rbg_key else ""}
            try:
                test_url = "https://www.gstatic.com/webp/gallery/1.jpg"
                with urllib.request.urlopen(test_url, timeout=10) as r:
                    test_data = r.read()
                result["input_bytes"] = len(test_data)
                out = remove_bg_removebg(test_data)
                result["ok"] = True
                result["output_bytes"] = len(out)
            except Exception as e:
                result["ok"] = False
                result["error"] = str(e)[:300]
            self._json(200, result)
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
