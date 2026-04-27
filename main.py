from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.request
import numpy as np
import cv2

PORT = int(os.environ.get("PORT", 8002))
PROC_DIM = 900
print(f"OpenCV background removal service ready on port {PORT}")


def is_simple_background(img: np.ndarray) -> tuple[bool, np.ndarray]:
    """Check if image has a uniform/studio background by sampling corners."""
    h, w = img.shape[:2]
    margin = max(20, int(min(h, w) * 0.06))

    corners = [
        img[0:margin, 0:margin],
        img[0:margin, w - margin:w],
        img[h - margin:h, 0:margin],
        img[h - margin:h, w - margin:w],
    ]

    samples = [c.reshape(-1, 3).astype(np.float32) for c in corners]
    all_samples = np.vstack(samples)

    mean_color = all_samples.mean(axis=0)
    std_dev = all_samples.std(axis=0).mean()

    # Low std = uniform background (studio/solid color)
    is_simple = std_dev < 28
    return is_simple, mean_color


def color_key_mask(img: np.ndarray, bg_color: np.ndarray, tolerance: float = 38) -> np.ndarray:
    """Create alpha mask by color distance from background color."""
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    diff = img_f - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # Soft transition: 0 at tolerance/2, 255 at tolerance
    alpha = np.clip((dist - tolerance * 0.5) / (tolerance * 0.5) * 255, 0, 255).astype(np.uint8)

    # Flood fill from all 4 corners to capture background connected to edges
    margin = max(5, int(min(h, w) * 0.04))
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    bg_thresh = np.where(dist < tolerance * 1.2, 0, 255).astype(np.uint8)

    for cy in [0, h // 2, h - 1]:
        for cx in [0, w // 2, w - 1]:
            if bg_thresh[cy, cx] == 0:
                cv2.floodFill(bg_thresh, flood_mask, (cx, cy), 128)

    flood_bg = (bg_thresh == 128).astype(np.uint8) * 255
    flood_bg_inv = 255 - flood_bg

    # Combine: color distance + flood fill
    combined = cv2.bitwise_and(alpha, flood_bg_inv)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Largest connected component only
    _, thresh = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        thresh = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Feather edges
    feathered = cv2.GaussianBlur(thresh.astype(np.float32), (11, 11), 3)
    border = cv2.dilate(thresh, np.ones((9, 9), np.uint8)) - cv2.erode(thresh, np.ones((9, 9), np.uint8))
    final = thresh.astype(np.float32)
    final[border > 0] = feathered[border > 0]

    return np.clip(final, 0, 255).astype(np.uint8)


def grabcut_mask(img: np.ndarray) -> np.ndarray:
    """GrabCut-based background removal for complex backgrounds."""
    h, w = img.shape[:2]

    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_enh = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mx = int(w * 0.06)
    my = int(h * 0.06)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    cv2.grabCut(img_enh, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))

    # Feather edges
    feathered = cv2.GaussianBlur(alpha.astype(np.float32), (11, 11), 3)
    border = cv2.dilate(alpha, np.ones((9, 9), np.uint8)) - cv2.erode(alpha, np.ones((9, 9), np.uint8))
    final = alpha.astype(np.float32)
    final[border > 0] = feathered[border > 0]

    return np.clip(final, 0, 255).astype(np.uint8)


def remove_background(image_data: bytes) -> bytes:
    nparr = np.frombuffer(image_data, np.uint8)
    orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if orig is None:
        raise ValueError("Could not decode image")

    oh, ow = orig.shape[:2]

    # Resize for processing
    scale = min(1.0, PROC_DIM / max(oh, ow))
    if scale < 1.0:
        proc = cv2.resize(orig, (int(ow * scale), int(oh * scale)), interpolation=cv2.INTER_AREA)
    else:
        proc = orig.copy()

    # Choose algorithm based on background complexity
    simple, bg_color = is_simple_background(proc)
    if simple:
        alpha_small = color_key_mask(proc, bg_color)
    else:
        alpha_small = grabcut_mask(proc)

    # Upscale alpha to original size
    if scale < 1.0:
        alpha = cv2.resize(alpha_small, (ow, oh), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1)
    else:
        alpha = alpha_small

    b_ch, g_ch, r_ch = cv2.split(orig)
    rgba = cv2.merge([b_ch, g_ch, r_ch, alpha])
    _, buf = cv2.imencode('.png', rgba)
    return buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

    def do_POST(self):
        if self.path == "/remove-bg":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                req = json.loads(body)
                image_url = ''.join(c for c in req["image_url"] if ord(c) >= 32)

                with urllib.request.urlopen(image_url, timeout=20) as r:
                    input_data = r.read()

                output_data = remove_background(input_data)

                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(output_data)))
                self.end_headers()
                self.wfile.write(output_data)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(err)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
