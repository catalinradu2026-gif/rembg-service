from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.request
import numpy as np
import cv2

PORT = int(os.environ.get("PORT", 8002))
PROC_DIM = 900  # processing resolution for GrabCut
print(f"OpenCV background removal service ready on port {PORT}")


def refine_mask(alpha: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """Refine alpha mask using edge-aware techniques."""
    # Close small holes inside the subject
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_close)

    # Remove small isolated blobs outside subject
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel_open)

    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        alpha = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Edge-aware feathering: use Canny edges from original to guide feathering
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # Soft blend only near edges
    border_zone = cv2.dilate(alpha, np.ones((9, 9), np.uint8)) - cv2.erode(alpha, np.ones((9, 9), np.uint8))
    feather_mask = cv2.GaussianBlur(alpha.astype(np.float32), (11, 11), 3)

    # Apply feathered alpha only in edge/border zone
    final = alpha.astype(np.float32)
    final[border_zone > 0] = feather_mask[border_zone > 0]
    final = np.clip(final, 0, 255).astype(np.uint8)

    return final


def remove_background(image_data: bytes) -> bytes:
    nparr = np.frombuffer(image_data, np.uint8)
    orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if orig is None:
        raise ValueError("Could not decode image")

    oh, ow = orig.shape[:2]

    # Resize for processing (keeps it fast, still good quality)
    scale = min(1.0, PROC_DIM / max(oh, ow))
    if scale < 1.0:
        pw = int(ow * scale)
        ph = int(oh * scale)
        proc = cv2.resize(orig, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        pw, ph = ow, oh
        proc = orig.copy()

    h, w = proc.shape[:2]

    # Enhance local contrast to help GrabCut distinguish subject from background
    lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    proc_enh = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # GrabCut — tighter rect leaves less background in subject area
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mx = int(w * 0.06)
    my = int(h * 0.06)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    cv2.grabCut(proc_enh, mask, rect, bgd, fgd, 7, cv2.GC_INIT_WITH_RECT)

    # Probable foreground gets another pass
    mask2 = np.where((mask == 3), cv2.GC_PR_FGD, mask).astype(np.uint8)
    cv2.grabCut(proc_enh, mask2, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)

    alpha_small = np.where((mask2 == 2) | (mask2 == 0), 0, 255).astype(np.uint8)

    # Refine on processing-size image
    alpha_small = refine_mask(alpha_small, proc)

    # Scale alpha back to original resolution
    if scale < 1.0:
        alpha = cv2.resize(alpha_small, (ow, oh), interpolation=cv2.INTER_LINEAR)
        # Re-threshold after upscale interpolation
        _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        # Final light feather on full-res
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
