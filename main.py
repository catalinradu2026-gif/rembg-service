from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.request
import numpy as np
import cv2

PORT = int(os.environ.get("PORT", 8002))
MAX_DIM = 800  # resize before GrabCut to stay under 30s timeout
print(f"OpenCV background removal service ready on port {PORT}")

def remove_background(image_data: bytes) -> bytes:
    nparr = np.frombuffer(image_data, np.uint8)
    orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if orig is None:
        raise ValueError("Could not decode image")

    oh, ow = orig.shape[:2]

    # Resize down for GrabCut processing
    scale = min(1.0, MAX_DIM / max(oh, ow))
    if scale < 1.0:
        pw = int(ow * scale)
        ph = int(oh * scale)
        proc = cv2.resize(orig, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        pw, ph = ow, oh
        proc = orig

    h, w = proc.shape[:2]

    # GrabCut with centered rectangle (8% margin treated as background)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mx = int(w * 0.08)
    my = int(h * 0.08)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    cv2.grabCut(proc, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    # Final mask: 0/2 = background, 1/3 = foreground
    alpha_small = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    # Smooth edges
    alpha_small = cv2.GaussianBlur(alpha_small, (5, 5), 0)
    _, alpha_small = cv2.threshold(alpha_small, 127, 255, cv2.THRESH_BINARY)

    # Scale alpha back to original size if we downscaled
    if scale < 1.0:
        alpha = cv2.resize(alpha_small, (ow, oh), interpolation=cv2.INTER_LINEAR)
        _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    else:
        alpha = alpha_small

    b, g, r = cv2.split(orig)
    rgba = cv2.merge([b, g, r, alpha])

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
