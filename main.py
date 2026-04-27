from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.request
import numpy as np
import cv2

PORT = int(os.environ.get("PORT", 8002))
print(f"OpenCV background removal service ready on port {PORT}")

def remove_background(image_data: bytes) -> bytes:
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w = img.shape[:2]

    # GrabCut cu dreptunghi centrat (marginea 8% ignorată ca fundal)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mx = int(w * 0.08)
    my = int(h * 0.08)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    cv2.grabCut(img, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)

    # Masca finală: 0/2 = fundal, 1/3 = prim-plan
    alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    # Smooth edges
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(img)
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
