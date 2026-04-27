from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.request
from rembg import remove, new_session

PORT = int(os.environ.get("PORT", 8002))
session = new_session("u2netp")
print(f"rembg model loaded, ready on port {PORT}")

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
                image_url = req["image_url"]

                with urllib.request.urlopen(image_url, timeout=20) as r:
                    input_data = r.read()

                output_data = remove(input_data, session=session)

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
