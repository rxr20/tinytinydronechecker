"""
DroneCheck demo server.

- serves the web page (static/)
- lists the images in camsim/ (the simulated camera feed)
- classifies a frame with the trained model (../model/drone_check.pth)

Run:  python server.py   ->  open http://localhost:47613
"""
import json
import mimetypes
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "model" / "drone_check.pth"
CAMSIM_DIR = HERE / "camsim"
STATIC_DIR = HERE / "static"
PORT = 47613
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------- load the model ----------
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
CLASSES = checkpoint["classes"]
IMAGE_SIZE = checkpoint["image_size"]

model = models.get_model(checkpoint["architecture"], weights=None)
if hasattr(model, "fc"):
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
else:
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Loaded {checkpoint['architecture']} | classes: {CLASSES}")

# same preparation as the training data: 80% center square, 256px, light sharpen
to_tensor = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def prepare(image):
    image = image.convert("RGB")
    w, h = image.size
    side = int(min(w, h) * 0.8)
    left, top = (w - side) // 2, (h - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((256, 256), Image.LANCZOS)
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))


def predict(path):
    tensor = to_tensor(prepare(Image.open(path))).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    best = probs.argmax().item()
    return {
        "label": CLASSES[best],
        "confidence": round(probs[best].item(), 4),
        "probs": {c: round(p.item(), 4) for c, p in zip(CLASSES, probs)},
    }


def camsim_frames():
    return sorted(f.name for f in CAMSIM_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS)


# ---------- web server ----------
class Handler(SimpleHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        if not path.is_file():
            return self.send_error(404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)

        if url.path == "/":
            return self.send_file(STATIC_DIR / "index.html")

        if url.path == "/api/frames":
            return self.send_json(camsim_frames())

        if url.path == "/api/predict":
            name = parse_qs(url.query).get("frame", [""])[0]
            if name not in camsim_frames():
                return self.send_json({"error": "unknown frame"}, 404)
            return self.send_json(predict(CAMSIM_DIR / name))

        if url.path.startswith("/camsim/"):
            name = unquote(Path(url.path).name)   # "camsim%20(1).jpg" -> "camsim (1).jpg"
            return self.send_file(CAMSIM_DIR / name) if name in camsim_frames() else self.send_error(404)

        if url.path.startswith("/static/"):
            path = (STATIC_DIR / unquote(url.path[len("/static/"):])).resolve()
            return self.send_file(path) if STATIC_DIR in path.parents else self.send_error(404)

        self.send_error(404)

    def log_message(self, *args):
        pass  # keep the console quiet


if __name__ == "__main__":
    CAMSIM_DIR.mkdir(exist_ok=True)
    server = HTTPServer(("localhost", PORT), Handler)   # server is listening from here
    print(f"DroneCheck demo running on http://localhost:{PORT}")
    webbrowser.open(f"http://localhost:{PORT}")
    server.serve_forever()
