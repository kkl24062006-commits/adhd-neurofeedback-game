from http.server import BaseHTTPRequestHandler, HTTPServer
import json, csv, os, time
import numpy as np

# ==============================
# CONFIG
# ==============================
PORT = 8000
WIN = 5                  # sliding window (seconds)
NSAMPLE_THRESH = 10
EMA_ALPHA = 0.3          # smoothing factor

# ==============================
# BUFFERS
# ==============================
buffer_x = []
buffer_y = []

# ==============================
# CSV: EMA PROCESSED GAZE
# ==============================
ema_csv = open("eye_gaze_log_ema.csv", "w", newline="")
ema_writer = csv.writer(ema_csv)
ema_writer.writerow([
    "sec_index",
    "ema_x_norm",
    "ema_y_norm",
    "screen_w",
    "screen_h",
    "timestamp"
])

# ==============================
# EMA FUNCTION
# ==============================
def apply_ema(x, y):
    xf = [x[0]]
    yf = [y[0]]

    for i in range(1, len(x)):
        xf.append(EMA_ALPHA * x[i] + (1 - EMA_ALPHA) * xf[-1])
        yf.append(EMA_ALPHA * y[i] + (1 - EMA_ALPHA) * yf[-1])

    return np.array(xf), np.array(yf)

# ==============================
# HTTP HANDLER
# ==============================
class GazeHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path in ["/","/index.html"]:
            self.serve("index.html","text/html")
        elif self.path == "/webgazer.js":
            self.serve("webgazer.js","application/javascript")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/gaze":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        data = json.loads(self.rfile.read(length).decode())

        sec = data["sec_index"]

        # ---- validity check ----
        if data["n_samples"] < NSAMPLE_THRESH:
            return
        if data["x_pos"] == 0 and data["y_pos"] == 0:
            return

        # ---- normalize ----
        x = data["x_pos"] / data["screen_w"]
        y = data["y_pos"] / data["screen_h"]

        buffer_x.append(x)
        buffer_y.append(y)

        if len(buffer_x) > WIN:
            buffer_x.pop(0)
            buffer_y.pop(0)

        if len(buffer_x) < WIN:
            return

        xs = np.array(buffer_x)
        ys = np.array(buffer_y)

        # ---- Apply EMA ----
        xf, yf = apply_ema(xs, ys)

        # take latest EMA value
        ema_x = xf[-1]
        ema_y = yf[-1]

        # ---- Save to CSV ----
        ema_writer.writerow([
            sec,
            ema_x,
            ema_y,
            data["screen_w"],
            data["screen_h"],
            time.time()
        ])
        ema_csv.flush()

        print(f"Sec {sec} | EMA X={ema_x:.3f} | EMA Y={ema_y:.3f}")

        self.send_response(200)
        self.end_headers()

    def serve(self, fname, ctype):
        if not os.path.exists(fname):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        with open(fname, "rb") as f:
            self.wfile.write(f.read())

    def log_message(self, *args):
        return

# ==============================
# RUN SERVER
# ==============================
print("Eye tracker server running (EMA only)")
print(f"Open http://localhost:{PORT}")

HTTPServer(("localhost", PORT), GazeHandler).serve_forever()