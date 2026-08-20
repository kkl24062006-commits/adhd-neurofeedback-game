"""
ADHD Neurofeedback Serious Game — Multi-Protocol Edition
=========================================================
Four neurofeedback protocols: Theta/Beta, Beta, SMR, Alpha/Theta.
Patient profile, protocol selection, adaptive difficulty, fullscreen,
session dashboard with plots and exportable report.

RUN:
    python adhd_neurofeedback_game.py --live         # live EEG from QNeuro
    python adhd_neurofeedback_game.py --sim          # keyboard test (SPACE)
    python adhd_neurofeedback_game.py                # CSV replay

Session data → data/session_log.csv
Dashboard reports → data/report_TIMESTAMP.png
"""

import argparse, csv, io, json, math, os, random, sys, threading, time
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt, welch
import pygame

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ── paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_FILE      = os.path.join(DATA_DIR, "session_log.csv")
PROFILE_FILE  = os.path.join(DATA_DIR, "last_profile.json")
BANDPOWER_CSV = os.path.join(DATA_DIR, "Normalized_BandPower.csv")

# ── EEG config ──
EEG_STREAM_NAME   = "NEW1_EEG"
EEG_FS            = 250
WINDOW_SECONDS    = 2       # 2 s — optimal for low latency + stable PSD
EEG_UPDATE_INTERVAL = 0.5   # 50 % overlap → new value every 500 ms
CHANNEL_NAMES = ["FP1","FP2","O1","O2","C3","CZ","FZ","C4"]

# ── display (overridden to fullscreen in main) ──
W, H = 1280, 720
FPS  = 0   # 0 = uncapped / vsync
PLAYBACK_INTERVAL      = 0.15
CALIBRATION_SEC        = 6
REALTIME_CALIBRATION_SEC = 20

# ── protocols ──
PROTOCOLS = {
    "theta_beta": {
        "name": "Theta / Beta",
        "short": "TBR",
        "desc": "Reduces theta, increases beta — core ADHD attention training.",
        "channels": ("FZ","CZ"),
        "metric_fn": "tbr",      # theta / beta — LOWER = better
        "direction": "lower",
        "clinical": "Inattention, hyperactivity, impulsivity",
    },
    "beta": {
        "name": "Beta",
        "short": "Beta",
        "desc": "Enhances beta band power (13-30 Hz) for focus and cognition.",
        "channels": ("FZ","CZ"),
        "metric_fn": "beta",     # absolute beta power — HIGHER = better
        "direction": "higher",
        "clinical": "Attention, focus, cognition, behaviour",
    },
    "smr": {
        "name": "SMR",
        "short": "SMR",
        "desc": "Increases sensorimotor rhythm (12-15 Hz) over motor cortex.",
        "channels": ("C3","CZ","C4"),
        "metric_fn": "smr",      # power 12-15 Hz — HIGHER = better
        "direction": "higher",
        "clinical": "Hyperactivity, impulsivity, focus",
    },
    "alpha_theta": {
        "name": "Alpha / Theta",
        "short": "ATR",
        "desc": "Increases alpha, reduces theta — relaxation and stress relief.",
        "channels": ("FZ","CZ"),
        "metric_fn": "atr",      # alpha / theta — HIGHER = better
        "direction": "higher",
        "clinical": "ADHD, ASD, stress, depression, addiction",
    },
}
PROTOCOL_ORDER = ["theta_beta","beta","smr","alpha_theta"]

DIFFICULTY = {
    1: {"threshold_pct": 0.08, "sustain_sec": 1.0, "distractors": 0, "label": "Easy"},
    2: {"threshold_pct": 0.15, "sustain_sec": 1.5, "distractors": 3, "label": "Medium"},
    3: {"threshold_pct": 0.22, "sustain_sec": 2.5, "distractors": 5, "label": "Hard"},
    4: {"threshold_pct": 0.30, "sustain_sec": 3.5, "distractors": 8, "label": "Expert"},
}

# ── colours ──
BG_DARK     = (10,14,28)
CARD_BG     = (18,26,52)
CARD_SEL    = (28,42,78)
ROCKET_BODY = (230,230,245)
ROCKET_NOSE = (80,180,255)
ROCKET_FIN  = (60,140,220)
FLAME_CORE  = (255,220,80)
FLAME_MID   = (255,140,40)
FLAME_OUTER = (255,60,20)
BAR_BG      = (30,36,60)
BAR_FOCUS   = (80,220,160)
BAR_LOW     = (220,140,60)
GOAL_LINE   = (80,200,140)
TEXT_WHITE   = (230,230,240)
TEXT_DIM     = (140,150,180)
ACCENT      = (80,180,255)
ACCENT_GLOW = (60,120,200)
SCORE_POP   = (255,220,80)
DISTRACT_CLR= [(140,90,180),(180,80,130),(90,130,190)]
PANEL_BG    = (18,24,50,200)
EXIT_RED    = (200,60,60)
EXIT_RED_H  = (230,80,80)
ORANGE      = (232,118,58)
GREEN       = (80,220,160)
TEAL        = (52,211,176)


# ══════════════════════════════════════════════════════
#  PROTOCOL ENGINE
# ══════════════════════════════════════════════════════

class ProtocolEngine:
    """Computes the primary metric + EI from filtered EEG for any protocol."""

    def __init__(self, protocol_key, fs=EEG_FS):
        self.proto = PROTOCOLS[protocol_key]
        self.fs = fs
        self.ch_idx = [CHANNEL_NAMES.index(c) for c in self.proto["channels"]]
        self.direction = self.proto["direction"]
        self._metric_fn = self.proto["metric_fn"]

    def _bp(self, sig, lo, hi):
        freqs, psd = welch(sig, self.fs, nperseg=min(len(sig), self.fs * 2))
        res = freqs[1] - freqs[0]
        idx = (freqs >= lo) & (freqs <= hi)
        return np.sum(psd[idx]) * res

    def compute(self, filt_data):
        """Returns (primary_metric, engagement_index) from filtered EEG array."""
        thetas, alphas, betas, smrs = [], [], [], []
        for idx in self.ch_idx:
            sig = filt_data[:, idx]
            m, s = np.mean(sig), np.std(sig)
            sig = np.clip(sig, m - 3*s, m + 3*s)  # artifact rejection
            thetas.append(self._bp(sig, 4, 8))
            alphas.append(self._bp(sig, 8, 13))
            betas.append(self._bp(sig, 13, 30))
            smrs.append(self._bp(sig, 12, 15))

        theta = float(np.mean(thetas))
        alpha = float(np.mean(alphas))
        beta  = float(np.mean(betas))
        smr   = float(np.mean(smrs))

        # Engagement Index (always computed)
        ei = beta / (alpha + theta + 1e-9)

        # Primary metric per protocol
        if self._metric_fn == "tbr":
            primary = theta / (beta + 1e-9)
        elif self._metric_fn == "beta":
            primary = beta
        elif self._metric_fn == "smr":
            primary = smr
        elif self._metric_fn == "atr":
            primary = alpha / (theta + 1e-9)
        else:
            primary = theta / (beta + 1e-9)

        return primary, ei

    def metric_to_focus(self, value, baseline_mu, baseline_sd):
        """Convert primary metric to 0-1 focus score relative to baseline."""
        if self.direction == "lower":
            # lower is better (TBR) → focus rises as value drops below baseline
            z = (baseline_mu - value) / (baseline_sd + 1e-9)
        else:
            # higher is better (beta, SMR, ATR) → focus rises as value exceeds baseline
            z = (value - baseline_mu) / (baseline_sd + 1e-9)
        return float(np.clip(0.5 + z * 0.25, 0.0, 1.0))


# ══════════════════════════════════════════════════════
#  EEG SOURCES
# ══════════════════════════════════════════════════════

class RealtimeEEGSource:
    is_realtime = True

    def __init__(self, engine, stream_name=EEG_STREAM_NAME, fs=EEG_FS,
                 window_sec=WINDOW_SECONDS, update_interval=EEG_UPDATE_INTERVAL):
        from pylsl import StreamInlet, resolve_stream
        print(f"Connecting to LSL stream '{stream_name}' ...")
        streams = resolve_stream("name", stream_name)
        if not streams:
            raise RuntimeError(f"EEG stream '{stream_name}' not found.")
        self.inlet = StreamInlet(streams[0])
        self.fs = fs
        self.window = fs * window_sec
        self.update_interval = update_interval
        self.engine = engine
        self.buffer = deque(maxlen=self.window)
        self.b, self.a = butter(4, [0.5/(fs/2), 30/(fs/2)], btype="band")

        self._lock = threading.Lock()
        self._focus = 0.5
        self._primary = 0.0
        self._ei = 0.0
        self._updated = False
        self._baseline_mu = None
        self._baseline_sd = 1.0
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("Connected.")

    def _loop(self):
        last = 0.0
        while self._running:
            sample, _ = self.inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue
            self.buffer.append(sample)
            now = time.time()
            if len(self.buffer) >= self.window and now - last >= self.update_interval:
                last = now
                try:
                    data = np.array(self.buffer)
                    data = data - np.mean(data, axis=0)
                    filt = filtfilt(self.b, self.a, data, axis=0)
                    primary, ei = self.engine.compute(filt)
                except Exception:
                    continue
                if self._baseline_mu is not None:
                    focus = self.engine.metric_to_focus(
                        primary, self._baseline_mu, self._baseline_sd)
                else:
                    focus = 0.5
                with self._lock:
                    self._focus = focus
                    self._primary = primary
                    self._ei = ei
                    self._updated = True

    def is_ready(self):
        return len(self.buffer) >= self.window

    def set_baseline(self, mu, sd):
        with self._lock:
            self._baseline_mu = mu
            self._baseline_sd = max(sd, 1e-3)

    def update(self):
        with self._lock:
            if not self._updated:
                return None
            self._updated = False
            return self._focus, self._primary, self._ei

    def stop(self):
        self._running = False


class SimulatedEEG:
    is_realtime = False
    def __init__(self, engine):
        self._focus = 0.3
        self._target = 0.3
        self.engine = engine
    def set_key(self, held):
        self._target = 0.85 if held else 0.25
    def update(self):
        self._focus += (self._target - self._focus) * 0.06
        primary = max(0.1, 4.0*(1 - self._focus)) if self.engine.direction=="lower" \
                  else self._focus * 100
        ei = 0.3 + self._focus * 0.7
        return self._focus, primary, ei


class CSVBandPowerSource:
    is_realtime = False
    def __init__(self, engine, csv_path=BANDPOWER_CSV, interval=PLAYBACK_INTERVAL):
        self.interval = interval
        self.engine = engine
        self._idx = 0
        self._last_t = 0.0
        self._focus = 0.5
        chs = engine.proto["channels"]

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)

        rel = any(h and h.endswith("_Relative_Z") for h in header)
        sfx = "_Relative_Z" if rel else ""

        self._metrics = []
        for row in rows:
            th = np.mean([float(row[f"{c}_Theta{sfx}"]) for c in chs])
            al = np.mean([float(row.get(f"{c}_Alpha{sfx}","0")) for c in chs])
            be = np.mean([float(row[f"{c}_Beta{sfx}"]) for c in chs])
            ei = be / (al + th + 1e-9)
            fn = engine._metric_fn
            if fn == "tbr":   p = th / (be + 1e-9)
            elif fn == "beta": p = be
            elif fn == "smr":  p = be * 0.35
            elif fn == "atr":  p = al / (th + 1e-9)
            else: p = th / (be + 1e-9)
            self._metrics.append((p, ei))

        ps = [m[0] for m in self._metrics]
        self._pmin, self._pmax = min(ps), max(ps)
        self._n = len(self._metrics)
        print(f"Loaded {self._n} CSV rows.")

    def update(self):
        now = time.time()
        if now - self._last_t < self.interval:
            return None
        self._last_t = now
        p, ei = self._metrics[self._idx]
        self._idx = (self._idx + 1) % self._n
        rng = self._pmax - self._pmin or 1
        if self.engine.direction == "lower":
            raw = 1.0 - (p - self._pmin) / rng
        else:
            raw = (p - self._pmin) / rng
        self._focus += (raw - self._focus) * 0.35
        return self._focus, p, ei


# ══════════════════════════════════════════════════════
#  VISUAL HELPERS
# ══════════════════════════════════════════════════════

class Star:
    def __init__(self, max_x, max_y):
        self.max_x, self.max_y = max_x, max_y
        self.reset(); self.y = random.randint(0, max_y)
    def reset(self):
        self.x = random.randint(0, self.max_x)
        self.y = -2
        self.speed = random.uniform(20, 100)
        self.brightness = random.randint(80, 255)
        self.size = 1 if self.speed < 60 else 2
    def update(self, dt, rs=0):
        self.y += (self.speed + rs*40)*dt
        if self.y > self.max_y+5: self.reset()
    def draw(self, surf):
        c = min(255, self.brightness)
        pygame.draw.circle(surf, (c,c,min(255,c+30)), (int(self.x),int(self.y)), self.size)


class Particle:
    def __init__(self,x,y,color,vel=None,life=0.6,size=4):
        self.x,self.y=x,y
        self.vx=vel[0] if vel else random.uniform(-40,40)
        self.vy=vel[1] if vel else random.uniform(-80,-20)
        self.life=self.max_life=life; self.color=color; self.size=size
    def update(self,dt):
        self.x+=self.vx*dt; self.y+=self.vy*dt; self.vy+=60*dt; self.life-=dt
    def draw(self,surf):
        a=max(0,self.life/self.max_life); r=max(1,int(self.size*a))
        c=tuple(int(ch*a) for ch in self.color)
        pygame.draw.circle(surf,c,(int(self.x),int(self.y)),r)


class ScorePopup:
    def __init__(self,x,y,text,color=SCORE_POP):
        self.x,self.y=x,y; self.text=text; self.color=color
        self.life=self.max_life=1.0
    def update(self,dt): self.y-=40*dt; self.life-=dt
    def draw(self,surf,font):
        if self.life<=0: return
        a=max(0,self.life/self.max_life)
        c=tuple(int(ch*a) for ch in self.color)
        txt=font.render(self.text,True,c)
        surf.blit(txt,(int(self.x)-txt.get_width()//2,int(self.y)))


class Distractor:
    def __init__(self,mx,my):
        self.x=random.randint(30,mx-30); self.y=random.randint(80,my//2)
        self.vx=random.choice([-1,1])*random.uniform(30,80)
        self.radius=random.randint(10,18)
        self.color=random.choice(DISTRACT_CLR)
        self.phase=random.uniform(0,math.pi*2)
        self.mx=mx
    def update(self,dt,t):
        self.x+=self.vx*dt; self.y+=math.sin(t*2+self.phase)*20*dt
        if self.x<10 or self.x>self.mx-10: self.vx*=-1
    def draw(self,surf,t):
        pulse=1.0+0.15*math.sin(t*4+self.phase); r=int(self.radius*pulse)
        pygame.draw.circle(surf,self.color,(int(self.x),int(self.y)),r)
        inner=tuple(min(255,c+60) for c in self.color)
        pygame.draw.circle(surf,inner,(int(self.x),int(self.y)),max(2,r//2))


def draw_rocket(surf,cx,cy,focus,t):
    pygame.draw.polygon(surf,ROCKET_BODY,[(cx,cy-28),(cx-14,cy+16),(cx+14,cy+16)])
    pygame.draw.polygon(surf,ROCKET_NOSE,[(cx,cy-28),(cx-8,cy-12),(cx+8,cy-12)])
    pygame.draw.polygon(surf,ROCKET_FIN,[(cx-14,cy+16),(cx-22,cy+24),(cx-10,cy+8)])
    pygame.draw.polygon(surf,ROCKET_FIN,[(cx+14,cy+16),(cx+22,cy+24),(cx+10,cy+8)])
    pygame.draw.circle(surf,ACCENT,(cx,int(cy-4)),5)
    pygame.draw.circle(surf,(200,230,255),(cx,int(cy-4)),3)
    fh=8+focus*30; fl=math.sin(t*25)*3+math.sin(t*37)*2
    for i,(clr,wf) in enumerate([(FLAME_OUTER,.9),(FLAME_MID,.6),(FLAME_CORE,.3)]):
        fw=int(12*wf); ffh=int(fh*(1-i*.2)+fl*(1-i*.3))
        pygame.draw.polygon(surf,clr,[(cx,cy+16+ffh),(cx-fw,cy+16),(cx+fw,cy+16)])


def draw_focus_bar(surf,x,y,w,h,focus,target):
    pygame.draw.rect(surf,BAR_BG,(x,y,w,h),border_radius=4)
    fw=int(w*focus)
    if fw>0:
        c=BAR_FOCUS if focus>=target else BAR_LOW
        pygame.draw.rect(surf,c,(x,y,fw,h),border_radius=4)
    tx=x+int(w*target)
    pygame.draw.line(surf,TEXT_WHITE,(tx,y-3),(tx,y+h+3),2)


def draw_panel(surf,rect):
    p=pygame.Surface((rect[2],rect[3]),pygame.SRCALPHA)
    pygame.draw.rect(p,PANEL_BG,(0,0,rect[2],rect[3]),border_radius=12)
    surf.blit(p,(rect[0],rect[1]))


def mpl_to_surface(fig, dpi=100):
    """Render matplotlib figure to a pygame surface."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    surf = pygame.image.load(buf, "png")
    plt.close(fig)
    return surf


# ══════════════════════════════════════════════════════
#  PATIENT PROFILE SCREEN
# ══════════════════════════════════════════════════════

def load_profile():
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE) as f: return json.load(f)
        except: pass
    return {"name":"","age":"","gender":"M"}

def save_profile(p):
    with open(PROFILE_FILE,"w") as f: json.dump(p,f)

def profile_screen(screen, clock):
    prof = load_profile()
    fields = [("Name", "name"), ("Age", "age")]
    genders = ["M","F"]
    active_field = 0  # 0=name, 1=age, 2=gender
    stars = [Star(W,H) for _ in range(50)]

    while True:
        dt = clock.tick(30)/1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return None
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    if prof["name"].strip():
                        save_profile(prof); return prof
                elif e.key == pygame.K_TAB:
                    active_field = (active_field+1) % 3
                elif e.key == pygame.K_BACKSPACE:
                    if active_field < 2:
                        key = fields[active_field][1]
                        prof[key] = prof[key][:-1]
                elif active_field == 2:
                    if e.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        prof["gender"] = genders[1-genders.index(prof["gender"])]
                else:
                    if e.unicode and e.unicode.isprintable():
                        key = fields[active_field][1]
                        if active_field == 1 and not e.unicode.isdigit():
                            continue
                        prof[key] += e.unicode

        for s in stars: s.update(dt)
        screen.fill(BG_DARK)
        for s in stars: s.draw(screen)

        ft = pygame.font.SysFont("arial",36,bold=True)
        fl = pygame.font.SysFont("arial",18)
        fi = pygame.font.SysFont("arial",22)

        t = ft.render("Patient Profile",True,TEXT_WHITE)
        screen.blit(t,(W//2-t.get_width()//2, H//4 - 60))

        cy = H//4 + 10
        for i,(label,key) in enumerate(fields):
            color = ACCENT if active_field==i else TEXT_DIM
            lbl = fl.render(label, True, color)
            screen.blit(lbl, (W//2 - 160, cy))
            box = pygame.Rect(W//2 - 160, cy+25, 320, 36)
            pygame.draw.rect(screen, CARD_BG, box, border_radius=6)
            pygame.draw.rect(screen, color, box, 2, border_radius=6)
            val = fi.render(prof[key] + ("│" if active_field==i else ""), True, TEXT_WHITE)
            screen.blit(val, (box.x+10, box.y+6))
            cy += 80

        # gender toggle
        color = ACCENT if active_field==2 else TEXT_DIM
        lbl = fl.render("Gender", True, color)
        screen.blit(lbl, (W//2-160, cy))
        for gi, g in enumerate(genders):
            bx = W//2-160+gi*100
            r = pygame.Rect(bx, cy+25, 80, 36)
            sel = prof["gender"]==g
            pygame.draw.rect(screen, ACCENT if sel else CARD_BG, r, border_radius=6)
            if not sel: pygame.draw.rect(screen, TEXT_DIM, r, 1, border_radius=6)
            gt = fi.render(g, True, BG_DARK if sel else TEXT_DIM)
            screen.blit(gt, (r.centerx-gt.get_width()//2, r.centery-gt.get_height()//2))

        # continue button
        cy += 100
        br = pygame.Rect(W//2-80, cy, 160, 46)
        pygame.draw.rect(screen, ACCENT, br, border_radius=10)
        bt = fi.render("Continue →", True, BG_DARK)
        screen.blit(bt, (br.centerx-bt.get_width()//2, br.centery-bt.get_height()//2))

        hint = fl.render("TAB to switch fields  |  ENTER to continue", True, TEXT_DIM)
        screen.blit(hint, (W//2-hint.get_width()//2, H-60))

        pygame.display.flip()


# ══════════════════════════════════════════════════════
#  PROTOCOL SELECT SCREEN
# ══════════════════════════════════════════════════════

def protocol_screen(screen, clock):
    sel = 0
    stars = [Star(W,H) for _ in range(50)]

    while True:
        dt = clock.tick(30)/1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return None
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    return PROTOCOL_ORDER[sel]
                if e.key == pygame.K_LEFT:  sel = max(0, sel-1)
                if e.key == pygame.K_RIGHT: sel = min(3, sel+1)
                if e.key == pygame.K_UP:    sel = max(0, sel-2)
                if e.key == pygame.K_DOWN:  sel = min(3, sel+2)
            if e.type == pygame.MOUSEBUTTONDOWN:
                for i, r in enumerate(card_rects):
                    if r.collidepoint(e.pos): sel = i
                if start_r.collidepoint(e.pos):
                    return PROTOCOL_ORDER[sel]

        for s in stars: s.update(dt)
        screen.fill(BG_DARK)
        for s in stars: s.draw(screen)

        ft = pygame.font.SysFont("arial",34,bold=True)
        fs = pygame.font.SysFont("arial",16)
        fb = pygame.font.SysFont("arial",22,bold=True)
        fi = pygame.font.SysFont("arial",13)

        t = ft.render("Select Neurofeedback Protocol", True, TEXT_WHITE)
        screen.blit(t, (W//2 - t.get_width()//2, H//6 - 30))

        cw, ch = min(280, W//5), 200
        gap = 20
        total = 4*cw + 3*gap
        x0 = W//2 - total//2
        y0 = H//3

        card_rects = []
        for i, key in enumerate(PROTOCOL_ORDER):
            p = PROTOCOLS[key]
            col = i % 4
            x = x0 + col*(cw+gap)
            r = pygame.Rect(x, y0, cw, ch)
            card_rects.append(r)
            bg = CARD_SEL if i==sel else CARD_BG
            pygame.draw.rect(screen, bg, r, border_radius=10)
            if i==sel:
                pygame.draw.rect(screen, ACCENT, r, 2, border_radius=10)
            nm = fb.render(p["name"], True, TEXT_WHITE)
            screen.blit(nm, (r.centerx - nm.get_width()//2, r.y + 18))
            # desc wrapped
            words = p["desc"].split()
            lines, line = [], ""
            for w_txt in words:
                test = line + " " + w_txt if line else w_txt
                if fs.size(test)[0] > cw - 20:
                    lines.append(line); line = w_txt
                else: line = test
            if line: lines.append(line)
            for li, ln in enumerate(lines[:4]):
                lt = fi.render(ln, True, TEXT_DIM)
                screen.blit(lt, (r.x+10, r.y+52+li*18))
            # channels
            ch_txt = fi.render(f"Channels: {', '.join(p['channels'])}", True, ACCENT)
            screen.blit(ch_txt, (r.x+10, r.y+ch-40))
            # clinical
            cl_txt = fi.render(p["clinical"][:35], True, TEAL)
            screen.blit(cl_txt, (r.x+10, r.y+ch-22))

        # start button
        start_r = pygame.Rect(W//2-80, y0+ch+40, 160, 46)
        pygame.draw.rect(screen, ACCENT, start_r, border_radius=10)
        st = fb.render("SELECT", True, BG_DARK)
        screen.blit(st, (start_r.centerx-st.get_width()//2,
                         start_r.centery-st.get_height()//2))

        hint = fi.render("← → to browse   |   ENTER / click to select", True, TEXT_DIM)
        screen.blit(hint, (W//2-hint.get_width()//2, H-50))
        pygame.display.flip()


# ══════════════════════════════════════════════════════
#  DIFFICULTY SELECT SCREEN
# ══════════════════════════════════════════════════════

def difficulty_screen(screen, clock, proto_name, source_label):
    fs_font = pygame.font.SysFont("arial",18)
    fb = pygame.font.SysFont("arial",26,bold=True)
    fi = pygame.font.SysFont("arial",15)
    ft = pygame.font.SysFont("arial",34,bold=True)
    stars = [Star(W,H) for _ in range(50)]
    lvl = 1; level_rects = {}

    while True:
        dt = clock.tick(30)/1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return None
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN: return lvl
                if e.key == pygame.K_LEFT:  lvl = max(1, lvl-1)
                if e.key == pygame.K_RIGHT: lvl = min(4, lvl+1)
            if e.type == pygame.MOUSEBUTTONDOWN:
                if start_r.collidepoint(e.pos): return lvl
                for l,r in level_rects.items():
                    if r.collidepoint(e.pos): lvl = l

        for s in stars: s.update(dt)
        screen.fill(BG_DARK)
        for s in stars: s.draw(screen)

        t = ft.render(f"Focus Rocket — {proto_name}", True, TEXT_WHITE)
        screen.blit(t, (W//2-t.get_width()//2, H//5))

        bw, bh, gap = 110, 56, 16
        total = 4*bw+3*gap
        sx = W//2 - total//2
        py = H//3 + 20

        draw_panel(screen, (sx-20, py-16, total+40, bh+32))
        for l in range(1,5):
            cfg = DIFFICULTY[l]
            bx = sx + (l-1)*(bw+gap)
            r = pygame.Rect(bx, py, bw, bh)
            level_rects[l] = r
            if l == lvl:
                pygame.draw.rect(screen, ACCENT, r, border_radius=8)
                tc = BG_DARK
            else:
                pygame.draw.rect(screen, CARD_BG, r, border_radius=8)
                pygame.draw.rect(screen, TEXT_DIM, r, 1, border_radius=8)
                tc = TEXT_DIM
            lb = fs_font.render(cfg["label"], True, tc)
            screen.blit(lb, (bx+bw//2-lb.get_width()//2, py+bh//2-lb.get_height()//2))

        cfg = DIFFICULTY[lvl]
        dy = py + bh + 30
        for i, line in enumerate([
            f"Focus threshold: +{int(cfg['threshold_pct']*100)}% above baseline",
            f"Sustain time: {cfg['sustain_sec']:.1f}s",
            f"Distractors: {cfg['distractors']}"]):
            txt = fi.render(line, True, TEXT_DIM)
            screen.blit(txt, (W//2-txt.get_width()//2, dy+i*24))

        start_r = pygame.Rect(W//2-80, dy+90, 160, 46)
        pygame.draw.rect(screen, ACCENT, start_r, border_radius=10)
        st = fb.render("START", True, BG_DARK)
        screen.blit(st, (start_r.centerx-st.get_width()//2,
                         start_r.centery-st.get_height()//2))

        src = fi.render(f"EEG: {source_label}", True, TEXT_DIM)
        screen.blit(src, (W//2-src.get_width()//2, H-50))
        pygame.display.flip()


# ══════════════════════════════════════════════════════
#  EXIT CONFIRM
# ══════════════════════════════════════════════════════

def exit_confirm(screen, clock, score, level):
    fb = pygame.font.SysFont("arial",30,bold=True)
    fm = pygame.font.SysFont("arial",20)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return "quit"
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_y or e.key == pygame.K_RETURN: return "quit"
                if e.key == pygame.K_n or e.key == pygame.K_ESCAPE: return "resume"
            if e.type == pygame.MOUSEBUTTONDOWN:
                if yes_r.collidepoint(e.pos): return "quit"
                if no_r.collidepoint(e.pos): return "resume"

        draw_panel(screen, (W//2-200, H//2-80, 400, 160))
        t = fb.render("End session?", True, TEXT_WHITE)
        screen.blit(t, (W//2-t.get_width()//2, H//2-60))
        info = fm.render(f"Score: {score}  |  Level {level}", True, TEXT_DIM)
        screen.blit(info, (W//2-info.get_width()//2, H//2-15))

        yes_r = pygame.Rect(W//2-120, H//2+25, 100, 40)
        no_r  = pygame.Rect(W//2+20,  H//2+25, 100, 40)
        pygame.draw.rect(screen, EXIT_RED, yes_r, border_radius=8)
        pygame.draw.rect(screen, ACCENT, no_r, border_radius=8)
        yt = fm.render("Yes", True, TEXT_WHITE)
        nt = fm.render("No", True, BG_DARK)
        screen.blit(yt,(yes_r.centerx-yt.get_width()//2, yes_r.centery-yt.get_height()//2))
        screen.blit(nt,(no_r.centerx-nt.get_width()//2, no_r.centery-nt.get_height()//2))
        pygame.display.flip()
        clock.tick(30)


# ══════════════════════════════════════════════════════
#  DASHBOARD (post-session report)
# ══════════════════════════════════════════════════════

def render_dashboard(screen, clock, profile, proto_key, level, result):
    """Full-screen session dashboard with plots and summary table."""
    proto = PROTOCOLS[proto_key]
    metric_label = proto["short"]
    direction = proto["direction"]

    # unpack result
    score        = result["score"]
    reward_evts  = result["rewards"]
    dur          = result["duration"]
    mean_p       = result["mean_primary"]
    min_p        = result["min_primary"] if direction=="lower" else result["max_primary"]
    mean_ei      = result["mean_ei"]
    baseline     = result["baseline"]
    target       = result["target"]
    time_in_tgt  = result["time_in_target_pct"]
    focus_hist   = result["focus_history"]
    primary_hist = result["primary_history"]
    ei_hist      = result["ei_history"]
    timestamps   = result["timestamps"]

    cfg = DIFFICULTY[level]

    # ── matplotlib plots ──
    plt.rcParams.update({
        "figure.facecolor":"white","axes.facecolor":"white",
        "axes.edgecolor":"#333","axes.labelcolor":"#222",
        "xtick.color":"#333","ytick.color":"#333","text.color":"#222",
        "grid.color":"#e0e0e0","font.family":"serif","font.size":10,
    })

    # Plot 1: focus timeline
    fig1, ax1 = plt.subplots(figsize=(5.5, 2.2))
    t_arr = np.array(timestamps)
    f_arr = np.array(focus_hist)
    ax1.fill_between(t_arr, 0, f_arr, where=f_arr>=target, color="#27ae60", alpha=0.25)
    ax1.fill_between(t_arr, 0, f_arr, where=f_arr<target, color="#c0392b", alpha=0.2)
    ax1.plot(t_arr, f_arr, color="#2c3e50", linewidth=0.8)
    ax1.axhline(target, color="#7f8c8d", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Focus (0–1)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title(f"Focus Timeline — {time_in_tgt:.0f}% in target", fontweight="bold", fontsize=11, loc="left")
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.4)
    fig1.tight_layout()
    surf_plot1 = mpl_to_surface(fig1, dpi=110)

    # Plot 2: primary metric trend
    fig2, ax2 = plt.subplots(figsize=(5.5, 2.2))
    p_arr = np.array(primary_hist)
    ax2.plot(t_arr[:len(p_arr)], p_arr, color="#2c5f8a", linewidth=0.8)
    if len(p_arr) >= 3:
        z = np.polyfit(np.arange(len(p_arr)), p_arr, 1)
        ax2.plot(t_arr[:len(p_arr)], np.polyval(z, np.arange(len(p_arr))),
                 "--", color="#e67e22", linewidth=1.5)
    ax2.set_ylabel(f"{metric_label}")
    ax2.set_xlabel("Time (s)")
    arrow = "↓" if direction=="lower" else "↑"
    ax2.set_title(f"{metric_label} Trend ({arrow} = improving)", fontweight="bold", fontsize=11, loc="left")
    ax2.grid(True, alpha=0.4)
    fig2.tight_layout()
    surf_plot2 = mpl_to_surface(fig2, dpi=110)

    # ── dashboard loop ──
    saved = False
    fb = pygame.font.SysFont("arial",28,bold=True)
    fm = pygame.font.SysFont("arial",18)
    fs = pygame.font.SysFont("arial",15)
    fl = pygame.font.SysFont("arial",13)
    fxl = pygame.font.SysFont("arial",40,bold=True)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN or e.key == pygame.K_ESCAPE: return
            if e.type == pygame.MOUSEBUTTONDOWN:
                if save_r.collidepoint(e.pos) and not saved:
                    # save screenshot
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(DATA_DIR, f"report_{ts}.png")
                    pygame.image.save(screen, path)
                    saved = True
                    print(f"Report saved → {path}")
                if close_r.collidepoint(e.pos): return

        screen.fill(BG_DARK)
        margin = 30
        pw = surf_plot1.get_width()
        ph = surf_plot1.get_height()

        # ── top bar ──
        bar_h = 60
        pygame.draw.rect(screen, CARD_BG, (0,0,W,bar_h))
        info_parts = [
            f"Patient: {profile.get('name','—')}",
            f"Age: {profile.get('age','—')}",
            f"Protocol: {proto['name']}",
            f"Level: {cfg['label']}",
            f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
        ]
        ix = margin
        for part in info_parts:
            t = fs.render(part, True, TEXT_DIM)
            screen.blit(t, (ix, bar_h//2 - t.get_height()//2))
            ix += t.get_width() + 30

        # ── left: plots ──
        plot_x = margin
        plot_y = bar_h + 20
        screen.blit(surf_plot1, (plot_x, plot_y))
        screen.blit(surf_plot2, (plot_x, plot_y + ph + 10))

        # ── right: score card ──
        card_x = plot_x + pw + 30
        card_y = bar_h + 20
        card_w = W - card_x - margin
        card_h = 2*ph + 10

        draw_panel(screen, (card_x, card_y, card_w, card_h))

        # big score
        sc_txt = fxl.render(str(score), True, SCORE_POP)
        sc_lbl = fs.render("SCORE", True, TEXT_DIM)
        screen.blit(sc_lbl, (card_x + 20, card_y + 15))
        screen.blit(sc_txt, (card_x + 20, card_y + 35))

        # stats
        stats = [
            ("Reward Events", str(reward_evts)),
            ("Current Level", cfg["label"]),
            ("Time in Target", f"{time_in_tgt:.0f}%"),
            (f"Mean {metric_label}", f"{mean_p:.3f}"),
            ("Mean EI", f"{mean_ei:.3f}"),
        ]
        sy = card_y + 100
        for label, val in stats:
            lt = fs.render(label, True, TEXT_DIM)
            vt = fm.render(val, True, TEXT_WHITE)
            screen.blit(lt, (card_x+20, sy))
            screen.blit(vt, (card_x+card_w-vt.get_width()-20, sy))
            sy += 32

        # focus gauge
        gy = sy + 10
        fl_txt = fl.render("Focus", True, TEXT_DIM)
        screen.blit(fl_txt, (card_x+20, gy))
        gw = card_w - 40
        pygame.draw.rect(screen, BAR_BG, (card_x+20, gy+18, gw, 14), border_radius=4)
        fw = int(gw * time_in_tgt / 100)
        pygame.draw.rect(screen, GREEN, (card_x+20, gy+18, fw, 14), border_radius=4)

        # EI gauge
        gy += 40
        ei_txt = fl.render("Engagement", True, TEXT_DIM)
        screen.blit(ei_txt, (card_x+20, gy))
        ei_pct = min(1.0, mean_ei / 2.0)  # normalize EI to 0-1 range (typical EI < 2)
        eiw = int(gw * ei_pct)
        pygame.draw.rect(screen, BAR_BG, (card_x+20, gy+18, gw, 14), border_radius=4)
        pygame.draw.rect(screen, TEAL, (card_x+20, gy+18, eiw, 14), border_radius=4)

        # ── bottom: summary table ──
        tbl_y = plot_y + 2*ph + 30
        tbl_h = H - tbl_y - 70
        pygame.draw.rect(screen, CARD_BG, (margin, tbl_y, W-2*margin, tbl_h), border_radius=8)

        hdr_txt = fb.render("SESSION SUMMARY", True, TEXT_WHITE)
        screen.blit(hdr_txt, (margin+15, tbl_y+8))

        table_data = [
            ("Session Duration", f"{int(dur//60)}:{int(dur%60):02d}"),
            ("Final Score", str(score)),
            (f"Mean {metric_label}", f"{mean_p:.3f}"),
            (f"{'Min' if direction=='lower' else 'Max'} {metric_label}", f"{min_p:.3f}"),
            ("Baseline Focus", f"{baseline:.1%}"),
            ("Target Focus", f"{target:.1%}"),
            ("Mean EI", f"{mean_ei:.3f}"),
            ("Difficulty", cfg["label"]),
        ]
        cols = 4
        col_w = (W - 2*margin - 30) // cols
        row_h = 28
        tx = margin + 15
        ty = tbl_y + 42
        for i, (label, val) in enumerate(table_data):
            col = i % cols
            row = i // cols
            cx = tx + col * col_w
            cy = ty + row * (row_h * 2 + 8)
            lt = fl.render(label, True, TEXT_DIM)
            vt = fm.render(val, True, TEXT_WHITE)
            screen.blit(lt, (cx, cy))
            screen.blit(vt, (cx, cy + 18))

        # ── buttons ──
        btn_y = H - 55
        save_r = pygame.Rect(W//2-170, btn_y, 150, 40)
        close_r = pygame.Rect(W//2+20, btn_y, 150, 40)
        sc = GREEN if not saved else TEXT_DIM
        pygame.draw.rect(screen, sc, save_r, border_radius=8)
        pygame.draw.rect(screen, ACCENT, close_r, border_radius=8)
        st = fm.render("Save Report" if not saved else "Saved ✓", True, BG_DARK)
        ct = fm.render("Close", True, BG_DARK)
        screen.blit(st, (save_r.centerx-st.get_width()//2, save_r.centery-st.get_height()//2))
        screen.blit(ct, (close_r.centerx-ct.get_width()//2, close_r.centery-ct.get_height()//2))

        pygame.display.flip()
        clock.tick(30)


# ══════════════════════════════════════════════════════
#  GAME LOOP
# ══════════════════════════════════════════════════════

def game_loop(screen, clock, source, engine, level, proto_key):
    cfg = DIFFICULTY[level]
    proto = PROTOCOLS[proto_key]
    metric_label = proto["short"]
    font = pygame.font.SysFont("arial",20)
    font_score = pygame.font.SysFont("arial",24,bold=True)
    font_sm = pygame.font.SysFont("arial",14)
    font_cal = pygame.font.SysFont("arial",32,bold=True)
    font_exit = pygame.font.SysFont("arial",14,bold=True)

    stars = [Star(W,H) for _ in range(80)]
    particles, popups = [], []
    distractors = [Distractor(W,H) for _ in range(cfg["distractors"])]

    # ── calibration ──
    realtime = getattr(source, "is_realtime", False)
    cal_secs = REALTIME_CALIBRATION_SEC if realtime else CALIBRATION_SEC
    cal_focus, cal_primary = [], []
    cal_start = time.time()

    while time.time() - cal_start < cal_secs:
        dt = clock.tick(60)/1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                if realtime: source.stop()
                return None
        if isinstance(source, SimulatedEEG): source.set_key(False)
        res = source.update()
        if res:
            cal_focus.append(res[0])
            cal_primary.append(res[1])
        screen.fill(BG_DARK)
        for s in stars: s.update(dt); s.draw(screen)
        pct = min(1.0, (time.time()-cal_start)/cal_secs)
        by = H//2+30
        pygame.draw.rect(screen,BAR_BG,(60,by,W-120,12),border_radius=6)
        pygame.draw.rect(screen,ACCENT,(60,by,int((W-120)*pct),12),border_radius=6)
        t=font_cal.render("Calibrating...",True,TEXT_WHITE)
        screen.blit(t,(W//2-t.get_width()//2,H//2-20))
        msg = "Measuring resting baseline..." if realtime else "Relax, look at the screen"
        if realtime and not source.is_ready():
            msg = "Filling EEG buffer..."
        h=font_sm.render(msg,True,TEXT_DIM)
        screen.blit(h,(W//2-h.get_width()//2,by+24))
        pygame.display.flip()

    if realtime and len(cal_primary) >= 3:
        source.set_baseline(float(np.median(cal_primary)), float(np.std(cal_primary)))
    baseline = float(np.median(cal_focus)) if cal_focus else 0.4
    target = min(0.95, baseline + cfg["threshold_pct"])

    # ── main loop ──
    rocket_y = float(H - 100)
    score, focus, primary, ei, sustain = 0, baseline, 0.0, 0.0, 0.0
    rewards = 0
    primary_hist, ei_hist, focus_hist, ts_hist = [],[],[],[]
    game_start = time.time()
    t = 0.0
    running = True
    exit_btn = pygame.Rect(W-120, 10, 100, 34)

    while running:
        dt = clock.tick(FPS)/1000.0 if FPS > 0 else clock.tick()/1000.0
        t += dt
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if exit_confirm(screen,clock,score,level)=="quit": running=False
            if e.type == pygame.MOUSEBUTTONDOWN:
                if exit_btn.collidepoint(e.pos):
                    if exit_confirm(screen,clock,score,level)=="quit": running=False

        if isinstance(source, SimulatedEEG):
            source.set_key(pygame.key.get_pressed()[pygame.K_SPACE])

        res = source.update()
        if res:
            focus, primary, ei = res
            elapsed = time.time() - game_start
            primary_hist.append(primary)
            ei_hist.append(ei)
            focus_hist.append(focus)
            ts_hist.append(elapsed)

        # reward logic
        if focus >= target:
            sustain += dt
            rocket_y -= 70*dt
            if sustain >= cfg["sustain_sec"]:
                score += 10; rewards += 1; sustain = 0.0
                for _ in range(12):
                    particles.append(Particle(
                        W//2, rocket_y+20,
                        random.choice([SCORE_POP,FLAME_CORE,ACCENT]),
                        vel=(random.uniform(-60,60),random.uniform(-100,-30)),
                        life=0.8, size=random.randint(3,6)))
                popups.append(ScorePopup(W//2, rocket_y-30, "+10"))
        else:
            sustain = max(0.0, sustain-dt*0.4)
            rocket_y += 30*dt
        rocket_y = max(70, min(H-90, rocket_y))

        if random.random() < 0.4:
            particles.append(Particle(
                W//2+random.uniform(-6,6), rocket_y+20,
                random.choice([FLAME_OUTER,FLAME_MID]),
                vel=(random.uniform(-15,15),random.uniform(20,60)),
                life=0.3, size=random.randint(2,4)))

        for p in particles: p.update(dt)
        particles=[p for p in particles if p.life>0]
        for p in popups: p.update(dt)
        popups=[p for p in popups if p.life>0]

        # ── draw ──
        screen.fill(BG_DARK)
        climb=max(0,(H-90-rocket_y)/(H-160))
        for s in stars: s.update(dt,climb); s.draw(screen)
        for i in range(0,W,12):
            pygame.draw.line(screen,GOAL_LINE,(i,85),(i+6,85),1)
        screen.blit(font_sm.render("GOAL",True,GOAL_LINE),(8,67))
        for d in distractors: d.update(dt,t); d.draw(screen,t)
        draw_rocket(screen,W//2,int(rocket_y),focus,t)
        for p in particles: p.draw(screen)
        for p in popups: p.draw(screen,font_score)

        # HUD
        screen.blit(font_score.render(f"Score: {score}",True,TEXT_WHITE),(16,16))
        lv=font_sm.render(f"Level {level} — {cfg['label']}",True,TEXT_DIM)
        screen.blit(lv,(W-lv.get_width()-140,18))
        mt=font.render(f"{metric_label} {primary:.2f}",True,TEXT_DIM)
        screen.blit(mt,(16,46))
        eit=font_sm.render(f"EI {ei:.2f}",True,TEAL)
        screen.blit(eit,(16,70))

        draw_focus_bar(screen,16,H-36,W-32,18,focus,target)
        screen.blit(font_sm.render("Focus",True,TEXT_DIM),(16,H-54))
        if cfg["sustain_sec"]>0:
            sw=int((W-32)*min(1.0,sustain/cfg["sustain_sec"]))
            pygame.draw.rect(screen,ACCENT_GLOW,(16,H-62,sw,4),border_radius=2)
        et=font_sm.render(f"{int(time.time()-game_start)}s",True,TEXT_DIM)
        screen.blit(et,(W//2-et.get_width()//2,18))

        # EXIT button
        mx,my = pygame.mouse.get_pos()
        hov = exit_btn.collidepoint(mx,my)
        pygame.draw.rect(screen, EXIT_RED_H if hov else EXIT_RED, exit_btn, border_radius=6)
        ex=font_exit.render("EXIT",True,TEXT_WHITE)
        screen.blit(ex,(exit_btn.centerx-ex.get_width()//2,exit_btn.centery-ex.get_height()//2))

        pygame.display.flip()

    if realtime: source.stop()

    duration = time.time() - game_start
    mean_p = float(np.mean(primary_hist)) if primary_hist else 0.0
    mean_ei_val = float(np.mean(ei_hist)) if ei_hist else 0.0
    time_in_tgt = (np.mean(np.array(focus_hist)>=target)*100) if focus_hist else 0.0

    if primary_hist:
        min_p = float(np.min(primary_hist))
        max_p = float(np.max(primary_hist))
    else:
        min_p = max_p = 0.0

    # log to CSV
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE,"a",newline="") as f:
        w=csv.writer(f)
        if new:
            w.writerow(["timestamp","patient_name","patient_age","protocol",
                        "session_level","baseline_focus","target_focus",
                        "mean_primary","min_primary","max_primary",
                        "mean_ei","final_score","reward_events",
                        "time_in_target_pct","duration_sec"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                    "", "", proto_key, level,
                    round(baseline,3), round(target,3),
                    round(mean_p,3), round(min_p,3), round(max_p,3),
                    round(mean_ei_val,3), score, rewards,
                    round(time_in_tgt,1), round(duration,1)])

    return {
        "score": score, "rewards": rewards, "duration": duration,
        "mean_primary": mean_p, "min_primary": min_p, "max_primary": max_p,
        "mean_ei": mean_ei_val, "baseline": baseline, "target": target,
        "time_in_target_pct": time_in_tgt, "level": level,
        "focus_history": focus_hist, "primary_history": primary_hist,
        "ei_history": ei_hist, "timestamps": ts_hist,
    }


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    global W, H

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true")
    g.add_argument("--sim", action="store_true")
    ap.add_argument("--stream", default=EEG_STREAM_NAME)
    args = ap.parse_args()

    if args.live:    source_label = f"Live EEG — {args.stream}"
    elif args.sim:   source_label = "Simulated (SPACE key)"
    else:            source_label = "CSV Playback"

    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    screen = pygame.display.set_mode((W,H),
        pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Focus Rocket — ADHD Neurofeedback")
    clock = pygame.time.Clock()
    print(f"Display: {W}x{H} fullscreen")

    # patient profile
    profile = profile_screen(screen, clock)
    if profile is None: pygame.quit(); return

    # protocol select
    proto_key = protocol_screen(screen, clock)
    if proto_key is None: pygame.quit(); return

    proto = PROTOCOLS[proto_key]
    engine = ProtocolEngine(proto_key)

    # difficulty
    level = difficulty_screen(screen, clock, proto["name"], source_label)
    if level is None: pygame.quit(); return

    # create EEG source
    try:
        if args.live:
            source = RealtimeEEGSource(engine, stream_name=args.stream)
        elif args.sim:
            source = SimulatedEEG(engine)
        else:
            if not os.path.exists(BANDPOWER_CSV):
                print(f"ERROR: {BANDPOWER_CSV} not found. Use --sim or --live.")
                pygame.quit(); sys.exit(1)
            source = CSVBandPowerSource(engine)
    except Exception as ex:
        print(f"Could not start EEG source: {ex}")
        pygame.quit(); sys.exit(1)

    # run game
    result = game_loop(screen, clock, source, engine, level, proto_key)

    # dashboard
    if result:
        render_dashboard(screen, clock, profile, proto_key, level, result)

    pygame.quit()


if __name__ == "__main__":
    main()
