# MooMotion Setup Guide

## What you need
- MacBook
- XIAO-LEFT sensor (ankle strap, left leg) — charged via USB-C
- XIAO-RIGHT sensor (ankle strap, right leg) — charged via USB-C

---

## One-time setup (do this once)

Open Terminal and run these commands one at a time:

**1. Install Python dependencies**
```bash
pip install bleak numpy matplotlib
```

**2. Download the code**
```bash
git clone https://github.com/coreyczhang/infant-kick-detector.git
```

---

## Every time you use it

**1. Power on both sensors**
- Hold each XIAO near a USB-C charger briefly to wake it, or just confirm the LED is on
- Both should have charged batteries — no USB cable needed during use

**2. Open Terminal and go to the project folder**
```bash
cd infant-kick-detector
```

**3. Run the kick detector**

Without live plot (terminal only):
```bash
python src/kick_detector.py --ble --threshold 2.0 --no-plot
```

With live plot (scrolling accelerometer graph with kick markers):
```bash
python src/kick_detector.py --ble --threshold 2.0
```
The plot shows both legs in real time — green lines are selective kicks, red lines are non-selective.

**4. Wait ~20 seconds** — you should see:
```
Found XIAO-LEFT: ...
Found XIAO-RIGHT: ...
Connected to XIAO-LEFT ✅
Connected to XIAO-RIGHT ✅
```

**5. Session data is saved automatically** to `kick_logs/session_YYYYMMDD_HHMMSS.csv`

**6. Stop the session** — press `Ctrl+C` and a summary prints automatically

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Sensor not found | Press reset button on XIAO once, wait 5 seconds, try again |
| `bleak not found` | Run `pip install bleak` |
| Only one sensor found | Make sure both batteries are charged |
| Kicks not registering | Lower the threshold: `--threshold 1.5` |

---

## Test with only one sensor

If only XIAO-RIGHT is available:
```bash
python src/kick_detector.py --ble --sim-left --threshold 2.0 --no-plot
```

If only XIAO-LEFT is available:
```bash
python src/kick_detector.py --ble --sim-right --threshold 2.0 --no-plot
```
