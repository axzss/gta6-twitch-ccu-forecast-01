"""gta6-twitch-ccu-forecast-01 entry: launch-day Twitch viewers, hour by hour.

Method
------
For every comparable past launch in the case file, take its release-day hourly
Twitch category viewers and form the log-ratios

    r_h = log(viewers(h) / viewers(1))          for h = 2..24.

The shape of a launch day is driven by *when* the world unlocks relative to
00:00 UTC, and hour 1's print already measures that: a small hour 1 means
most of the world had not unlocked yet (the US still to come), so the day
rises steeply; a big hour 1 means most of the audience was already in, so
the day stays flat. Across the kit's 37 comparable launches the correlation
between log(hour-1 viewers) and log(peak / hour-1) is -0.94, and the slope of
log-ratio on log(hour 1) grows from ~0 at hour 2 to about -0.93 at hour 20.

So, per hour, fit a locally-weighted linear regression of the log-ratio on
log(hour-1 viewers) - a Gaussian kernel of bandwidth 1.7 in log units,
evaluated at this case's hour-1 print - and take the spread from the
weighted median absolute deviation of the residuals (x1.4826, floored at
0.05), times 0.9. Quantiles are

    q_tau(h) = p1 * exp(pred_h + z_tau * sigma_h),

with p1 the case's hour-1 print and z_tau the standard normal quantiles.
Hours with fewer than 5 comparable points fall back to the house baseline's
persistence shape (median = hour 1, sigma = 0.15 * sqrt(h - 1)).

Local validation (exact pinball loss on the log scale, scored against
SullyGnome's hourly averages fetched for the four practice launches;
the scorer reproduces the rules' worked example exactly, 0.112545):

    case   hour-1   baseline   this entry
    p00     23,357     0.7436      0.1801
    p01    116,550     0.2868      0.2116
    p02    269,517     0.2146      0.1453
    p03    119,087     0.3033      0.1992
    mean               0.3871      0.1840

Leave-one-out over the 37 comparable launches: baseline 0.9296, this entry
0.3899. Every practice case beats the baseline, and the mean is 52% below
it. No second feature helped: residuals show no correlation with hour-1
channels, channels/viewers, or the pre-release viewer level (|r| < 0.13),
so the entry uses hour 1 and the comparables only.

What is still unproven: the final is scored on GTA VI's real release day,
which no one has seen. The practice score is a guide, not a guarantee -
the rules say so themselves - and a launch as big as GTA VI sits at the
top end of the comparable range, where the curve is flattest and the
regression is an extrapolation in the direction the data supports.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Gaussian kernel bandwidth (log-viewer units) for the local regression.
BANDWIDTH = 1.7
# Spread multiplier on the weighted-MAD sigma.
SPREAD = 0.9
# Minimum comparable points per hour before falling back to persistence.
MIN_POINTS = 5
# Floor on the log-spread, and the baseline's per-root-hour sigma.
SIGMA_FLOOR = 0.05
BASE_SIGMA = 0.15
MAX_VIEWERS = 1_000_000_000


def parse_time(text):
    """ISO 8601 UTC timestamp -> datetime (handles the trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as the starter rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def day_curve(comparable):
    """Release-day hourly viewers {hour 1..24: viewers} for one comparable."""
    day_start = parse_time(comparable["day_start"])
    curve = {}
    for point in comparable.get("hourly", []):
        try:
            when = parse_time(point["from"])
        except (KeyError, ValueError):
            continue
        hours = (when - day_start).total_seconds() / 3600.0
        if 0 <= hours < 24 and abs(hours - round(hours)) < 1e-6:
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                curve[int(hours) + 1] = float(viewers)
    return curve


def training_curves(case):
    """Usable comparable launch-day curves, sorted by name for determinism.

    The target game's own pre-release history (in the final case, the
    Grand Theft Auto VI category's hours from its premiere onward) is not a
    launch day and is left out. Every other comparable in the case file is
    fair game: the house bundled them into the case deliberately, whatever
    year they launched in.
    """
    target_name = (case.get("game") or {}).get("name", "")
    curves = []
    for comparable in case.get("comparables", []):
        name = comparable.get("name", "")
        if target_name and target_name in name:
            continue
        curve = day_curve(comparable)
        if 1 in curve and curve[1] > 0:
            curves.append((name, curve))
    curves.sort(key=lambda item: item[0])
    return curves


def local_fit(points, x0):
    """Locally-weighted regression of log-ratio on log hour-1 viewers.

    points: [(log p1, log ratio)] for one hour. Returns (prediction at x0,
    weighted-MAD sigma). The prediction is the fitted value at the target's
    own log hour-1 print, clamped to one log unit beyond the training
    range: far outside it the kernel fit is an extrapolation, and the
    nearest launches are the honest answer.
    """
    xs = [x for x, _ in points]
    x0 = min(max(x0, min(xs) - 1.0), max(xs) + 1.0)
    weights = [math.exp(-0.5 * ((x - x0) / BANDWIDTH) ** 2) for x, _ in points]
    total = sum(weights)
    if total <= 0:
        return None, None
    mean_x = sum(w * x for w, (x, _) in zip(weights, points)) / total
    mean_y = sum(w * y for w, (_, y) in zip(weights, points)) / total
    sxx = sum(w * (x - mean_x) ** 2 for w, (x, _) in zip(weights, points))
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, (x, y) in zip(weights, points))
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    prediction = mean_y + slope * (x0 - mean_x)
    residuals = [y - (mean_y + slope * (x - mean_x)) for x, y in points]
    median_resid = sorted(residuals)[len(residuals) // 2]
    devs = sorted((abs(r - median_resid), w)
                  for r, w in zip(residuals, weights))
    acc = 0.0
    mad = 0.3
    for dev, w in devs:
        acc += w
        if acc >= total / 2:
            mad = dev
            break
    sigma = 1.4826 * mad if mad > 0 else 0.3
    return prediction, max(sigma, SIGMA_FLOOR)


def hour_quantiles(p1, hour, curves):
    """Five quantiles for one hour, from the comparables' log-ratios."""
    x0 = math.log(p1)
    points = []
    for _, curve in curves:
        if hour in curve and curve[hour] > 0:
            points.append((math.log(curve[1]),
                           math.log(curve[hour] / curve[1])))
    if len(points) >= MIN_POINTS:
        prediction, sigma = local_fit(points, x0)
        spread = sigma * SPREAD
    else:
        prediction = 0.0
        spread = BASE_SIGMA * math.sqrt(hour - 1)
    quantiles = []
    previous = 0.0
    for z in Z:
        value = p1 * math.exp(prediction + z * spread)
        value = min(max(value, 1.0), float(MAX_VIEWERS))
        value = round2(value)
        # Quantiles must not decrease; rounding can only tie them.
        if value < previous:
            value = previous
        previous = value
        quantiles.append(value)
    return quantiles


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    curves = training_curves(case)
    return {"hours": [{"hour": row["hour"],
                       "q": hour_quantiles(p1, row["hour"], curves)}
                      for row in case["forecast"]]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
