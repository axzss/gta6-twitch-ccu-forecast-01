[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/axzss/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/axzss/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers, hour by hour

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), as five quantiles an hour, from
everything recorded up to the end of the day's first hour.

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
`main.py` (Python 3.13, standard library only) reads a case at `/input` and
writes `/output/<id>.out` with 23 rows, hours 2 to 24, each five quantiles
(5%, 25%, 50%, 75%, 95%) of that hour's concurrent viewers.

## Method

For every comparable past launch in the case file, take its release-day
hourly Twitch viewers and form the log-ratios `log(viewers(h)/viewers(1))`.
The shape of a launch day is driven by *when* the world unlocks relative to
00:00 UTC, and hour 1's print already measures that: a small hour 1 means
most of the world had not unlocked yet, so the day rises steeply; a big
hour 1 means most of the audience was already in, so the day stays flat.
Across the kit's 37 comparable launches the correlation between log(hour-1
viewers) and log(peak / hour-1) is -0.94.

So, per hour, the entry fits a locally-weighted linear regression of the
log-ratio on log(hour-1 viewers) — Gaussian kernel, bandwidth 1.7 in log
units, evaluated at the case's hour-1 print — and takes the spread from the
weighted median absolute deviation of the residuals (x1.4826, floored at
0.05, times 0.9). Quantiles are `p1 * exp(pred + z_tau * sigma)`.

## Local validation

Exact pinball loss on the log scale, scored against SullyGnome's hourly
averages for the four practice launches (the scorer reproduces the rules'
worked example exactly, 0.112545):

| case | hour-1 | house baseline | this entry |
|------|-------:|---------------:|-----------:|
| Marvel's Wolverine | 23,357 | 0.7436 | 0.1801 |
| Red Dead Redemption 2 | 116,550 | 0.2868 | 0.2116 |
| God of War Ragnarök | 269,517 | 0.2146 | 0.1453 |
| Marvel's Spider-Man 2 | 119,087 | 0.3033 | 0.1992 |
| **mean** | | **0.3871** | **0.1840** |

Leave-one-out over the 37 comparable launches: baseline 0.9296, this entry
0.3899. Every practice case beats the baseline. 90% coverage 0.92 on the
practice (honest target 0.90). Runtime ~0.1 s per case, deterministic.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
