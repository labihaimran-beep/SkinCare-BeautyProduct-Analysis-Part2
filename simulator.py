"""Seeded simulator of digital-advertising records (SIMULATED DATA, not real campaigns).

Each row is one (date, campaign, platform, audience-segment) record with impressions, clicks, spend,
conversions and revenue in Pakistani rupees (Rs).

Built-in ground truth (so the recommendation engine can be validated):
  * each platform has base CPM / CTR / CVR / order value and a saturation exponent GAMMA;
  * on a given day a platform's "intensity" m scales impressions (so spend) by m, while CTR and CVR are
    scaled by m**(-GAMMA/2) each. Daily revenue therefore grows like spend**(1 - GAMMA):
    the TRUE elasticity of every platform is 1 - GAMMA (diminishing returns).
Because the generator is block-seeded, the first N rows of a larger dataset equal the smaller dataset.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PLATFORMS = ["Facebook", "Instagram", "Google", "TikTok", "YouTube"]
CAMPAIGNS = ["Eid_Sale", "Ramadan_Offers", "Summer_Collection", "Back_To_School", "Independence_Day", "Black_Friday",
             "Winter_Launch", "Brand_Awareness", "Retargeting", "New_Arrivals", "Loyalty_Rewards", "Clearance"]
CAMPAIGN_CVR = [1.15, 1.10, 0.95, 0.90, 1.00, 1.25, 0.95, 0.45, 1.70, 0.90, 1.30, 1.05]   # conversion multipliers
AGES = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS = ["Female", "Male", "Other"]
CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar", "Quetta"]
DEVICES = ["Mobile", "Desktop", "Tablet"]
COLUMNS = ["date", "campaign", "platform", "impressions", "clicks", "spend", "conversions", "revenue",
           "age", "gender", "city", "device"]

# platform: (CPM Rs, base CTR, base CVR (conv per click), mean order value Rs, GAMMA saturation)
PLATFORM_PARAMS = {
    "Facebook":  (350, 0.011, 0.022, 3200, 0.30),
    "Instagram": (420, 0.009, 0.018, 3500, 0.25),
    "Google":    (600, 0.030, 0.030, 3600, 0.45),
    "TikTok":    (250, 0.010, 0.014, 2600, 0.15),
    "YouTube":   (300, 0.007, 0.015, 3000, 0.20),
}
TRUE_ELASTICITY = {p: round(1 - v[4], 4) for p, v in PLATFORM_PARAMS.items()}

AGE_P = {"Facebook": [.10, .30, .28, .20, .12], "Instagram": [.28, .38, .20, .10, .04], "Google": [.12, .34, .28, .16, .10],
         "TikTok": [.45, .35, .12, .06, .02], "YouTube": [.22, .33, .24, .13, .08]}
AGE_MULT = {"Facebook": [0.8, 1.0, 1.1, 1.1, 0.9], "Instagram": [1.0, 1.1, 1.0, 0.8, 0.6], "Google": [0.8, 1.1, 1.2, 1.1, 0.9],
            "TikTok": [1.2, 1.0, 0.7, 0.5, 0.4], "YouTube": [1.0, 1.0, 1.0, 0.9, 0.8]}
GENDER_P = [0.48, 0.50, 0.02]
GENDER_MULT = {"Facebook": [1.05, 0.95, 1.0], "Instagram": [1.12, 0.90, 1.0], "Google": [0.98, 1.02, 1.0],
               "TikTok": [1.05, 0.97, 1.0], "YouTube": [0.92, 1.06, 1.0]}
CITY_P = [.28, .22, .12, .08, .09, .07, .08, .06]
CITY_MULT = [1.10, 1.08, 1.15, 0.97, 0.92, 0.90, 0.88, 0.85]
DEVICE_P = {"Facebook": [.82, .14, .04], "Instagram": [.90, .07, .03], "Google": [.62, .34, .04],
            "TikTok": [.97, .01, .02], "YouTube": [.60, .22, .18]}
DEVICE_MULT = [1.0, 1.25, 0.9]            # conversion multiplier by device
MONTH_MULT = [0.95, 0.95, 1.10, 1.20, 1.00, 0.95, 1.00, 1.05, 1.00, 1.00, 1.20, 1.10]
BLOCK = 100_000
INTENSITY_SIGMA = 0.45
START = pd.Timestamp("2024-01-01")
N_DAYS = 366


def _intensity(seed: int) -> np.ndarray:
    """Daily platform intensity m[d, p], mean about 1 (independent of dataset size)."""
    rng = np.random.default_rng([seed, 999])
    return np.exp(rng.normal(-INTENSITY_SIGMA ** 2 / 2, INTENSITY_SIGMA, size=(N_DAYS, len(PLATFORMS))))


def _block(seed: int, block: int, m: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng([seed, block])
    n = BLOCK
    pid = rng.integers(0, len(PLATFORMS), n)
    day = rng.integers(0, N_DAYS, n)
    cid = rng.integers(0, len(CAMPAIGNS), n)
    gid = rng.choice(len(GENDERS), n, p=GENDER_P)
    city = rng.choice(len(CITIES), n, p=CITY_P)
    age, dev = np.zeros(n, int), np.zeros(n, int)
    for i, p in enumerate(PLATFORMS):
        k = pid == i
        age[k] = rng.choice(len(AGES), k.sum(), p=AGE_P[p])
        dev[k] = rng.choice(len(DEVICES), k.sum(), p=DEVICE_P[p])
    cpm = np.array([PLATFORM_PARAMS[p][0] for p in PLATFORMS])[pid]
    ctr0 = np.array([PLATFORM_PARAMS[p][1] for p in PLATFORMS])[pid]
    cvr0 = np.array([PLATFORM_PARAMS[p][2] for p in PLATFORMS])[pid]
    aov = np.array([PLATFORM_PARAMS[p][3] for p in PLATFORMS])[pid]
    gam = np.array([PLATFORM_PARAMS[p][4] for p in PLATFORMS])[pid]
    age_m = np.array([AGE_MULT[p] for p in PLATFORMS])[pid, age]
    gen_m = np.array([GENDER_MULT[p] for p in PLATFORMS])[pid, gid]
    mday = m[day, pid]
    month = (START + pd.to_timedelta(day, unit="D")).month.values - 1
    imp = np.maximum(1, (rng.lognormal(np.log(6000), 0.6, n) * mday).astype(np.int64))
    sat = mday ** (-gam / 2)
    ctr = np.clip(ctr0 * age_m * gen_m * sat * np.where(dev == 0, 1.0, 0.9), 1e-4, 0.5)
    clicks = rng.binomial(imp, ctr)
    cvr = np.clip(cvr0 * np.array(CAMPAIGN_CVR)[cid] * np.array(DEVICE_MULT)[dev] * np.array(CITY_MULT)[city]
                  * np.array(MONTH_MULT)[month] * age_m * sat, 1e-4, 0.9)
    conv = rng.binomial(clicks, cvr)
    spend = imp * cpm / 1000.0 * rng.lognormal(-0.005, 0.10, n)
    revenue = conv * aov * np.array(CITY_MULT)[city] * np.exp(rng.normal(-0.03, 0.25 / np.sqrt(np.maximum(conv, 1))))
    return pd.DataFrame({
        "date": (START + pd.to_timedelta(day, unit="D")).strftime("%Y-%m-%d"),
        "campaign": np.array(CAMPAIGNS)[cid], "platform": np.array(PLATFORMS)[pid],
        "impressions": imp, "clicks": clicks, "spend": spend.round(2), "conversions": conv, "revenue": revenue.round(2),
        "age": np.array(AGES)[age], "gender": np.array(GENDERS)[gid], "city": np.array(CITIES)[city],
        "device": np.array(DEVICES)[dev]})[COLUMNS]


def generate_chunks(n_rows: int, seed: int = 42, chunk_rows: int = BLOCK):
    """Yield DataFrames totalling n_rows (prefix-stable across n_rows). chunk_rows must be a multiple of BLOCK."""
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    m = _intensity(seed)
    done, b = 0, 0
    per = max(chunk_rows // BLOCK, 1)
    while done < n_rows:
        parts = []
        for _ in range(per):
            if done >= n_rows:
                break
            blk = _block(seed, b, m)
            take = min(BLOCK, n_rows - done)
            parts.append(blk.iloc[:take]); done += take; b += 1
        yield pd.concat(parts, ignore_index=True)


def generate(n_rows: int, seed: int = 42) -> pd.DataFrame:
    return pd.concat(list(generate_chunks(n_rows, seed)), ignore_index=True)


def write_csv(path, n_rows: int, seed: int = 42, chunk_rows: int = 1_000_000) -> int:
    """Stream n_rows to a CSV (no quoted fields, so simple line parsers work). Returns rows written.
    Written to a .tmp file and renamed at the end, so an interrupted run never leaves a truncated dataset."""
    import os
    first = True
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        for ch in generate_chunks(n_rows, seed, chunk_rows):
            ch.to_csv(f, index=False, header=first, lineterminator="\n")
            first = False
    os.replace(tmp, path)
    return n_rows


if __name__ == "__main__":       # python -m src.data_generation.simulator OUT.csv N [SEED]
    import sys
    write_csv(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 42)
