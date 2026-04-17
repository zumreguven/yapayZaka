"""
Örnek büyük CSV üretir: studytime, failures, absences, G1, G2, G3 sütunları.
Kullanım:
  python araclar/sentetik_csv_uret.py --satir 50000
Çıktı: data/veri_ozel.csv (mevcut veri_ozel.csv üzerine yazar)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "data" / "veri_ozel.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--satir", type=int, default=30_000, help="Üretilecek satır sayısı")
    ap.add_argument("--tohum", type=int, default=42)
    args = ap.parse_args()

    n = max(100, args.satir)
    rng = np.random.default_rng(args.tohum)

    study = rng.integers(1, 5, size=n)
    fail = rng.integers(0, 6, size=n)
    absn = rng.integers(0, 31, size=n)
    g1 = rng.integers(0, 21, size=n)
    g2 = rng.integers(0, 21, size=n)

    # Basit yapay ilişki + gürültü → G3
    hami = (
        0.9 * study
        - 0.75 * fail
        - 0.11 * absn
        + 0.32 * g1
        + 0.42 * g2
        + rng.normal(0, 1.8, size=n)
    )
    g3 = np.clip(np.round(hami), 0, 20).astype(int)

    df = pd.DataFrame(
        {
            "studytime": study,
            "failures": fail,
            "absences": absn,
            "G1": g1,
            "G2": g2,
            "G3": g3,
        }
    )
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CIKTI, index=False)
    print(f"Yazıldı: {CIKTI} ({len(df)} satır)")


if __name__ == "__main__":
    main()
