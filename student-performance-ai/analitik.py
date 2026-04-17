"""
Sınıf ortalaması (test G3), sanal sıra, yüzdelik karşılaştırma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_sinif_ortalamasi_g3(y_test: pd.Series) -> float:
    """Yalnızca test kümesindeki gerçek G3 ortalaması."""
    return float(y_test.mean())


def ortalamayla_karsilastir(tahmini_g3: float, sinif_ort: float) -> str:
    if tahmini_g3 > sinif_ort + 0.01:
        return "Sınıf ortalamasının üstünde"
    if tahmini_g3 < sinif_ort - 0.01:
        return "Sınıf ortalamasının altında"
    return "Sınıf ortalamasına yakın"


def sanal_siralama(tahmin: float, y_test: pd.Series) -> tuple[int, int]:
    """Tahmini test kümesine ekleyip sırala; 1 = en yüksek G3."""
    taban = y_test.values.astype(float)
    birlesik = np.concatenate([taban, np.array([float(tahmin)])])
    sira_indeksleri = np.argsort(-birlesik, kind="stable")
    konum = int(np.where(sira_indeksleri == len(taban))[0][0])
    return konum + 1, len(birlesik)


def testte_yuzde_kac_ogrenciden_yuksek(tahmin: float, y_test: pd.Series) -> float:
    """Testteki öğrencilerin kaç yüzdesinden tahmin daha yüksek."""
    return float(100.0 * np.mean(y_test.values.astype(float) < float(tahmin)))


def dersten_gecme_olasiligi_sinifi(tahmin: float) -> tuple[str, str]:
    """
    Tahmine göre kullanıcı dilinde özet: Riskli / Orta / Başarılı.
    İkinci değer: stil sınıfı anahtarı (dusuk / orta / yuksek).
    """
    t = float(np.clip(tahmin, 0, 20))
    if t <= 9:
        return "Riskli", "dusuk"
    if t <= 14:
        return "Orta", "orta"
    return "Başarılı", "yuksek"
