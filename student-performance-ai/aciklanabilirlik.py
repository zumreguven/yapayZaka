"""
Tahmin için sade Türkçe açıklama (model türüne göre).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yapilandirma import OZELLIK_SUTUNLARI, RISK_ORTA_UST, RISK_ZAYIF_UST


def _risk_etiketi(tahmin: float) -> tuple[str, str]:
    p = float(np.clip(tahmin, 0, 20))
    if p <= RISK_ZAYIF_UST:
        return "Zayıf", "dusuk"
    if p <= RISK_ORTA_UST:
        return "Orta", "orta"
    return "Başarılı", "yuksek"


def _dogrusal_yerel_etkiler(model, satir: pd.DataFrame) -> list[tuple[str, float]]:
    olcekle = model.named_steps["olcekle"]
    reg = model.named_steps["regresyon"]
    xs = olcekle.transform(satir[OZELLIK_SUTUNLARI])
    katsayilar = reg.coef_.ravel()
    katkilar = (xs.ravel() * katsayilar).tolist()
    ciftler = list(zip(OZELLIK_SUTUNLARI, katkilar))
    ciftler.sort(key=lambda x: abs(x[1]), reverse=True)
    return ciftler


def _agac_yerel_etkiler(model, satir: pd.DataFrame, egitim_ort: pd.Series) -> list[tuple[str, float]]:
    if not hasattr(model, "feature_importances_"):
        return [(c, 0.0) for c in OZELLIK_SUTUNLARI]
    onem = model.feature_importances_.ravel()
    degerler = satir[OZELLIK_SUTUNLARI].values.ravel()
    ortalamalar = egitim_ort[OZELLIK_SUTUNLARI].values
    imza = (degerler - ortalamalar) * onem
    ciftler = list(zip(OZELLIK_SUTUNLARI, imza.tolist()))
    ciftler.sort(key=lambda x: abs(x[1]), reverse=True)
    return ciftler


def aciklama_uret(
    model_anahtari: str,
    tahminleyici,
    girdiler: dict,
    tahmini_g3: float,
    egitim_ort: pd.Series,
    egitim_std: pd.Series,
    ozellik_adlari_tr: dict[str, str],
) -> dict:
    satir = pd.DataFrame([girdiler])
    risk_metin, risk_kademe = _risk_etiketi(tahmini_g3)

    if model_anahtari == "dogrusal_regresyon":
        etkiler = _dogrusal_yerel_etkiler(tahminleyici, satir)
    else:
        etkiler = _agac_yerel_etkiler(tahminleyici, satir, egitim_ort)

    maddeler = []
    for ozellik, etki in etkiler[:5]:
        o_ort = float(egitim_ort[ozellik])
        o_std = float(egitim_std[ozellik])
        deger = float(girdiler[ozellik])
        yon = (
            "eğitim ortalamasının üzerinde"
            if deger > o_ort
            else "eğitim ortalamasının altında"
            if deger < o_ort
            else "ortalamaya yakın"
        )
        buyukluk = abs(deger - o_ort) / o_std if o_std > 0 else 0.0
        siddet = "Belirgin şekilde " if buyukluk > 1.0 else ""
        ad = ozellik_adlari_tr.get(ozellik, ozellik)

        if model_anahtari == "dogrusal_regresyon":
            if etki > 0:
                yorum = "bu doğrusal modelde tahmini notu yukarı çeken yönde etkili."
            elif etki < 0:
                yorum = "bu doğrusal modelde tahmini notu aşağı çeken yönde etkili."
            else:
                yorum = "bu noktada etkisi nötre yakın."
        else:
            if etki > 0:
                yorum = "ağaç tabanlı modelde daha yüksek G3 ile ilişkilendirilen bir profil sinyali veriyor."
            elif etki < 0:
                yorum = "ağaç tabanlı modelde risk artışı yönünde okunan bir sinyal içeriyor."
            else:
                yorum = "bu profilde nötre yakın okunuyor."

        maddeler.append(f"**{ad}** {siddet}{yon}; {yorum}")

    baslik = (
        f"Model, final not (G3) için yaklaşık **{tahmini_g3:.1f}/20** bekliyor; "
        f"risk düzeyi **{risk_metin}** bandında."
    )

    return {
        "baslik": baslik,
        "risk_etiketi": risk_metin,
        "risk_kademe": risk_kademe,
        "ozellik_etkileri": etkiler,
        "maddeler": maddeler,
    }
