"""
Ham veri yükleme ve ön işleme.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from yapilandirma import HEDEF_SUTUN, OZELLIK_SUTUNLARI, VERI_YOLU, VERI_YOLU_500K, VERI_YOLU_OZEL


def veri_dosya_yolu_sec() -> Path:
    """
    Öncelik sırası:
    1) student_performance_500k.csv (büyük veri)
    2) veri_ozel.csv
    3) student-mat.csv (UCI)
    """
    if VERI_YOLU_500K.exists():
        return VERI_YOLU_500K
    if VERI_YOLU_OZEL.exists():
        return VERI_YOLU_OZEL
    return VERI_YOLU


def ham_veriyi_yukle(yol: Path | str | None = None) -> pd.DataFrame:
    """CSV dosyasını oku. yol verilmezse yukarıdaki öncelik sırasına göre dosya seçilir."""
    dosya = Path(yol) if yol else veri_dosya_yolu_sec()
    if not dosya.exists():
        raise FileNotFoundError(
            f"Veri bulunamadı: {dosya}. data/student_performance_500k.csv, data/veri_ozel.csv veya data/student-mat.csv gerekir."
        )
    return pd.read_csv(dosya, sep=",")


def ogrenci_verisini_temizle(cerceve: pd.DataFrame) -> pd.DataFrame:
    """
    Sadece gerekli sütunları al, sayıya çevir, eksik satırları at.
    G3 dışındaki sütunlar özellik olarak kullanılmaz (sızıntı yok).
    """
    gerekli = OZELLIK_SUTUNLARI + [HEDEF_SUTUN]
    eksik = [s for s in gerekli if s not in cerceve.columns]
    if eksik:
        raise ValueError(
            f"Veri setinde şu sütunlar yok: {eksik}. CSV’de tam olarak şunlar olmalı: {gerekli}"
        )

    temiz = cerceve[gerekli].copy()
    for sutun in gerekli:
        temiz[sutun] = pd.to_numeric(temiz[sutun], errors="coerce")

    return temiz.dropna(subset=gerekli)


def ozellik_ve_hedefi_ayir(cerceve: pd.DataFrame):
    """X (özellikler) ve y (G3) döndür."""
    x = cerceve[OZELLIK_SUTUNLARI]
    y = cerceve[HEDEF_SUTUN]
    return x, y
