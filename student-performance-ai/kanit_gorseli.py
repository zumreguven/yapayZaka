"""Model adı ve test metriklerini özetleyen PNG üretir."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from yapilandirma import KANIT_GORSEL_DOSYASI, KANIT_KLASORU


def _tr_yuzde_str(r2_skoru: float) -> str:
    """R² ≥ 0 için Türkçe ondalık virgüllü yüzde kısa metin."""
    if r2_skoru >= 0:
        y = r2_skoru * 100
        return f"{y:.1f}".replace(".", ",")
    return "—"


def model_kanit_ekrani_png_kaydet(
    model_etiketi: str,
    r2_skoru: float,
    mae: float,
    rmse: float,
    egitim_sayisi: int,
    test_sayisi: int,
    hedef_yol: Path | None = None,
) -> Path:
    """
    Panoya/rapora konabilir, ekran görüntüsü görünümlü bir PNG kaydeder.
    Dönüş: kaydedilen dosya yolu.
    """
    KANIT_KLASORU.mkdir(parents=True, exist_ok=True)
    cikti = hedef_yol or KANIT_GORSEL_DOSYASI

    yuzde_kisa = _tr_yuzde_str(r2_skoru)
    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=140)
    fig.patch.set_facecolor("#e2e8f0")
    ax.set_facecolor("#e2e8f0")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Dış çerçeve (pencere benzeri)
    kutu = FancyBboxPatch(
        (0.04, 0.06),
        0.92,
        0.88,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#ffffff",
        edgecolor="#334155",
        linewidth=2,
    )
    ax.add_patch(kutu)

    ax.text(
        0.5,
        0.88,
        "Model — test performans özeti",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.805,
        f"Oluşturulma: {zaman}",
        ha="center",
        va="center",
        fontsize=10,
        color="#64748b",
    )

    ax.text(
        0.5,
        0.68,
        "Kullanılan model",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )
    ax.text(
        0.5,
        0.60,
        model_etiketi,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0c4a6e",
        wrap=True,
    )

    ax.text(
        0.5,
        0.46,
        "Test verisindeki başarı oranı (R² → açıklanan varyans)",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )

    if r2_skoru >= 0:
        ax.text(
            0.5,
            0.34,
            f"%{yuzde_kisa}",
            ha="center",
            va="center",
            fontsize=42,
            fontweight="bold",
            color="#059669",
        )
    else:
        ax.text(
            0.5,
            0.34,
            "R² < 0",
            ha="center",
            va="center",
            fontsize=36,
            fontweight="bold",
            color="#dc2626",
        )

    ax.text(
        0.5,
        0.20,
        f"R² = {r2_skoru:.4f}  |  MAE = {mae:.3f}  |  RMSE = {rmse:.3f}",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )
    ax.text(
        0.5,
        0.12,
        f"Eğitim: {egitim_sayisi} örnek  ·  Test: {test_sayisi} örnek  ·  Değerlendirme yalnızca test kümesinde",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b",
    )
    ax.text(
        0.5,
        0.04,
        "Otomatik üretilmiş özet görseli.",
        ha="center",
        va="center",
        fontsize=8,
        color="#94a3b8",
        style="italic",
    )

    plt.tight_layout()
    fig.savefig(cikti, facecolor=fig.patch.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return cikti
