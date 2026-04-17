"""
Kullanıcı girdileri ve sınıf karşılaştırması için Türkçe grafikler.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def eksen_stili(eksen):
    eksen.grid(True, alpha=0.25, axis="y")
    eksen.spines["top"].set_visible(False)
    eksen.spines["right"].set_visible(False)


def ciz_kullanici_girdi_panelleri(girdiler: dict):
    """
    Ölçekler farklı olduğu için üç davranış göstergesi + notlar ayrı panellerde.
    """
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    cal = float(girdiler["studytime"])
    fail = float(girdiler["failures"])
    absn = float(girdiler["absences"])
    g1 = float(girdiler["G1"])
    g2 = float(girdiler["G2"])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar([0], [cal], color="#2563eb", width=0.45)
    ax0.set_xticks([0])
    ax0.set_xticklabels(["Haftalık çalışma\n(1–4 ölçeği)"], fontsize=9)
    ax0.set_ylim(0, 4.5)
    ax0.set_ylabel("Düzey")
    ax0.text(0, cal + 0.15, f"{cal:.0f}", ha="center", fontweight="bold")
    eksen_stili(ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar([0], [fail], color="#7c3aed", width=0.45)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Geçmiş tekrar /\nbaşarısızlık"], fontsize=9)
    ax1.set_ylim(0, 5.5)
    ax1.text(0, fail + 0.12, f"{fail:.0f}", ha="center", fontweight="bold")
    eksen_stili(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar([0], [absn], color="#db2777", width=0.45)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Devamsızlık\n(gün)"], fontsize=9)
    ax2.set_ylim(0, max(32, absn * 1.15))
    ax2.text(0, absn + max(1, absn * 0.03), f"{absn:.0f}", ha="center", fontweight="bold")
    eksen_stili(ax2)

    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(2)
    ax3.bar(x, [g1, g2], color=["#059669", "#d97706"], width=0.45, edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["1. dönem (G1)", "2. dönem (G2)"])
    ax3.set_ylim(0, 22)
    ax3.set_ylabel("Not (0–20)")
    eksen_stili(ax3)
    for i, v in enumerate([g1, g2]):
        ax3.text(i, v + 0.5, f"{v:.0f}", ha="center", fontweight="bold")

    fig.suptitle("Girdiğiniz değerler — görsel özet", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def ciz_tahmin_sinif_karsilastirma(tahmini_g3: float, sinif_ortalama_g3: float):
    """Tahmin ile referans sınıf ortalaması yan yana."""
    fig, eksen = plt.subplots(figsize=(6, 4))
    isimler = ["Tahmininiz\n(G3)", "Sınıf ortalaması\n(referans)"]
    degerler = [tahmini_g3, sinif_ortalama_g3]
    renkler = ["#0284c7", "#64748b"]
    x = np.arange(2)
    eksen.bar(x, degerler, color=renkler, width=0.55, edgecolor="white")
    eksen.set_xticks(x)
    eksen.set_xticklabels(isimler, fontsize=10)
    eksen.set_ylim(0, 22)
    eksen.axhline(20, color="#e2e8f0", linestyle="--", linewidth=1)
    eksen.set_ylabel("Not (0–20)")
    eksen.set_title("Sınıfa göre konum: tahmininiz ve ortalama")
    eksen_stili(eksen)
    for i, v in enumerate(degerler):
        eksen.text(i, v + 0.4, f"{v:.1f}", ha="center", fontweight="bold")
    fig.tight_layout()
    return fig
