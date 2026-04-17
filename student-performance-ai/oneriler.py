"""
Kullanıcı girdilerine göre sade Türkçe analiz ve tavsiyeler (kural tabanlı).
"""
from __future__ import annotations

import pandas as pd


def tavsiye_listesi_analiz(
    girdiler: dict,
    egitim_ort: pd.Series,
    tahmini_g3: float,
) -> list[str]:
    """
    Devamsızlık / çalışma süresi vb. için doğrudan ifadeler:
    fazla/az/yeterli/gayet yeterli.
    """
    satirlar: list[str] = []

    calisma = float(girdiler["studytime"])
    tekrar = float(girdiler["failures"])
    devamsizlik = float(girdiler["absences"])
    g1 = float(girdiler["G1"])
    g2 = float(girdiler["G2"])

    o_calisma = float(egitim_ort["studytime"])
    o_tekrar = float(egitim_ort["failures"])
    o_dev = float(egitim_ort["absences"])
    o_g1 = float(egitim_ort["G1"])
    o_g2 = float(egitim_ort["G2"])

    # Devamsızlık
    if devamsizlik > o_dev + 0.5:
        satirlar.append(
            f"Devamsızlık sayınız **{devamsizlik:.0f} gün**; veri setindeki öğrenci ortalamasına göre (**{o_dev:.1f}**) **fazla**. "
            "Devamsızlığı azaltmak, dersten geçme şansınızı artırmanın en somut yollarından biridir."
        )
    elif devamsizlik < max(0, o_dev - 2):
        satirlar.append(
            f"Devamsızlık sayınız **{devamsizlik:.0f} gün**; ortalamaya göre **düşük**. Düzenli devamınız güçlü bir avantaj."
        )
    else:
        satirlar.append(
            f"Devamsızlık sayınız **{devamsizlik:.0f} gün**; kabaca **ortalama civarında**. Gereksiz devamsızlıkları azaltmaya devam edin."
        )

    # Çalışma süresi (1–4 ölçeği)
    if calisma < o_calisma - 0.01:
        satirlar.append(
            f"Haftalık çalışma düzeyiniz **{calisma:.0f}/4**; ortalamaya (**{o_calisma:.1f}/4**) göre **düşük**. "
            "**Çalışma saatiniz az** görünüyor; haftalık planlı çalışma süresini artırmanız önerilir."
        )
    elif calisma > o_calisma + 0.01:
        satirlar.append(
            f"Haftalık çalışma düzeyiniz **{calisma:.0f}/4**; ortalamadan **yüksek**. "
            "**Çalışma saatiniz gayet yeterli**; bu düzeyi koruyun."
        )
    else:
        satirlar.append(
            f"Haftalık çalışma düzeyiniz **{calisma:.0f}/4**; ortalamaya (**{o_calisma:.1f}/4**) **yakın**. "
            "**Çalışma saatiniz yeterli** sayılabilir; mümkünse bir kademe daha artırmayı deneyebilirsiniz."
        )

    # Başarısızlık / tekrar
    if tekrar > o_tekrar + 0.01:
        satirlar.append(
            f"Geçmiş ders tekrarı / başarısızlık sayınız (**{tekrar:.0f}**) ortalamadan (**{o_tekrar:.1f}**) yüksek; "
            "zayıf kaldığınız konularda ek pekiştirme faydalı olur."
        )
    elif tekrar == 0 and o_tekrar > 0.3:
        satirlar.append("Geçmiş başarısızlık kaydınız yok; bu olumlu bir profil.")

    # Notlar
    if g2 < o_g2 - 1:
        satirlar.append(
            f"2. dönem notunuz (**{g2:.0f}**) sınıf ortalamasının (**{o_g2:.1f}**) altında; "
            "final öncesi konu tekrarı ve deneme çözümüne ağırlık verin."
        )
    elif g2 > o_g2 + 1:
        satirlar.append(
            f"2. dönem notunuz (**{g2:.0f}**) ortalamanın üzerinde; iyi gidiyorsunuz, tempoyu koruyun."
        )

    if g1 < g2 - 2:
        satirlar.append(
            "1. dönem notunuz 2. döneme göre geride kalmıştı; yükselişi sürdürmek için çalışma düzeninizi bozmayın."
        )

    # Tahmine bağlı genel cümle
    if tahmini_g3 < 10:
        satirlar.append(
            "Tahmini final notu düşük banda yakın; öncelik devam, çalışma süresi ve zayıf konulara destek olmalı."
        )
    elif tahmini_g3 < 15:
        satirlar.append(
            "Tahmini final notu orta banda yakın; düzenli tekrar ve sınav hazırlığı ile yükseltme mümkün."
        )
    else:
        satirlar.append(
            "Tahmini final notu güçlü bir bantta; mevcut çalışma ve devam alışkanlıklarınızı koruyun."
        )

    return satirlar
