"""
Öğrenci başarı tahmini — Hesapla ile sonuç; öneriler ve kullanıcı grafikleri.
Model test metrikleri yalnızca terminalde (matris). Arayüzde hoca/TXT/PNG yok.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analitik import (
    dersten_gecme_olasiligi_sinifi,
    ortalamayla_karsilastir,
    sanal_siralama,
    testte_yuzde_kac_ogrenciden_yuksek,
)
from grafikler import ciz_kullanici_girdi_panelleri, ciz_tahmin_sinif_karsilastirma
from model_egitimi import ModelEgitici
from oneriler import tavsiye_listesi_analiz
from pdf_rapor import pdf_raporu_olustur
from yapilandirma import OZELLIK_ETIKETLERI, SABIT_PROJE_MODEL_ANAHTARI

st.set_page_config(
    page_title="Öğrenci Başarı Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .ust-bant {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0c4a6e 100%);
        padding: 1.35rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 30px rgba(15,23,42,0.2);
    }
    .ust-bant h1 {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .ust-bant p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.98rem;
    }
    .kart {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 12px rgba(15,23,42,0.06);
    }
    /* Streamlit temasında h4 bazen görünmez; başlıkları zorunlu koyu yap */
    .kart h4 {
        color: #0f172a !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    .tahmin-sayi {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0284c7, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }
    .gecme-etiket {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 0.35rem;
    }
    .gecme-dusuk { background: #fee2e2; color: #991b1b; }
    .gecme-orta { background: #fef9c3; color: #854d0e; }
    .gecme-yuksek { background: #dcfce7; color: #166534; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


def gecme_stil_sinifi(kademe: str) -> str:
    if kademe == "dusuk":
        return "gecme-etiket gecme-dusuk"
    if kademe == "orta":
        return "gecme-etiket gecme-orta"
    return "gecme-etiket gecme-yuksek"


@st.cache_resource(show_spinner="Model yükleniyor…")
def egitilmis_paketi_al() -> dict:
    return ModelEgitici(SABIT_PROJE_MODEL_ANAHTARI).egit_ve_paketle()


def oturum_baslatici():
    varsayilan = {
        "studytime": 2,
        "failures": 0,
        "absences": 4,
        "G1": 12,
        "G2": 12,
    }
    for anahtar, deger in varsayilan.items():
        if anahtar not in st.session_state:
            st.session_state[anahtar] = deger
    if "hesaplandi" not in st.session_state:
        st.session_state.hesaplandi = False


oturum_baslatici()

paket = egitilmis_paketi_al()
tahminleyici = paket["tahminleyici"]
y_test = paket["y_test"]
sinif_ort = paket["test_g3_ortalamasi"]
egitim_ort = paket["egitim_ortalamalari"]

st.sidebar.markdown("### Bilgileriniz")
calisma = st.sidebar.slider(OZELLIK_ETIKETLERI["studytime"], 1, 4, key="studytime")
tekrar = st.sidebar.slider(OZELLIK_ETIKETLERI["failures"], 0, 5, key="failures")
devamsizlik = st.sidebar.slider(OZELLIK_ETIKETLERI["absences"], 0, 30, key="absences")
g1 = st.sidebar.slider(OZELLIK_ETIKETLERI["G1"], 0, 20, key="G1")
g2 = st.sidebar.slider(OZELLIK_ETIKETLERI["G2"], 0, 20, key="G2")

if st.sidebar.button("Hesapla", type="primary", use_container_width=True):
    g = {
        "studytime": int(calisma),
        "failures": int(tekrar),
        "absences": int(devamsizlik),
        "G1": int(g1),
        "G2": int(g2),
    }
    x_df = pd.DataFrame([g])
    ham = float(tahminleyici.predict(x_df)[0])
    st.session_state.hesaplandi = True
    st.session_state.sgirdiler = g
    st.session_state.stahmin = float(np.clip(ham, 0, 20))
    st.rerun()

st.markdown(
    '<div class="ust-bant"><h1>Akademik Performans Tahmini</h1>'
    "<p>Bilgilerinizi girin, <strong>Hesapla</strong> ile tahmin ve önerilerinizi alın.</p></div>",
    unsafe_allow_html=True,
)

if not st.session_state.hesaplandi:
    st.info(
        "👈 Sol menüden haftalık çalışma, devamsızlık ve not bilgilerinizi girin; ardından **Hesapla** düğmesine basın. "
        "Tahmini not, sınıfa göre analiz, öneriler ve grafikler burada görünecektir."
    )
    st.stop()

girdiler = st.session_state.sgirdiler
tahmin = st.session_state.stahmin

gecme_metin, gecme_kademe = dersten_gecme_olasiligi_sinifi(tahmin)
karsilastirma = ortalamayla_karsilastir(tahmin, sinif_ort)
sira, toplam_sira = sanal_siralama(tahmin, y_test)
yuzde_yuksek = testte_yuzde_kac_ogrenciden_yuksek(tahmin, y_test)

sinif_analiz_metni = (
    f"Tahmininiz, referans sınıf ortalamasına göre {karsilastirma.lower()}. "
    f"Benzer profiller arasında tahmin sıranız yaklaşık **{sira} / {toplam_sira}** "
    f"(1 en yüksek tahmin). Tahmininiz, bu referans gruptaki öğrencilerin yaklaşık "
    f"**%{yuzde_yuksek:.1f}**'inden daha yüksek bir final notu öngörüsüne denk geliyor."
)

oneri_listesi = tavsiye_listesi_analiz(girdiler, egitim_ort, tahmin)

c1, c2, c3 = st.columns([1.05, 1.0, 1.15])

with c1:
    st.markdown(
        f'<div class="kart"><h4>Tahmini final notu (G3)</h4>'
        f'<div class="tahmin-sayi">{tahmin:.1f}</div>'
        f'<span style="color:#64748b"> / 20</span></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div class="kart"><h4>Dersten geçme olasılığı</h4>'
        f'<span class="{gecme_stil_sinifi(gecme_kademe)}">{gecme_metin}</span>'
        f'<p style="color:#64748b;font-size:0.82rem;margin:0.75rem 0 0 0;">'
        "Tahmine göre özet: düşük banda yakınsa <strong>Riskli</strong>, orta banda <strong>Orta</strong>, "
        "yüksek banda <strong>Başarılı</strong>.</p></div>",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        '<div class="kart"><h4>Sınıfa göre başarı</h4></div>',
        unsafe_allow_html=True,
    )
    st.metric("Referans sınıf ortalaması (G3)", f"{sinif_ort:.2f}")
    st.markdown(sinif_analiz_metni)

st.markdown("---")
st.subheader("Analiz edilip öneriler")
for satir in oneri_listesi:
    st.markdown(f"- {satir}")

st.markdown("---")
st.subheader("Grafikler")

gc1, gc2 = st.columns(2)
with gc1:
    fig_a = ciz_kullanici_girdi_panelleri(girdiler)
    st.pyplot(fig_a, clear_figure=True)
    plt.close(fig_a)

with gc2:
    fig_b = ciz_tahmin_sinif_karsilastirma(tahmin, sinif_ort)
    st.pyplot(fig_b, clear_figure=True)
    plt.close(fig_b)

pdf_baytlar = pdf_raporu_olustur(
    girdiler=girdiler,
    girdi_etiketleri=OZELLIK_ETIKETLERI,
    tahmini_g3=tahmin,
    gecme_olasiligi=gecme_metin,
    sinif_analiz_paragrafi=sinif_analiz_metni.replace("**", ""),
    oneriler=oneri_listesi,
)

st.markdown("---")
st.download_button(
    label="Özeti PDF olarak indir",
    data=pdf_baytlar,
    file_name="ogrenci_performans_ozeti.pdf",
    mime="application/pdf",
    type="primary",
    use_container_width=False,
)
