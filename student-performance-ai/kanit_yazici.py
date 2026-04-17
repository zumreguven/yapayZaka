"""Test özetini TXT ve PNG olarak kanitlar/ altına yazar."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from kanit_gorseli import model_kanit_ekrani_png_kaydet
from yapilandirma import KANIT_DOSYASI, KANIT_GORSEL_DOSYASI, KANIT_KLASORU


def _yuzde_ifadesi_tr(r2_skoru: float) -> str:
    """Türkçe ondalık virgül ile yüzde metni."""
    if r2_skoru >= 0:
        return f"{r2_skoru * 100:.1f}".replace(".", ",")
    return "anlamlı bir yüzde olarak verilemez (R² negatif)"


def performans_kanitini_dosyaya_yaz(
    model_etiketi: str,
    r2_skoru: float,
    mae: float,
    rmse: float,
    egitim_sayisi: int,
    test_sayisi: int,
    hedef_txt: Path | None = None,
    hedef_png: Path | None = None,
) -> dict:
    """
    1) PNG kanıt ekranını kaydeder.
    2) TXT'ye kullanıcının istediği net cümle + teknik özet yazar.
    Dönüş: kanit_metni, gorsel_yolu, gorsel_baytlari
    """
    KANIT_KLASORU.mkdir(parents=True, exist_ok=True)
    txt_yol = hedef_txt or KANIT_DOSYASI
    png_yol = hedef_png or KANIT_GORSEL_DOSYASI

    model_kanit_ekrani_png_kaydet(
        model_etiketi=model_etiketi,
        r2_skoru=r2_skoru,
        mae=mae,
        rmse=rmse,
        egitim_sayisi=egitim_sayisi,
        test_sayisi=test_sayisi,
        hedef_yol=png_yol,
    )
    gorsel_baytlari = png_yol.read_bytes()

    yuzde_str = _yuzde_ifadesi_tr(r2_skoru)
    simdi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if r2_skoru >= 0:
        ana_cumle = (
            f"Bu projede «{model_etiketi}» modeli kullanılmıştır ve başarı oranı yüzde {yuzde_str}'dir.\n\n"
            f"(Başarı oranı: yalnızca test kümesinde hesaplanan R² belirlilik katsayısına karşılık gelen "
            f"açıklanan varyans yüzdesidir; eğitim verisi bu ölçümde kullanılmamıştır.)"
        )
    else:
        ana_cumle = (
            f"Bu projede «{model_etiketi}» modeli kullanılmıştır; testte R² negatif olduğu için "
            f"başarı oranı anlamlı bir yüzde olarak ifade edilememektedir. Başka bir model seçilmesi önerilir."
        )

    metin = f"""================================================================================
MODEL TEST ÖZETİ (TXT)
================================================================================
Oluşturulma zamanı: {simdi}

--- ÖZET ---
{ana_cumle}

--- İLİŞKİLİ GÖRSEL ---
Aynı özet değerleri şu PNG dosyasında da yer alır: {png_yol.name}
Tam yol: {png_yol.resolve()}

--- TEKNİK DETAY (TEST KÜMESİ) ---
R² skoru: {r2_skoru:.4f}
Ortalama mutlak hata (MAE): {mae:.4f}
Kök ortalama kare hata (RMSE): {rmse:.4f}
Eğitim örnek sayısı: {egitim_sayisi}
Test örnek sayısı: {test_sayisi}

Metin dosyası yolu: {txt_yol.resolve()}
================================================================================
"""
    txt_yol.write_text(metin, encoding="utf-8")

    return {
        "kanit_metni": metin,
        "gorsel_yolu": png_yol,
        "gorsel_baytlari": gorsel_baytlari,
    }
