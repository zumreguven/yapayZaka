"""
Tek seçilen regresyon modelini eğitir, testte ölçer, kanıt dosyası yazar.
Büyük veri: örnekleme + HistGradientBoostingRegressor (gradyan_artirma).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from kanit_yazici import performans_kanitini_dosyaya_yaz
from on_isleme import ham_veriyi_yukle, ogrenci_verisini_temizle, ozellik_ve_hedefi_ayir, veri_dosya_yolu_sec
from yapilandirma import (
    BUYUK_VERI_ESIK,
    HEDEF_SUTUN,
    OZELLIK_SUTUNLARI,
    RASTGELE_TOHUM,
    SABIT_PROJE_MODEL_ANAHTARI,
    TEST_ORANI,
    VERI_EGITIM_MAX_SATIR,
)


MODEL_ANAHTARLARI = {
    "dogrusal_regresyon": "Doğrusal Regresyon",
    "karar_agaci": "Karar Ağacı Regresyonu",
    "rastgele_orman": "Rastgele Orman (Random Forest)",
    "gradyan_artirma": "Gradyan Artırma (Gradient Boosting)",
}


def terminal_performans_matrisi_yaz(
    model_etiketi: str,
    r2: float,
    mae: float,
    rmse: float,
    egitim_sayisi: int,
    test_sayisi: int,
    veri_kaynagi: str,
    temiz_satir_toplam: int | None = None,
    egitime_giren_satir: int | None = None,
    orneklenmis_mi: bool = False,
) -> None:
    """Test kümesi metriklerini terminale tablo olarak yazar."""
    basari_str = f"{max(r2, 0) * 100:.1f}".replace(".", ",")
    basari_hucre = f"%{basari_str}" if r2 >= 0 else "— (R² negatif)"

    satirlar = [
        ("Veri dosyası", veri_kaynagi),
        ("Model", model_etiketi),
        ("Değerlendirme", "Yalnızca test kümesi (veri sızıntısı yok)"),
    ]
    if temiz_satir_toplam is not None:
        satirlar.append(("Temiz satır (CSV sonrası)", str(temiz_satir_toplam)))
    if orneklenmis_mi and egitime_giren_satir is not None:
        satirlar.append(("Eğitime alınan (alt örneklem)", str(egitime_giren_satir)))
    satirlar.extend(
        [
            ("R² skoru (belirlilik)", f"{r2:.4f}"),
            ("Başarı oranı (açıklanan varyans, R²×100)", basari_hucre),
            ("MAE", f"{mae:.4f}"),
            ("RMSE", f"{rmse:.4f}"),
            ("Eğitim örnek sayısı", str(egitim_sayisi)),
            ("Test örnek sayısı", str(test_sayisi)),
        ]
    )

    w0 = max(len(s[0]) for s in satirlar)
    w1 = max(len(s[1]) for s in satirlar)
    ust = "+" + "-" * (w0 + 2) + "+" + "-" * (w1 + 2) + "+"
    print("\n")
    print("=" * (len(ust)))
    print("TEST KÜMESİ PERFORMANS MATRİSİ")
    print("=" * (len(ust)))
    print(ust)
    for a, b in satirlar:
        print(f"| {a:<{w0}} | {b:>{w1}} |")
        print(ust)
    print()


class ModelEgitici:
    """Seçilen tek bir sklearn modelini eğiten ve paketleyen sınıf."""

    def __init__(self, model_anahtari: str):
        if model_anahtari not in MODEL_ANAHTARLARI:
            raise ValueError(f"Bilinmeyen model: {model_anahtari}")
        self.model_anahtari = model_anahtari
        self.model_etiketi = MODEL_ANAHTARLARI[model_anahtari]

    def _tahminleyici_olustur(self, buyuk_veri: bool):
        if self.model_anahtari == "dogrusal_regresyon":
            return Pipeline(
                [
                    ("olcekle", StandardScaler()),
                    ("regresyon", LinearRegression()),
                ]
            )
        if self.model_anahtari == "karar_agaci":
            return DecisionTreeRegressor(
                random_state=RASTGELE_TOHUM, max_depth=8, min_samples_leaf=3
            )
        if self.model_anahtari == "rastgele_orman":
            return RandomForestRegressor(
                n_estimators=200,
                random_state=RASTGELE_TOHUM,
                max_depth=12,
                min_samples_leaf=2,
                n_jobs=-1,
            )
        if buyuk_veri:
            return HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                max_depth=6,
                random_state=RASTGELE_TOHUM,
                early_stopping=True,
                validation_fraction=0.05,
                n_iter_no_change=15,
            )
        return GradientBoostingRegressor(
            random_state=RASTGELE_TOHUM,
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
        )

    def egit_ve_paketle(self, cerceve: pd.DataFrame | None = None) -> dict:
        veri_yolu = veri_dosya_yolu_sec()
        if cerceve is None:
            ham = ham_veriyi_yukle()
            cerceve = ogrenci_verisini_temizle(ham)

        temiz_satir_toplam = len(cerceve)
        orneklenmis = False
        if temiz_satir_toplam > VERI_EGITIM_MAX_SATIR:
            print(
                f"\n[{temiz_satir_toplam} satır] Eğitim süresi için {VERI_EGITIM_MAX_SATIR} satıra rastgele indirgeniyor.\n"
            )
            cerceve = cerceve.sample(
                n=VERI_EGITIM_MAX_SATIR, random_state=RASTGELE_TOHUM
            ).reset_index(drop=True)
            orneklenmis = True

        egitime_giren = len(cerceve)
        buyuk_veri = egitime_giren >= BUYUK_VERI_ESIK
        if buyuk_veri and self.model_anahtari == "gradyan_artirma":
            self.model_etiketi = "Gradyan Artırma (Histogram GB — büyük veri)"

        x, y = ozellik_ve_hedefi_ayir(cerceve)
        x_egitim, x_test, y_egitim, y_test = train_test_split(
            x,
            y,
            test_size=TEST_ORANI,
            random_state=RASTGELE_TOHUM,
            shuffle=True,
        )

        tahminleyici = self._tahminleyici_olustur(buyuk_veri)
        tahminleyici.fit(x_egitim, y_egitim)
        y_tahmin_test = tahminleyici.predict(x_test)

        r2 = float(r2_score(y_test, y_tahmin_test))
        mae = float(mean_absolute_error(y_test, y_tahmin_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_tahmin_test)))

        egitim_ortalamalari = x_egitim.mean()
        egitim_std = x_egitim.std().replace(0, 1e-9)

        # Kanıt metninde görünen model adı (Histogram ayrımı)
        kanit_model_adi = self.model_etiketi

        kanit = performans_kanitini_dosyaya_yaz(
            model_etiketi=kanit_model_adi,
            r2_skoru=r2,
            mae=mae,
            rmse=rmse,
            egitim_sayisi=len(x_egitim),
            test_sayisi=len(x_test),
        )
        kanit_metni = kanit["kanit_metni"]
        kanit_gorsel_baytlari = kanit["gorsel_baytlari"]
        kanit_gorsel_yolu = kanit["gorsel_yolu"]

        terminal_performans_matrisi_yaz(
            kanit_model_adi,
            r2,
            mae,
            rmse,
            len(x_egitim),
            len(x_test),
            veri_kaynagi=str(veri_yolu.name),
            temiz_satir_toplam=temiz_satir_toplam,
            egitime_giren_satir=egitime_giren,
            orneklenmis_mi=orneklenmis,
        )

        # Sabit etiketi geri al (sonraki çağrı için)
        self.model_etiketi = MODEL_ANAHTARLARI[self.model_anahtari]

        return {
            "model_anahtari": self.model_anahtari,
            "model_etiketi": kanit_model_adi,
            "tahminleyici": tahminleyici,
            "x_egitim": x_egitim,
            "x_test": x_test,
            "y_egitim": y_egitim,
            "y_test": y_test,
            "y_tahmin_test": y_tahmin_test,
            "r2_skoru": r2,
            "mae": mae,
            "rmse": rmse,
            "egitim_ortalamalari": egitim_ortalamalari,
            "egitim_std": egitim_std,
            "ozellik_listesi": list(OZELLIK_SUTUNLARI),
            "hedef_sutun": HEDEF_SUTUN,
            "test_g3_ortalamasi": float(y_test.mean()),
            "kanit_metni": kanit_metni,
            "kanit_gorsel_baytlari": kanit_gorsel_baytlari,
            "kanit_gorsel_yolu": kanit_gorsel_yolu,
            "satir_sayisi": len(cerceve),
            "veri_dosyasi": str(veri_yolu),
        }


def ozellik_onemlerini_cikar(model_anahtari: str, tahminleyici, ozellik_isimleri: list[str]):
    """Grafik için (isim, göreli önem) listeleri."""
    if model_anahtari == "dogrusal_regresyon":
        reg = tahminleyici.named_steps["regresyon"]
        katsayi = np.abs(reg.coef_.ravel())
        toplam = katsayi.sum() or 1.0
        etiketler = [ozellik_etiket_cevir(n) for n in ozellik_isimleri]
        return etiketler, (katsayi / toplam).tolist()

    if hasattr(tahminleyici, "feature_importances_"):
        onem = np.asarray(tahminleyici.feature_importances_).ravel()
        toplam = float(onem.sum()) or 1.0
        etiketler = [ozellik_etiket_cevir(n) for n in ozellik_isimleri]
        return etiketler, (onem / toplam).tolist()

    n = len(ozellik_isimleri)
    return [ozellik_etiket_cevir(x) for x in ozellik_isimleri], [1.0 / n] * n


def ozellik_etiket_cevir(sutun_adi: str) -> str:
    from yapilandirma import OZELLIK_ETIKETLERI

    return OZELLIK_ETIKETLERI.get(sutun_adi, sutun_adi)


if __name__ == "__main__":
    ModelEgitici(SABIT_PROJE_MODEL_ANAHTARI).egit_ve_paketle()
