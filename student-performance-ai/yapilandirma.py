"""
Proje ayarları: veri yolları, sütun adları (CSV İngilizce kalır), risk eşikleri.
"""
from pathlib import Path

# Proje kökü
KOK_DIZIN = Path(__file__).resolve().parent
# Varsayılan (UCI) veri — diğer dosyalar yoksa bu kullanılır
VERI_YOLU = KOK_DIZIN / "data" / "student-mat.csv"
# Büyük veri seti (data/ içine koyduğunuz dosya; varsa en önce bu seçilir)
VERI_YOLU_500K = KOK_DIZIN / "data" / "student_performance_500k.csv"
# İsteğe bağlı alternatif isim
VERI_YOLU_OZEL = KOK_DIZIN / "data" / "veri_ozel.csv"
# Çok büyük dosyalarda eğitim süresi/bellek için üst sınır (rastgele örnek)
VERI_EGITIM_MAX_SATIR = 200_000
# Bu satır sayısının üzerinde Histogram tabanlı Gradyan Artırma kullanılır (daha hızlı)
BUYUK_VERI_ESIK = 25_000
KANIT_KLASORU = KOK_DIZIN / "kanitlar"
KANIT_DOSYASI = KANIT_KLASORU / "model_basari_kaniti.txt"
# Ekran görüntüsü benzeri otomatik kanıt görseli (model adı + başarı %)
KANIT_GORSEL_DOSYASI = KANIT_KLASORU / "model_kanit_ekrani.png"

# Model girdileri ve hedef (G3 asla özellik değildir — veri sızıntısı yok)
OZELLIK_SUTUNLARI = ["studytime", "failures", "absences", "G1", "G2"]
HEDEF_SUTUN = "G3"

# Arayüzde gösterilecek Türkçe özellik adları (CSV sütun anahtarı → etiket)
OZELLIK_ETIKETLERI = {
    "studytime": "Haftalık çalışma (1–4)",
    "failures": "Geçmiş ders tekrarı / başarısızlık (0–5)",
    "absences": "Devamsızlık (gün)",
    "G1": "1. dönem notu (G1)",
    "G2": "2. dönem notu (G2)",
}

# Eğitim / test ayrımı
TEST_ORANI = 0.2
RASTGELE_TOHUM = 42

# Arayüzde seçim yok: tüm proje bu tek modelle çalışır (gerekçe hocaya sözlü anlatılır)
SABIT_PROJE_MODEL_ANAHTARI = "gradyan_artirma"

# Tahmini G3 risk bantları (dahil)
RISK_ZAYIF_UST = 9
RISK_ORTA_UST = 14
# 15–20 → başarılı
