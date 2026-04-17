"""
Öğrenci özeti PDF — model / hoca metrikleri yok; yalnızca kullanıcı sonuçları.
"""
from __future__ import annotations

from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def pdf_raporu_olustur(
    girdiler: dict,
    girdi_etiketleri: dict[str, str],
    tahmini_g3: float,
    gecme_olasiligi: str,
    sinif_analiz_paragrafi: str,
    oneriler: list[str],
) -> bytes:
    tampon = BytesIO()
    belge = SimpleDocTemplate(
        tampon,
        pagesize=letter,
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    stiller = getSampleStyleSheet()
    baslik_stili = ParagraphStyle(
        "BaslikOzel",
        parent=stiller["Heading1"],
        fontSize=18,
        spaceAfter=14,
        textColor=colors.HexColor("#0f172a"),
    )
    govde = ParagraphStyle(
        "GovdeOzel",
        parent=stiller["Normal"],
        fontSize=10,
        leading=14,
    )
    hikaye = []

    hikaye.append(Paragraph("Öğrenci Performans Özeti", baslik_stili))
    hikaye.append(Spacer(1, 0.15 * inch))

    satirlar = [[girdi_etiketleri.get(k, k), str(v)] for k, v in girdiler.items()]
    tablo = Table([["Girdi", "Değer"]] + satirlar, colWidths=[2.4 * inch, 2.0 * inch])
    tablo.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    hikaye.append(tablo)
    hikaye.append(Spacer(1, 0.2 * inch))

    hikaye.append(Paragraph(f"<b>Tahmini final notu (G3):</b> {tahmini_g3:.2f} / 20", govde))
    hikaye.append(
        Paragraph(
            f"<b>Dersten geçme olasılığı (tahmine göre özet):</b> {escape(gecme_olasiligi)}",
            govde,
        )
    )
    hikaye.append(Spacer(1, 0.12 * inch))
    hikaye.append(Paragraph("<b>Sınıfa göre başarı analizi</b>", govde))
    hikaye.append(Paragraph(escape(sinif_analiz_paragrafi), govde))
    hikaye.append(Spacer(1, 0.15 * inch))

    hikaye.append(Paragraph("<b>Analiz edilip öneriler</b>", govde))
    for i, oneri in enumerate(oneriler, 1):
        temiz = oneri.replace("**", "")
        hikaye.append(Paragraph(f"{i}. {escape(temiz)}", govde))

    belge.build(hikaye)
    pdf_baytlar = tampon.getvalue()
    tampon.close()
    return pdf_baytlar
