"""Arma Entregable_Final_CDP.docx y .pdf desde contenido.py (una sola fuente).
Uso: python3 entrega/build.py [--final]. Necesita python-docx y LibreOffice (soffice).
Las tablas de resultados se leen de reports/gcp/phase2_tables.json.
Mientras quede algun token [[...]] se avisa; con --final el programa termina con error."""
import json, pathlib, re, shutil, subprocess, sys
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
from contenido import BLOQUES  # noqa: E402

DOCX = AQUI / "Entregable_Final_CDP.docx"
FUENTE, MONO, NEGRO = "Times New Roman", "Courier New", RGBColor(0, 0, 0)
MAX_PAGINAS = 6


def num(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}".replace(".", ",")


def filas_resultados():
    d = json.loads((RAIZ / "reports/gcp/phase2_tables.json").read_text())
    r = {x["id"]: x for x in d["rows"]}
    orden = [("Logística", "local_logistic", "Local y Vertex"), ("XGBoost", "local_xgboost", "Local y Vertex"),
             ("LightGBM", "local_lightgbm", "Local y Vertex"), ("Heurística", "local_heuristic", "Local"),
             ("AutoML v1", "automl_v1", "Vertex AutoML"), ("AutoML v2", "automl_v2", "Vertex AutoML")]
    out = []
    for nombre, k, plat in orden:
        x = r[k]; lo, hi = x["pr_auc_ci"]
        out.append([nombre, plat, f"{num(x['pr_auc'])} [{num(lo)}; {num(hi)}]", num(x["roc_auc"]), num(x["gini"]),
                    num(x["f1"]), num(x["recall"]), f"{num(x['accuracy'] * 100, 1)} %"])
    return out


def fuente(run, mono=False, size=None, bold=False):
    run.font.name = MONO if mono else FUENTE
    run._element.rPr.rFonts.set(qn("w:eastAsia"), MONO if mono else FUENTE)
    run.font.color.rgb = NEGRO
    run.bold = bold
    if size: run.font.size = Pt(size - 0.5 if mono else size)


def texto(par, s, size, bold=False):
    for i, trozo in enumerate(re.split(r"`", s)):
        if trozo:
            fuente(par.add_run(trozo), mono=i % 2 == 1, size=size, bold=bold)


def espaciado(par, antes=0, despues=4, interlineado=1.0):
    f = par.paragraph_format
    f.space_before, f.space_after, f.line_spacing = Pt(antes), Pt(despues), interlineado


def bordes(tabla):
    tbl = tabla._tbl
    pr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
        b.append(e)
    pr.append(b)
    m = OxmlElement("w:tblCellMar")
    for lado, v in (("top", 20), ("left", 60), ("bottom", 20), ("right", 60)):
        e = OxmlElement(f"w:{lado}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); m.append(e)
    pr.append(m)


def tabla(doc, cab, filas, anchos, pie, size=8.5):
    if pie:
        p = doc.add_paragraph(); texto(p, pie, 9); espaciado(p, 4, 2); p.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(filas), cols=len(cab))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    bordes(t)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(lay)
    for j, a in enumerate(anchos): t.columns[j].width = Cm(a)
    for i, fila in enumerate([cab] + filas):
        for j, c in enumerate(fila):
            cel = t.cell(i, j); cel.width = Cm(anchos[j])
            p = cel.paragraphs[0]; espaciado(p, 0, 0)
            texto(p, c, size, bold=(i == 0))
        if i == 0:
            trPr = t.rows[0]._tr.get_or_add_trPr(); h = OxmlElement("w:tblHeader"); trPr.append(h)
        cant = OxmlElement("w:cantSplit"); t.rows[i]._tr.get_or_add_trPr().append(cant)
    espaciado(doc.add_paragraph(), 0, 2)


def figura(doc, archivo, ancho, pie, n):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; espaciado(p, 3, 1)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(AQUI / "figuras" / archivo), width=Cm(ancho))
    c = doc.add_paragraph(); texto(c, f"Figura {n}. {pie}", 9); espaciado(c, 0, 5)


def marcos(doc, pies, n0):
    t = doc.add_table(rows=2, cols=len(pies)); t.autofit = False; bordes(t)
    for j, pie in enumerate(pies):
        w = Cm(17.4 / len(pies))
        a, b = t.cell(0, j), t.cell(1, j)
        a.width = b.width = w
        a.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        texto(a.paragraphs[0], "Espacio para la captura", 7.5); espaciado(a.paragraphs[0], 14, 14)
        texto(b.paragraphs[0], f"Figura {n0 + j}. [captura pendiente: {pie}]", 7.5); espaciado(b.paragraphs[0], 0, 0)
    espaciado(doc.add_paragraph(), 0, 2)
    return n0 + len(pies)


def pie_pagina(sec):
    p = sec.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(); fuente(r, size=9)
    for tipo, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tipo:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tipo)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r._r.append(e)


def construir():
    shutil.copy(RAIZ / "docs/gcp/plots/metric_bars.png", AQUI / "figuras" / "metric_bars.png")
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin, sec.bottom_margin = Cm(1.5), Cm(1.6)
    pie_pagina(sec)
    st = doc.styles["Normal"]; st.font.name = FUENTE; st.font.size = Pt(10.5); st.font.color.rgb = NEGRO
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FUENTE)
    n = 0
    for b in BLOQUES:
        k = b[0]
        if k == "titulo":
            p = doc.add_paragraph(); texto(p, b[1], 15, bold=True); espaciado(p, 0, 4)
        elif k == "sub":
            p = doc.add_paragraph(); texto(p, b[1], 9); espaciado(p, 0, 1)
        elif k == "h":
            p = doc.add_paragraph(); texto(p, b[1], 11.5, bold=True); espaciado(p, 7, 3)
            p.paragraph_format.keep_with_next = True
            p.style = doc.styles["Heading 1"]
            for r in p.runs: fuente(r, size=11.5, bold=True)
        elif k == "p":
            p = doc.add_paragraph(); texto(p, b[1], 10.5); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; espaciado(p, 0, 4)
        elif k == "tabla":
            tabla(doc, b[1], b[2], b[3], b[4])
        elif k == "tabla_resultados":
            tabla(doc, ["Modelo", "Plataforma", "PR-AUC [IC 95 %]", "ROC-AUC", "Gini", "F1", "Recall", "Accuracy"],
                  filas_resultados(), [2.2, 2.7, 4.2, 1.8, 1.6, 1.5, 1.7, 1.7], b[1])
        elif k == "fig":
            n += 1; figura(doc, b[1], b[2], b[3], n)
        elif k == "marcos":
            n = marcos(doc, b[1], n + 1) - 1
    for nombre in ("Heading 1",):
        s = doc.styles[nombre]; s.font.color.rgb = NEGRO; s.font.name = FUENTE
    doc.core_properties.title = "Entregable final CDP"; doc.core_properties.author = ""
    doc.save(DOCX)


def tokens():
    return sorted(set(re.findall(r"\[\[[A-Z_]+\]\]", json.dumps(BLOQUES, ensure_ascii=False))))


def main():
    construir()
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(AQUI), str(DOCX)],
                   check=True, capture_output=True, timeout=180)
    pdf = AQUI / "Entregable_Final_CDP.pdf"
    pag = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
    print(f"Listo: {DOCX.name} y {pdf.name} ({pag} páginas)")
    error = 0
    if pag > MAX_PAGINAS:
        print(f"ERROR: el PDF tiene {pag} páginas y el máximo es {MAX_PAGINAS}"); error = 1
    t = tokens()
    if t:
        print("*" * 70 + f"\nATENCIÓN: faltan datos, quedan estos tokens en el texto: {', '.join(t)}\n" + "*" * 70)
        if "--final" in sys.argv: error = 1
    return error


if __name__ == "__main__":
    sys.exit(main())
