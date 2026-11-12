"""Arma entrega/Entregable_Final_CDP.pdf a partir de contenido.html, estilo.css y figuras/.

Uso: python3 entrega/build.py   (o `make pdf`). Solo necesita Python 3 y google-chrome.
Las tablas de resultados se leen de reports/gcp/phase2_tables.json para no copiar cifras a mano.
"""
import json, pathlib, re, shutil, subprocess, sys, tempfile

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent
SALIDA = AQUI / "Entregable_Final_CDP.pdf"
CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

AWS_COLUMNA = "AWS (el servicio se probó; lo demás, documentación)"


def num(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}".replace(".", ",")


def tabla_resultados():
    d = json.loads((RAIZ / "reports/gcp/phase2_tables.json").read_text())
    plat = {"local": "Python local", "vertex_pipeline": "Vertex Pipelines", "automl": "Vertex AutoML"}
    filas = []
    for r in d["rows"]:
        lo, hi = r["pr_auc_ci"]
        filas.append(
            f"<tr><td>{plat[r['platform']]}</td><td>{r['label']}</td>"
            f"<td class='n'>{num(r['pr_auc'])} [{num(lo)}; {num(hi)}]</td><td class='n'>{num(r['roc_auc'])}</td>"
            f"<td class='n'>{num(r['gini'])}</td><td class='n'>{num(r['f1'])}</td><td class='n'>{num(r['precision'])}</td>"
            f"<td class='n'>{num(r['recall'])}</td><td class='n'>{num(r['accuracy'] * 100, 1)} %</td>"
            f"<td class='n'>{num(r['brier'])}</td></tr>")
    cab = ("<tr><th>Plataforma</th><th>Modelo</th><th>PR-AUC [IC 95 %]</th><th>ROC-AUC</th><th>Gini</th><th>F1</th>"
           "<th>Precisión</th><th>Recall</th><th>Accuracy</th><th>Brier</th></tr>")
    return f"<table>{cab}{''.join(filas)}</table>"


def numerar(html):
    ids = {}
    cuenta = {"F": 0, "T": 0}

    def sustituir(m):
        tipo = m.group(1)
        cuenta[tipo] += 1
        return f"{'Figura' if tipo == 'F' else 'Tabla'} {cuenta[tipo]}"

    # Primera pasada: asignar numero a cada figura/tabla con id, en orden de aparicion.
    n = {"F": 0, "T": 0}
    for m in re.finditer(r'(?:id="([^"]+)"[^>]*>(?:(?!</figure>|</p>).)*?\{\{([FT])\}\})', html, re.S):
        n[m.group(2)] += 1
        ids[m.group(1)] = ("figura " if m.group(2) == "F" else "tabla ") + str(n[m.group(2)])
    html = re.sub(r"\{\{([FT])\}\}", sustituir, html)
    return re.sub(r"\{\{ref:([^}]+)\}\}", lambda m: ids[m.group(1)], html)


def main():
    html = (AQUI / "contenido.html").read_text()
    html = html.replace("{{TABLA_RESULTADOS}}", tabla_resultados()).replace("{{AWS_COLUMNA}}", AWS_COLUMNA)
    html = numerar(html)
    css = (AQUI / "estilo.css").read_text()
    pagina = f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Entregable final CDP</title><style>{css}</style></head><body>{html}</body></html>'
    tmp = AQUI / "_pagina.html"
    tmp.write_text(pagina)
    subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={SALIDA}", f"file://{tmp}"], check=True, capture_output=True, timeout=180)
    tmp.unlink()
    print(f"Listo: {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
