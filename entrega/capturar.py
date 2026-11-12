"""Regenera las capturas de pantalla de figuras/ con Chrome headless (no hace falta iniciar sesion)."""
import subprocess, tempfile, pathlib, html
from PIL import Image

AQUI = pathlib.Path(__file__).parent
FIG = AQUI / "figuras"
RAIZ = AQUI.parent
CHROME = "google-chrome"
GH = "https://github.com/rosvend/MLOps"
PAGINAS = {
    "deploy_ok": (f"{GH}/actions/runs/43121043675", "1280,1000"),
    "deploy_rollback": (f"{GH}/actions/runs/43119956072", "1280,1000"),
    "train_keep": (f"{GH}/actions/runs/43044039876", "1280,1000"),
    "train_aprobacion": (f"{GH}/actions/runs/43046530838", "1280,1000"),
    "drift": (f"{GH}/actions/runs/43048336590", "1280,1000"),
}


def captura(url, salida, tam):
    subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={tam}", "--virtual-time-budget=15000", f"--screenshot={salida}", url],
                   check=True, capture_output=True, timeout=120)


def recortar(img, caja):
    return Image.open(img).convert("RGB").crop(caja)


def apilar(imgs, hueco=14):
    ancho = max(i.width for i in imgs)
    out = Image.new("RGB", (ancho, sum(i.height for i in imgs) + hueco * (len(imgs) - 1)), "white")
    y = 0
    for i in imgs:
        out.paste(i, (0, y)); y += i.height + hueco
    return out


def terminal(texto, salida):
    pagina = ("<body style='margin:0;background:#1e1e1e'><pre style=\"margin:0;padding:18px 22px;color:#e6e6e6;"
              "font:15px/1.45 'DejaVu Sans Mono',monospace;white-space:pre-wrap\">" + html.escape(texto) + "</pre></body>")
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(pagina)
    captura("file://" + f.name, salida, "980,300")


if __name__ == "__main__":
    tmp = pathlib.Path(tempfile.mkdtemp())
    for nombre, (url, tam) in PAGINAS.items():
        captura(url, tmp / f"{nombre}.png", tam)
    ancho = 1280
    apilar([recortar(tmp / "deploy_ok.png", (0, 175, ancho, 545)),
            recortar(tmp / "deploy_rollback.png", (0, 175, ancho, 545))]).save(FIG / "gh_deploy.png")
    apilar([recortar(tmp / "train_keep.png", (0, 175, ancho, 545)),
            recortar(tmp / "drift.png", (0, 175, ancho, 545))]).save(FIG / "gh_train_drift.png")
    recortar(tmp / "train_aprobacion.png", (0, 175, ancho, 850)).save(FIG / "gh_aprobacion.png")
    out = subprocess.run("uv run pytest tests/test_repo_structure.py -q 2>&1 | tail -3; echo; "
                         "echo '$ make check-metrics'; make check-metrics 2>&1 | tail -2",
                         shell=True, cwd=RAIZ, capture_output=True, text=True).stdout
    terminal("$ uv run pytest tests/test_repo_structure.py -q\n" + out, tmp / "term.png")
    im = Image.open(tmp / "term.png")
    im.crop((0, 0, im.width, 170)).save(FIG / "terminal_ci.png")
