"""Regenera las capturas de figuras/ con Chrome headless (paginas publicas, sin iniciar sesion)."""
import subprocess, tempfile, pathlib
from PIL import Image

FIG = pathlib.Path(__file__).parent / "figuras"
GH = "https://github.com/rosvend/MLOps/actions/runs"
RUNS = {"ok": "43138809064", "rollback": "43140255090", "aprob": "43046530838"}


def captura(url, salida, tam, escala=1):
    subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={tam}", f"--force-device-scale-factor={escala}", "--virtual-time-budget=15000",
                    f"--screenshot={salida}", url], check=True, capture_output=True, timeout=120)


def apilar(imgs, hueco=10):
    out = Image.new("RGB", (max(i.width for i in imgs), sum(i.height for i in imgs) + hueco * (len(imgs) - 1)), "white")
    y = 0
    for i in imgs:
        out.paste(i, (0, y)); y += i.height + hueco
    return out


if __name__ == "__main__":
    tmp = pathlib.Path(tempfile.mkdtemp())
    for nombre, run in RUNS.items():
        captura(f"{GH}/{run}", tmp / f"{nombre}.png", "1280,1000")
    abrir = lambda n, caja: Image.open(tmp / f"{n}.png").convert("RGB").crop(caja)
    apilar([abrir("ok", (350, 190, 1270, 530)), abrir("rollback", (350, 190, 1270, 530))]).save(FIG / "gh_canary.png")
    abrir("aprob", (350, 380, 1270, 850)).save(FIG / "gh_aprobacion.png")
    captura((FIG / "arquitectura.svg").resolve().as_uri(), FIG / "arquitectura.png", "1000,470", 2)
