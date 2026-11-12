"""Load test for the scoring API on Cloud Run. See loadtest/README.md for the commands."""

import os
import random
import subprocess
import time

from locust import HttpUser, between, task

_REFRESH_SECONDS = 45 * 60  # an ID token lives about an hour
_token = {"value": os.environ.get("ID_TOKEN", ""), "at": time.time()}


def bearer() -> str:
    """The ID token from ID_TOKEN, or a fresh one from gcloud every 45 minutes."""
    if os.environ.get("ID_TOKEN"):
        return _token["value"]
    if not _token["value"] or time.time() - _token["at"] > _REFRESH_SECONDS:
        out = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True, check=True)
        _token.update(value=out.stdout.strip(), at=time.time())
    return _token["value"]


def application(rng: random.Random, n: int) -> dict:
    """A plausible application: money is log-normal, a few optional fields are sometimes absent."""
    salary = int(rng.lognormvariate(14.9, 0.5))
    term = rng.choice([6, 12, 12, 18, 24, 24, 36, 48])
    capital = int(rng.lognormvariate(15.0, 0.7))
    record = {
        "application_id": f"LOAD-{n}",
        "tipo_credito": rng.choice(["4", "4", "9", "10"]),
        "capital_prestado": capital,
        "plazo_meses": term,
        "edad_cliente": rng.randint(20, 68),
        "tipo_laboral": rng.choice(["Empleado", "Empleado", "Independiente"]),
        "salario_cliente": salary,
        "total_otros_prestamos": int(rng.lognormvariate(14.0, 1.0)),
        "cuota_pactada": max(50_000, capital // term),
        "puntaje_datacredito": rng.randint(450, 900),
        "cant_creditosvigentes": rng.randint(0, 9),
        "huella_consulta": rng.randint(0, 15),
        "tendencia_ingresos": rng.choice(["Decreciente", "Estable", "Creciente"]),
    }
    if rng.random() < 0.15:
        del record["salario_cliente"]
    if rng.random() < 0.15:
        del record["tendencia_ingresos"]
    return record


class ScoringUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.rng = random.Random()
        self.n = 0

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {bearer()}"}

    def _score(self, name: str, size: int):
        self.n += size
        records = [application(self.rng, self.n + i) for i in range(size)]
        with self.client.post("/predict/batch", json={"records": records}, headers=self._headers(),
                              name=name, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            body = r.json()
            if len(body["predictions"]) != size or not all(0 <= p["probability_default"] <= 1 for p in body["predictions"]):
                r.failure("malformed predictions")

    @task(8)
    def one_application(self):
        self._score("POST /predict/batch (1 application)", 1)

    @task(2)
    def small_batch(self):
        self._score("POST /predict/batch (20 applications)", 20)

    @task(1)
    def health(self):
        self.client.get("/health", headers=self._headers(), name="GET /health")
