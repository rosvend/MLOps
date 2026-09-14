"""Band cut-points measured once on the 10 763-loan dataset in notebooks/eda.ipynb.

Frozen on purpose: recomputing them per batch would make one application score
differently depending on who else was scored alongside it.
"""

PUNTAJE_BUREAU_TERCILES = (770, 813)
HUELLA_BANDAS = (3, 6)
BRECHA_INGRESO_CUARTILES = (1.081, 1.807, 3.594)
CAPITAL_CUARTIL_ALTO = 3_084_840
PLAZO_LARGO_MESES = 13
EDAD_JOVEN = 36
