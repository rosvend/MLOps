from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    """Read-only access to the credit portfolio, whatever storage holds it."""

    def read(self) -> pd.DataFrame: ...
