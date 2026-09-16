"""The entity every feature view joins on."""

import sys
from pathlib import Path

# feast apply imports this file with its own working directory, so the project root
# has to be on the path before src is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feast import Entity  # noqa: E402
from feast.value_type import ValueType  # noqa: E402

from src.features.spec import default_spec  # noqa: E402

_SPEC = default_spec()

cliente = Entity(
    name=_SPEC.entity_name,
    join_keys=[_SPEC.entity_key],
    value_type=ValueType.STRING,
    description=(
        "Surrogate key for one credit application (CLI-0000001...). It identifies a loan, "
        "not a borrower: none of the source columns was unique per row, so repeat customers "
        "cannot be detected until the business exports the real client identifier."
    ),
)
