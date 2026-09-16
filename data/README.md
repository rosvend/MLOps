# Data

Here goes the data for the project, such as training datasets. When using DVC, it should point at this folder

## `cliente_id`

A surrogate key (`CLI-0000001`…) added so the feature store has an entity to join on. It is one
ID per row, so it identifies a *loan*, not a borrower — repeat customers cannot be detected with
it and none of the original 23 columns was unique per row. Replace it with the real client
identifier when the business re-exports, and borrower-history features become possible.
