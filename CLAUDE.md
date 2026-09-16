# CLAUDE.md

## Overview 

This project builds an MLOps pipeline end-to-end. It goes all the way from ingestion and EDA, to deployment and monitoring. 

## Code Style

Modular, loosely coupled components; KISS and YAGNI by default; SOLID reserved for complex business logic, favouring single responsibility and injected
dependencies. Do not overcomment the code, use single line comments.

## Tools

- **context7 CLI** — current docs for scikit-learn, FastAPI, MLflow, DVC, Pydantic and Hydra. Prefer it over
  recalling an API; these move faster than training data. Use 'ctx7' CLI. 


## Constraints

Make sure there is no data leakage anywhere. Enfore preprocessing pipelines should be isolated so any operations are fit in the training data and transformed in the test data.  