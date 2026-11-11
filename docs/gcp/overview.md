# Target architecture on Google Cloud

Everything lives in one GCP project, `mlops-roysandoval`, in `us-central1`. Keeping every
resource in one region avoids cross-region errors and transfer costs.

```mermaid
flowchart LR
    you["You"] -->|git push| gh["GitHub<br/>repo + Actions"]

    subgraph gcp["Google Cloud project - us-central1"]
        gcs[("Cloud Storage<br/>raw data, CSVs, models, metrics")]
        vtx["Vertex AI<br/>AutoML + custom pipeline"]
        cb["Cloud Build<br/>builds images"]
        ar["Artifact Registry<br/>Docker images"]
        run["Cloud Run<br/>FastAPI scoring API"]
        bq[("BigQuery<br/>prediction log")]
    end

    users["Users / apps"] -->|HTTPS| run
    gcs -->|train + evaluate| vtx
    cb -->|push image| ar
    ar -->|training image| vtx
    gh -.->|docker push| ar
    ar -.->|deploy| run
    run -.->|logs| bq

    classDef built fill:#e7f6ec,stroke:#2f9e5b
    classDef planned fill:#fff6e0,stroke:#d9a400,stroke-dasharray:4 3
    class gcs,vtx,cb,ar built
    class run,bq,gh planned
```

Solid arrows are built and working. Dashed arrows are planned (Phase 1: CI/CD and deploy).

## What each service does

| Service | Role here | Why this one |
| --- | --- | --- |
| Cloud Storage | Holds the raw loans, the time-split training file, model files and metrics | Cheap, durable, and every other service reads it |
| Vertex AI | Runs AutoML and the custom training pipeline | Managed training without owning machines |
| Cloud Build | Builds the training image that Vertex runs | No local Docker needed |
| Artifact Registry | Stores the training and serving images | One private place that Vertex and Cloud Run both pull from. The serving image is built on the GitHub runner and pushed here |
| Cloud Run | Serves the FastAPI scoring API | Runs the existing container as-is and scales to zero when idle |
| BigQuery | Keeps the prediction log for drift monitoring | Serverless and append-friendly; no database to run |

OpenTofu (`infra/`) creates the project setup: APIs, the bucket, the budget alert, the registry
and IAM. See [Foundations](foundations.md).

## How a model reaches production

```mermaid
flowchart TB
    a["1. Pipeline trains a candidate<br/>Vertex AI"] --> b["2. Gate compares candidate<br/>to champion on fresh loans"]
    b -->|better| c["3. Human approves<br/>GitHub Environment"]
    b -->|not better| x["Stop: keep champion"]
    c --> d["4. Champion file updated<br/>Cloud Storage"]
    d --> e["5. Deploy builds image,<br/>sends 10 % of traffic"]
    e -->|healthy| f["6. 100 % of traffic"]
    e -->|errors| g["Roll back"]
```

The details are on the [CI/CD page](cicd.md).
