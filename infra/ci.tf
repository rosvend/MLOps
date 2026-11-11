resource "google_service_account" "github_ci" {
  account_id   = "github-ci"
  display_name = "Github Actions CI/CD"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "Github Actions"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }

  attribute_condition = "assertion.repository == 'rosvend/MLOps'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_can_impersonate" {
  service_account_id = google_service_account.github_ci.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/rosvend/MLOps"
}

# --- What the CI identity may do: only what the deploy workflow needs ---

# The identity the scoring API runs as. It starts with no permissions of its own.
resource "google_service_account" "api_runtime" {
  account_id   = "scoring-api"
  display_name = "Scoring API (Cloud Run runtime)"
}

# Push the serving image to the Docker repository.
resource "google_artifact_registry_repository_iam_member" "ci_push" {
  repository = google_artifact_registry_repository.mlops.name
  location   = google_artifact_registry_repository.mlops.location
  role       = "roles/artifactregistry.writer"
  member     = google_service_account.github_ci.member
}

# Deploy revisions and shift traffic. Developer, not admin: it cannot change IAM policies.
resource "google_project_iam_member" "ci_run_deploy" {
  project = data.google_project.this.project_id
  role    = "roles/run.developer"
  member  = google_service_account.github_ci.member
}

# Deploying a service that runs as scoring-api requires permission to act as it.
resource "google_service_account_iam_member" "ci_acts_as_runtime" {
  service_account_id = google_service_account.api_runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.github_ci.member
}

# Read the raw loans (tests) and the champion model (image build) from the bucket.
resource "google_storage_bucket_iam_member" "ci_read_data" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = google_service_account.github_ci.member
}

# Lets API calls made by the CI identity be billed to this project.
resource "google_project_iam_member" "ci_use_apis" {
  project = data.google_project.this.project_id
  role    = "roles/serviceusage.serviceUsageConsumer"
  member  = google_service_account.github_ci.member
}

# --- Workflows that train and deploy: what CI needs beyond building the image ---

# A dedicated identity for Vertex pipeline runs, instead of the broad default compute account.
resource "google_service_account" "vertex_pipeline" {
  account_id   = "vertex-pipeline"
  display_name = "Vertex AI pipeline runs"
}

resource "google_project_iam_member" "pipeline_vertex" {
  project = data.google_project.this.project_id
  role    = "roles/aiplatform.user"
  member  = google_service_account.vertex_pipeline.member
}

resource "google_storage_bucket_iam_member" "pipeline_data" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectUser"
  member = google_service_account.vertex_pipeline.member
}

resource "google_artifact_registry_repository_iam_member" "pipeline_pull" {
  repository = google_artifact_registry_repository.mlops.name
  location   = google_artifact_registry_repository.mlops.location
  role       = "roles/artifactregistry.reader"
  member     = google_service_account.vertex_pipeline.member
}

resource "google_project_iam_member" "pipeline_logs" {
  project = data.google_project.this.project_id
  role    = "roles/logging.logWriter"
  member  = google_service_account.vertex_pipeline.member
}

# Vertex's service agent must be allowed to run jobs as the pipeline account.
resource "google_service_account_iam_member" "vertex_agent_acts_as_pipeline" {
  service_account_id = google_service_account.vertex_pipeline.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
}

# GitHub Actions: submit pipelines and register models.
resource "google_project_iam_member" "ci_vertex" {
  project = data.google_project.this.project_id
  role    = "roles/aiplatform.user"
  member  = google_service_account.github_ci.member
}

# GitHub Actions: write candidate models and move the champion in the bucket.
resource "google_storage_bucket_iam_member" "ci_write_data" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectUser"
  member = google_service_account.github_ci.member
}

resource "google_service_account_iam_member" "ci_acts_as_pipeline" {
  service_account_id = google_service_account.vertex_pipeline.name
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.github_ci.member
}

# GitHub Actions: mint an ID token as itself, to smoke-test the private Cloud Run service.
resource "google_service_account_iam_member" "ci_mints_own_token" {
  service_account_id = google_service_account.github_ci.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = google_service_account.github_ci.member
}

resource "google_cloud_run_v2_service_iam_member" "ci_invokes_api" {
  project  = data.google_project.this.project_id
  location = var.region
  name     = "scoring-api"
  role     = "roles/run.invoker"
  member   = google_service_account.github_ci.member
}
