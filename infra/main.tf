terraform {
  required_version = ">=1.6"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}

data "google_project" "this" {}

resource "google_project_service" "apis" {
  for_each           = toset(var.apis)
  service            = each.value
  disable_on_destroy = false
}

resource "google_storage_bucket" "data" {
  name                        = var.bucket_name
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true # overwriting the champion keeps the previous one
  }

  depends_on = [google_project_service.apis]
}

resource "google_billing_budget" "budget" {
  billing_account = var.billing_account
  display_name    = "mlops-final-budget"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_usd)
    }
  }

  threshold_rules { threshold_percent = 0.5 }
  threshold_rules { threshold_percent = 0.9 }
}

resource "google_artifact_registry_repository" "mlops" {
  repository_id = "mlops"
  location      = var.region
  format        = "DOCKER"
  description   = "Training images for the Vertex AI pipeline"
  depends_on    = [google_project_service.apis]
}

# Vertex's pipeline service agent must be able to pull the training image.
resource "google_artifact_registry_repository_iam_member" "vertex_pull" {
  repository = google_artifact_registry_repository.mlops.name
  location   = var.region
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
}
