# Prediction log for the API: BigQuery instead of an ephemeral SQLite file on Cloud Run.

resource "google_project_service" "bigquery" {
  service            = "bigquery.googleapis.com"
  disable_on_destroy = false
}

resource "google_bigquery_dataset" "monitoring" {
  dataset_id = "mlops_monitoring"
  location   = var.region

  depends_on = [google_project_service.bigquery]
}

resource "google_bigquery_table" "predictions" {
  dataset_id          = google_bigquery_dataset.monitoring.dataset_id
  table_id            = "predictions"
  deletion_protection = true

  schema = jsonencode([
    { name = "scored_at", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "payload", type = "JSON", mode = "REQUIRED" },
  ])
}

# The runtime writes and reads rows in this dataset only.
resource "google_bigquery_dataset_iam_member" "api_log_editor" {
  dataset_id = google_bigquery_dataset.monitoring.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_service_account.api_runtime.member
}

# Needed to run the read query (a job); grants no data access by itself.
resource "google_project_iam_member" "api_bq_jobs" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = google_service_account.api_runtime.member
}
