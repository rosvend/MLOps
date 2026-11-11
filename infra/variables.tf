variable "project_id" { type = string }
variable "billing_account" { type = string }
variable "bucket_name" { type = string }
variable "region" {
  type    = string
  default = "us-central1"
}

variable "budget_usd" {
  type    = number
  default = 40
}

variable "apis" {
  type = list(string)
  default = [
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "billingbudgets.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com"
  ]
}