variable "aws_profile" {
  description = "AWS CLI profile used to run tofu (set it in terraform.tfvars or TF_VAR_aws_profile)."
  type        = string
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  description = "Prefix of every resource this stack creates."
  type        = string
  default     = "mlops-loan"
}

variable "github_repo" {
  description = "owner/name of the only repository the deploy role trusts."
  type        = string
  default     = "rosvend/MLOps"
}

variable "github_branch" {
  type    = string
  default = "main"
}

variable "image_tag" {
  description = "Image tag the function is first created from; CI moves it afterwards."
  type        = string
  default     = "bootstrap"
}

variable "memory_mb" {
  type    = number
  default = 1024
}

variable "timeout_s" {
  type    = number
  default = 30
}
