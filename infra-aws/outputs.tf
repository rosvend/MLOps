output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "function_name" {
  value = aws_lambda_function.api.function_name
}

output "live_url" {
  value = aws_lambda_function_url.stage["live"].function_url
}

output "candidate_url" {
  value = aws_lambda_function_url.stage["candidate"].function_url
}

output "models_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "deploy_role_arn" {
  value = aws_iam_role.deploy.arn
}
