data "aws_caller_identity" "current" {}

locals {
  function_name = "${var.name}-scoring"
  # Bucket names are global; a short hash of the account keeps ours unique without printing the id.
  bucket_name = "${var.name}-models-${substr(sha1(data.aws_caller_identity.current.account_id), 0, 8)}"
  aliases     = toset(["live", "candidate"])
}

# --- Image registry ---------------------------------------------------------------------------

resource "aws_ecr_repository" "api" {
  name                 = "${var.name}-api"
  force_delete         = true
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

# --- Champion model store (versioned, like the Cloud Storage bucket) --------------------------

resource "aws_s3_bucket" "models" {
  bucket        = local.bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- The function -----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 7
}

data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "runtime" {
  name               = "${var.name}-runtime"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "runtime" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.api.arn}:*"]
  }
}

resource "aws_iam_role_policy" "runtime" {
  name   = "logs"
  role   = aws_iam_role.runtime.id
  policy = data.aws_iam_policy_document.runtime.json
}

resource "aws_lambda_function" "api" {
  function_name = local.function_name
  role          = aws_iam_role.runtime.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
  memory_size   = var.memory_mb
  timeout       = var.timeout_s
  publish       = true

  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy.runtime]

  # After the first apply the pipeline owns the code; tofu must not roll it back.
  lifecycle {
    ignore_changes = [image_uri]
  }
}

# `live` serves users, `candidate` is the new version under test; the pipeline moves both.
resource "aws_lambda_alias" "stage" {
  for_each         = local.aliases
  name             = each.key
  function_name    = aws_lambda_function.api.function_name
  function_version = aws_lambda_function.api.version

  lifecycle {
    ignore_changes = [function_version, routing_config]
  }
}

# AWS_IAM: callers must sign requests (SigV4); nothing is public.
resource "aws_lambda_function_url" "stage" {
  for_each           = local.aliases
  function_name      = aws_lambda_function.api.function_name
  qualifier          = aws_lambda_alias.stage[each.key].name
  authorization_type = "AWS_IAM"
}

# --- Keyless CI: GitHub OIDC -> a role for one repo and one branch ----------------------------

# The account already has one provider for GitHub (shared); we only read it.
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/${var.github_branch}"]
    }
  }
}

resource "aws_iam_role" "deploy" {
  name               = "${var.name}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_trust.json
}

data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # the API does not accept a resource
  }
  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer", "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.api.arn]
  }
  statement {
    sid = "FunctionRollout"
    actions = [
      "lambda:GetAlias", "lambda:GetFunction", "lambda:GetFunctionConfiguration",
      "lambda:GetFunctionUrlConfig", "lambda:PublishVersion", "lambda:UpdateAlias",
      "lambda:UpdateFunctionCode", "lambda:InvokeFunctionUrl", "lambda:InvokeFunction",
    ]
    resources = [aws_lambda_function.api.arn, "${aws_lambda_function.api.arn}:*"]
  }
  statement {
    sid       = "ReadChampion"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "deploy"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}
