param([string]$OutputFolder = ".\results")
Get-Content (Join-Path $OutputFolder "logs\STAGE_STATUS.md") -Wait
