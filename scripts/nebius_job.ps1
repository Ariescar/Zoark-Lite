# Runs scripts/cloud_job.sh as a Nebius AI Cloud Serverless job on one GPU.
# Needs the Nebius CLI, logged in (nebius profile create). First run shows the job without creating it:
#   powershell -ExecutionPolicy Bypass -File scripts\nebius_job.ps1
# When the printed job looks right, create it for real:
#   powershell -ExecutionPolicy Bypass -File scripts\nebius_job.ps1 -Run
param([switch]$Run, [int]$N = 64, [string]$Platform = "gpu-l40s-a", [string]$Preset = "1gpu-8vcpu-32gb")
$name = "zoark-lite-" + (Get-Date -Format "yyyyMMdd-HHmm")
$inner = "git clone --depth 1 https://github.com/Ariescar/Zoark-Lite /app && N=$N bash /app/scripts/cloud_job.sh"
$a = @("ai", "job", "create", "--name", $name, "--image", "python:3.12",
       "--container-command", "bash", "--args", "-c '$inner'",
       "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
       "--platform", $Platform, "--preset", $Preset, "--timeout", "1h")
if (-not $Run) { $a += "--dry-run" }
& nebius @a
if ($Run) {
    Write-Host "`nJob $name created. Follow it with:" -ForegroundColor Green
    Write-Host "  nebius ai job get-by-name --name $name"
    Write-Host "  nebius ai job logs <job id from the line above>"
}
