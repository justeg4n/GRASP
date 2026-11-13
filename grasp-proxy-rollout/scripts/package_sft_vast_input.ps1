[CmdletBinding()]
param(
    [string]$SftArtifactDir = "D:\FPTU\FPTU-SP26-CaptonesProject\hf-artifacts\grasp-sft",
    [string]$OutputZip = "D:\FPTU\FPTU-SP26-CaptonesProject\vast-upload\grasp-sft-vast-input.zip",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$arguments = @(
    (Join-Path $PSScriptRoot "package_sft_vast_input.py"),
    "--repo-dir", $repoRoot,
    "--sft-dir", $SftArtifactDir,
    "--output", $OutputZip
)
if ($Force) {
    $arguments += "--force"
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Input packaging failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
