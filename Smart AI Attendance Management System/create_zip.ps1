param(
    [string]$ArchiveName = "smart-attendance-system.zip"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ArchivePath = Join-Path -Path $ProjectRoot -ChildPath $ArchiveName
$TempArchivePath = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath $ArchiveName

if (Test-Path -LiteralPath $TempArchivePath) {
    Remove-Item -LiteralPath $TempArchivePath -Force
}

if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

$ExcludedNames = @(
    ".venv",
    "__pycache__",
    ".pytest_cache",
    $ArchiveName
)

$Items = Get-ChildItem -LiteralPath $ProjectRoot -Force |
    Where-Object { $ExcludedNames -notcontains $_.Name }

Compress-Archive -LiteralPath $Items.FullName -DestinationPath $TempArchivePath -Force
Move-Item -LiteralPath $TempArchivePath -Destination $ArchivePath -Force

Write-Host "Created archive: $ArchivePath"
