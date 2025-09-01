# === CHECK ADMIN PRIVILEGES ===
$IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host "Error: This script must be run as Administrator."
    exit 1
}

# === CONFIGURATION ===
$NCU = "ncu"   # Make sure Nsight Compute is in the PATH
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Project directory: parent of the folder containing the script
$ProjectDir = Split-Path (Split-Path $ScriptDir -Parent) -Parent

# Directories
$BuildDir   = Join-Path $ScriptDir "build"
$ProfileDir = Join-Path $ScriptDir "profiling"
$DataDir    = Join-Path $ProjectDir "data"

# Create profiling folder if it doesn't exist
if (-Not (Test-Path $ProfileDir)) {
    New-Item -ItemType Directory -Path $ProfileDir | Out-Null
}

# === CHECK DIRECTORIES ===
if (-Not (Test-Path $BuildDir)) {
    Write-Host "Error: Build directory not found: $BuildDir"
    exit 1
}

if (-Not (Test-Path $DataDir)) {
    Write-Host "Error: Data directory not found: $DataDir"
    exit 1
}

# Create profiling directory if it doesn’t exist
if (-Not (Test-Path $ProfileDir)) {
    New-Item -ItemType Directory -Path $ProfileDir | Out-Null
}

# === FIND FILES ===
$ExeFiles = Get-ChildItem -Path $BuildDir -Filter *.exe
if ($ExeFiles.Count -eq 0) {
    Write-Host "No executables found in $BuildDir"
    exit 1
}

$InputFiles = Get-ChildItem -Path $DataDir -Filter *.txt
if ($InputFiles.Count -eq 0) {
    Write-Host "No input .txt files found in $DataDir"
    exit 1
}

# === RUN PROFILING ===
foreach ($exe in $ExeFiles) {
    $ExeName = [System.IO.Path]::GetFileNameWithoutExtension($exe.Name)

    if ($ExeName -notmatch 'v3') {
        Write-Host "Skipping $($exe.Name) to perform only NOE profile"
        continue
    }

    foreach ($InputFile in $InputFiles) {
        $InputName = [System.IO.Path]::GetFileNameWithoutExtension($InputFile.Name)

        # Extract number of complex numbers from filename, e.g., numbers_131072
        if ($InputName -match 'numbers_(\d+)') {
            $NumComplex = [int]$Matches[1]
        } else {
            Write-Host "Warning: could not parse number of complex numbers from $($InputFile.Name), skipping"
            continue
        }

        # Do only file with 4194304 complex numbers
        if ($NumComplex -ne 4194304) {
            Write-Host "Skipping $($InputFile.Name) to perform only 4194304 complex numbers"
            continue
        }

        $ProfileOut = Join-Path $ProfileDir "${ExeName}_${NumComplex}_PROFILE"
        
        Write-Host "Profiling $($exe.FullName) with input $($InputFile.FullName) -> $ProfileOut.ncu-rep"
        
        $ThreadsPerBlock = 256  # example value
        $cmd = "$NCU --set full -f -o `"$ProfileOut`" `"$($exe.FullName)`" $ThreadsPerBlock `"$($InputFile.FullName)`""
        Invoke-Expression $cmd
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Profiling successful: $ProfileOut.ncu-rep"
        } else {
            Write-Host "Profiling failed for: $($exe.FullName) with input $($InputFile.Name)"
        }
    }
}

Write-Host "All profiling runs finished. Results are in $ProfileDir"