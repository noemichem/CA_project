# === CONFIGURATION ===
$NVCC = "nvcc"  # Assicurati che nvcc sia nel PATH
$NVCCFLAGS = '-O3 -Xptxas -v -use_fast_math -lineinfo "-gencode=arch=compute_86,code=sm_86"'

# === GET SCRIPT DIRECTORY ===
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# === SOURCE AND BUILD DIRECTORIES ===
$SourceDir = Join-Path $ScriptDir "source"
$BuildDir = Join-Path $ScriptDir "build"

# Create build directory if it doesn't exist
if (-Not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# === DETERMINE FILES TO COMPILE ===
if ($args.Count -eq 0) {
    # Nessun argomento: prendi tutti i file .cu da source
    $CuFiles = Get-ChildItem -Path $SourceDir -Filter *.cu
} else {
    # Argomenti passati: usa percorsi assoluti o relativi dalla directory corrente
    $CuFiles = @()
    foreach ($arg in $args) {
        $fullPath = if (Test-Path $arg) { Resolve-Path $arg } else { Join-Path (Get-Location) $arg }
        if (Test-Path $fullPath) {
            $CuFiles += Get-Item $fullPath
        } else {
            Write-Host "File not found: $arg"
            exit 1
        }
    }
}

if ($CuFiles.Count -eq 0) {
    Write-Host "No .cu files found"
    exit 1
}

# === COMPILE EACH CU FILE INDIVIDUALLY INTO BUILD DIRECTORY ===
foreach ($file in $CuFiles) {
    $ExecName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $OutputPath = Join-Path $BuildDir "$ExecName.exe"
    
    Write-Host "Compiling $($file.FullName) -> $OutputPath"

    # Decide se linkare librerie speciali
    if ($ExecName.StartsWith("CUBLAS")) {
        $cmd = "$NVCC $NVCCFLAGS `"$($file.FullName)`" -o `"$OutputPath`" -lcublas"
    } elseif ($ExecName.Contains("cuFFT")) {
        $cmd = "$NVCC $NVCCFLAGS `"$($file.FullName)`" -o `"$OutputPath`" -lcufft"
    } else {
        $cmd = "$NVCC $NVCCFLAGS `"$($file.FullName)`" -o `"$OutputPath`""
    }

    Invoke-Expression $cmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation successful: $OutputPath"
    } else {
        Write-Host "Compilation failed for: $($file.FullName)"
        exit 1
    }
}

Write-Host "All CUDA compilations finished successfully. Executables are in $BuildDir"