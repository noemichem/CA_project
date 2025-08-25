# === CONFIGURATION ===
$COMPILER = "g++"  # Assicurati che g++ sia nel PATH (MinGW o MSYS2)
$CXXFLAGS = "-O3 -march=native -fopenmp -g"

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
    # Nessun argomento: prendi tutti i file .cpp da source
    $CppFiles = Get-ChildItem -Path $SourceDir -Filter *.cpp
} else {
    # Argomenti passati: usa percorsi assoluti o relativi dalla directory corrente
    $CppFiles = @()
    foreach ($arg in $args) {
        $fullPath = if (Test-Path $arg) { Resolve-Path $arg } else { Join-Path (Get-Location) $arg }
        if (Test-Path $fullPath) {
            $CppFiles += Get-Item $fullPath
        } else {
            Write-Host "File not found: $arg"
            exit 1
        }
    }
}

if ($CppFiles.Count -eq 0) {
    Write-Host "No .cpp files found"
    exit 1
}

# === COMPILE EACH CPP FILE INDIVIDUALLY INTO BUILD DIRECTORY ===
foreach ($file in $CppFiles) {
    $ExecName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $OutputPath = Join-Path $BuildDir $ExecName
    
    Write-Host "Compiling $($file.FullName) -> $OutputPath"
    
    $cmd = "$COMPILER $CXXFLAGS -o `"$OutputPath`" `"$($file.FullName)`""
    Invoke-Expression $cmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation successful: $OutputPath"
    } else {
        Write-Host "Compilation failed for: $($file.FullName)"
        exit 1
    }
}

Write-Host "All compilations finished successfully. Executables are in $BuildDir"