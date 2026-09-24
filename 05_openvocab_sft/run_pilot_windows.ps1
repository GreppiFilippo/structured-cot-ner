# Pilot on a local 8 GB GPU (Windows, PowerShell): trains Qwen3-1.7B with
# LoRA on 1,000 open-vocabulary examples, then evaluates the base model and
# the pilot adapter on the first 100 rows of ncbi with the baseline prompt.
# Purpose: validate the whole Qwen pipeline end to end before using the
# cluster - the numbers are NOT thesis results (small model, 100 rows).
#
# Run from the repository root, in a normal (interactive) terminal:
#     powershell -ExecutionPolicy Bypass -File 05_openvocab_sft\run_pilot_windows.ps1
# Log: logs\pilot.log

param(
    [string]$Python = "D:\tesi\venv\Scripts\python.exe",
    [string]$ModelId = "Qwen/Qwen3-1.7B",
    [int]$TrainSamples = 1000,
    [int]$EvalRows = 100,
    [string]$EvalDataset = "ncbi"
)

# "Continue": in Windows PowerShell 5 every stderr line of python (warnings,
# progress bars) becomes an error record, which "Stop" would treat as fatal
$ErrorActionPreference = "Continue"
$env:HF_HUB_CACHE = "D:\hf_cache\hub"          # models on D:, the HF login stays where it is
$env:HF_DATASETS_CACHE = "D:\hf_cache\datasets"
$env:PYTHONUTF8 = "1"                           # JSON files are UTF-8; Windows defaults to cp1252
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$root = (Get-Location).Path
New-Item -ItemType Directory -Force "$root\logs" | Out-Null
$log = "$root\logs\pilot.log"
Set-Content -Path $log -Value "pilot started $(Get-Date)" -Encoding utf8

# Start-Transcript does not capture native programs' output in Windows
# PowerShell 5, so every step's output (stdout + stderr) is teed explicitly
# (Add-Content rather than Tee: in PowerShell 5 Tee writes UTF-16)
function Say($line) { Write-Host $line; Add-Content -Path $log -Value $line -Encoding utf8 }
function Step($name, [scriptblock]$block) {
    Say "`n===== $name ($(Get-Date -Format HH:mm:ss)) ====="
    & $block 2>&1 | ForEach-Object { Say "$_" }
    if ($LASTEXITCODE -ne 0) { Say "FAILED: $name (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}

Step "CUDA check" {
    & $Python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
}
Step "Train pilot adapter" {
    Push-Location "$root\05_openvocab_sft"
    & $Python train_qwen.py --data-path data\openvocab_train.jsonl --model-id $ModelId `
        --n-train-samples $TrainSamples --num-epochs 1 --eval-steps 50 `
        --output-dir work\pilot --skip-push
    Pop-Location
}
$slug = ($ModelId -split "/")[-1].ToLower()
Step "Evaluate base $ModelId" {
    Push-Location "$root\04_evaluation"
    & $Python evaluate.py --model-type causal --model-id $ModelId --run-name "${slug}_base_pilot" `
        --dataset-name $EvalDataset --limit $EvalRows --batch-size 8 --max-new-tokens 4096 --do-sample `
        --output-dir outputs_pilot
    Pop-Location
}
Step "Evaluate pilot adapter" {
    Push-Location "$root\04_evaluation"
    & $Python evaluate.py --model-type causal --model-id $ModelId --adapter "$root\05_openvocab_sft\work\pilot\adapter" `
        --run-name "${slug}_sft_pilot" --dataset-name $EvalDataset --limit $EvalRows --batch-size 8 `
        --max-new-tokens 4096 --do-sample --output-dir outputs_pilot
    Pop-Location
}
Say "`n===== DONE ($(Get-Date -Format HH:mm:ss)) ====="
