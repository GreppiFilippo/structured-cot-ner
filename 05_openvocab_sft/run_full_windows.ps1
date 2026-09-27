# Full local run (Windows, 8 GB GPU): LoRA SFT of a Qwen3 model in bf16 on a
# whole open-vocabulary dataset, then evaluation of the base model and of the
# fine-tuned adapter on all 9 BioNER benchmarks (the same fixed subset and
# metric used for the MedGemma runs), with the baseline's closed-schema prompt.
#
# Run from the repository root:
#     powershell -ExecutionPolicy Bypass -File 05_openvocab_sft\run_full_windows.ps1
# Log: logs\full_<Tag>.log. Safe to re-run after an interruption: training
# resumes from its last checkpoint, each evaluation from its last generated
# batch, and finished evaluations are skipped.

param(
    [string]$Python = "C:\tesi\venv\Scripts\python.exe",
    [string]$HfCache = "C:\hf_cache",
    [string]$ModelId = "Qwen/Qwen3-1.7B",
    [string]$DataPath = "data\openvocab_train_v3_qwen.jsonl",
    [string]$Tag = "v3",
    [double]$Epochs = 2,
    [int]$MaxNewTokens = 4096,
    [int]$BatchSize = 8,
    [string]$OutputDir = "outputs_qwen",
    [string[]]$Datasets = @("ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM", "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored"),
    # train only (e.g. when the evaluation runs elsewhere, like Modal with vLLM)
    [switch]$TrainOnly
)

# see run_pilot_windows.ps1 for why each of these is set
$ErrorActionPreference = "Continue"
$env:HF_HUB_CACHE = "$HfCache\hub"
$env:HF_DATASETS_CACHE = "$HfCache\datasets"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$root = (Get-Location).Path
New-Item -ItemType Directory -Force "$root\logs" | Out-Null
$log = "$root\logs\full_$Tag.log"
function Say($line) { Write-Host $line; Add-Content -Path $log -Value $line -Encoding utf8 }
function Step($name, [scriptblock]$block) {
    Say "`n===== $name ($(Get-Date -Format "yyyy-MM-dd HH:mm:ss")) ====="
    & $block 2>&1 | ForEach-Object { Say "$_" }
    if ($LASTEXITCODE -ne 0) { Say "FAILED: $name (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
Say "full run $Tag started $(Get-Date) (model: $ModelId, data: $DataPath, epochs: $Epochs)"

$slug = ($ModelId -split "/")[-1].ToLower()
$adapter = "$root\05_openvocab_sft\work\full_${slug}_$Tag\adapter"

Step "CUDA check" {
    & $Python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
}
Step "Train $ModelId on $DataPath" {
    Push-Location "$root\05_openvocab_sft"
    & $Python train_qwen.py --data-path $DataPath --model-id $ModelId --num-epochs $Epochs `
        --output-dir "work\full_${slug}_$Tag" --skip-push
    Pop-Location
}
if ($TrainOnly) { Say "`n===== DONE (train only) ($(Get-Date -Format "yyyy-MM-dd HH:mm:ss")) ====="; exit 0 }
foreach ($ds in $Datasets) {
    Step "Evaluate base $ModelId on $ds" {
        Push-Location "$root\04_evaluation"
        & $Python evaluate.py --model-type causal --model-id $ModelId --run-name "${slug}_base" `
            --dataset-name $ds --batch-size $BatchSize --max-new-tokens $MaxNewTokens --do-sample `
            --output-dir $OutputDir
        Pop-Location
    }
    Step "Evaluate $Tag adapter on $ds" {
        Push-Location "$root\04_evaluation"
        & $Python evaluate.py --model-type causal --model-id $ModelId --adapter $adapter `
            --run-name "${slug}_sft_$Tag" --dataset-name $ds --batch-size $BatchSize `
            --max-new-tokens $MaxNewTokens --do-sample --output-dir $OutputDir
        Pop-Location
    }
}
Say "`n===== DONE ($(Get-Date -Format "yyyy-MM-dd HH:mm:ss")) ====="
