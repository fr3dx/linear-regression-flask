# run_model_and_app.ps1

Write-Host "Running model training script..."
python .\model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Model training failed. Exiting."
    exit 1
}

Write-Host "Starting Flask app..."
python .\app.py
