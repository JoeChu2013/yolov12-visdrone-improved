@echo off
echo ========================================================
echo   Auto-Training & Sync Script for LR Comparison
echo ========================================================

echo [1/3] Starting LR Experiments (0.01, 0.001, 0.0001)...
python train_lr_comparison.py

echo [2/3] Committing Results to Git...
git add .
git commit -m "Add Learning Rate convergence comparison results"

echo [3/3] Pushing to Remote...
git push

echo ========================================================
echo   All Done!
echo ========================================================
pause
