@echo off
echo ========================================================
echo   Auto-Training & Sync Script for YOLOv12 P2 Improved
echo ========================================================

echo [1/4] Starting Training (YOLOv12-P2-Improved)...
echo This may take 2-3 hours. Please do not close this window.
python train_final_p2.py
if %errorlevel% neq 0 (
    echo Error during training!
    pause
    exit /b %errorlevel%
)

echo [2/4] Updating Comparison Tables...
python compare_sota.py

echo [3/4] Committing Results to Git...
git add .
git commit -m "Auto-commit: Finished P2 Head training and updated comparisons"

echo [4/4] Pushing to Remote...
git push
if %errorlevel% neq 0 (
    echo Push failed (maybe authentication needed). 
    echo Don't worry, results are saved locally and committed.
    echo You can push manually later.
)

echo ========================================================
echo   All Done! You can now close this window.
echo ========================================================
pause
