@echo off
echo Starting sequential training tasks to avoid crashes...

echo [1/2] Training Ablation P2 Model (via generate_ablation_v2.py)...
python generate_ablation_v2.py
if %errorlevel% neq 0 (
    echo Error in generate_ablation_v2.py
    pause
    exit /b %errorlevel%
)

echo [2/2] Training Tiny P2P3 Model (via train_tiny_p2p3.py)...
python train_tiny_p2p3.py
if %errorlevel% neq 0 (
    echo Error in train_tiny_p2p3.py
    pause
    exit /b %errorlevel%
)

echo All training complete. Syncing to Git...
git add .
git commit -m "Auto-commit: Sequential training completed for P2 and Tiny P2P3 models"
git push

echo Locking screen...
rundll32.exe user32.dll,LockWorkStation
