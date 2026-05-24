@echo off
cd /d E:\MicroLM
echo Started at %DATE% %TIME% > reports\b1_pretrain_status.txt
.\.venv\Scripts\python.exe scripts\train_pretrain.py --config configs\pretrain_full_corpus.json --wandb_mode disabled > reports\b1_pretrain_stdout.log 2> reports\b1_pretrain_stderr.log
set EXITCODE=%ERRORLEVEL%
echo %EXITCODE% > reports\b1_pretrain.exitcode
echo Finished at %DATE% %TIME% with exit code %EXITCODE% >> reports\b1_pretrain_status.txt
exit /b %EXITCODE%
