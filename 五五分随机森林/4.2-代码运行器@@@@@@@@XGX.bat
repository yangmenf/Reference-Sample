@echo off
chcp 65001
cd %~dp0

REM 激活指定的 Anaconda 环境
call D:\Anaconda\envs\AICPU\python.exe 4.2-XGX.py

REM 暂停，以便查看输出
pause

REM 可选：如果您希望在脚本执行后禁用环境
call conda deactivate