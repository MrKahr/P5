@echo off





:install-conda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" .\miniconda.exe /S
del miniconda.exe


:install-msvc-buildtools
curl https://github.com/Data-Oriented-House/PortableBuildTools/releases/latest/download/PortableBuildTools.exe
start /wait "" .\PortableBuildTools.exe accept_license target=x64 host=x64 path=%~dp0.buildtools
del PortableBuildTools.exe