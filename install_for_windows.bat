@echo off





:install-conda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" .\miniconda.exe /S
del miniconda.exe


:copy-files-to-cuda-path




:: Fix bug in PyTorch code!
:: See https://github.com/pytorch/pytorch/issues/138211 and https://github.com/woct0rdho/triton-windows/issues/10
:: In `torch/_inductor/codecache.py` do
:: REPLACE 
  tmp_path.rename
:: WITH
    try:
        tmp_path.rename(target=path)
    except FileExistsError as e_file_exist:
        if not _IS_WINDOWS:
            raise