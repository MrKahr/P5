@echo off
CD /D "%~dp0"

::Runners to simulate function calls
SET call_condainstaller=0
SET call_condaenviron=0
SET call_cudapath=0
SET call_torchpatch=0

:: Conda
SET conda_name=Miniconda3
SET conda_license=https://legal.anaconda.com/policies/en/
SET conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py312_24.9.2-0-Windows-x86_64.exe
SET conda_env_name=.conda
SET conda_env_prefix=./%conda_env_name%
SET conda_env_path=%~dp0%conda_env_name%
SET conda_environ=environment.yml
SET conda_path=%USERPROFILE%\miniconda3\_conda.exe

::Triton Windows
SET triton_windows_name=Triton Windows
SET cudalib_file=cuda.lib
SET cudalib_basepath=.conda\Library\lib
SET actual_cudalib_path=%cudalib_basepath%\%cudalib_file%
SET expected_cudalib_path=%cudalib_basepath%\x64\%cudalib_file%

:: PyTorch
SET torch_name=PyTorch
SET torch_codecache_file=codecache.py
SET torch_codecache_path=".conda\Lib\site-packages\torch\_inductor\%torch_codecache_file%"
SET torch_patched_codecache_path=installer\%torch_codecache_file%

:intro
    echo:
    echo: ______________________________________________________________
    echo:
    echo:  Hello there!
    echo:  This script helps with installing the necessary dependencies
    echo:  and setting everything up.
    echo:  To do this, the following steps will be executed:
    echo:
    echo:  - Download and install '%conda_name%' from
    echo:    %conda_url%
    echo:    This means accepting their license: %conda_license%
    echo:
    echo:  - Create a virtual environment at '%conda_env_prefix%'
    echo:    and install dependencies from %conda_environ%
    echo:
    echo:  - Copy CUDA file '%cudalib_file%' to the location %triton_windows_name% expects
    echo:
    echo:  - Patch bug in %torch_name% file '%torch_codecache_file%'
    echo: ______________________________________________________________
    echo:

:menu
    echo:
    echo: ____Start______________
    echo:  1. Install everything
    echo:
    echo: ____%conda_name%___________
    echo:  c1. Download and install %conda_name%
    echo:  c2. Create %conda_name% virtual environment
    echo:      Expected %conda_name% location: '%conda_path%'
    echo:
    echo: ____%triton_windows_name%______________
    echo:  t1. Copy '%actual_cudalib_path%' to '%expected_cudalib_path%'
    echo:  h1. Explain why this is required.
    echo:
    echo: ____%torch_name%______________
    echo:  p1. Patch bug in file %torch_codecache_path% using file '%torch_patched_codecache_path%'
    echo:  h2. Explain why this is required.
    echo:
    echo: ____Other______________
    echo:  o1. Reinstall environment
    echo:
    set /p menuChoice="Select option: "

    IF %menuChoice%==1 (
        SET call_condainstaller=1
        SET call_condaenviron=1
        SET call_cudapath=1
        SET call_torchpatch=1
        GOTO run
    )
    IF %menuChoice%==c1 (
        SET call_condainstaller=1
        SET call_condaenviron=0
        SET call_cudapath=0
        SET call_torchpatch=0
        GOTO run
    )
    IF %menuChoice%==c2 (
        SET call_condainstaller=0
        SET call_condaenviron=1
        SET call_cudapath=0
        SET call_torchpatch=0
        GOTO run
    )
    IF %menuChoice%==t1 (
        SET call_condainstaller=0
        SET call_condaenviron=0
        SET call_cudapath=1
        SET call_torchpatch=0
        GOTO run
    )
    IF %menuChoice%==p1 (
        SET call_condainstaller=0
        SET call_condaenviron=0
        SET call_cudapath=0
        SET call_torchpatch=1
        GOTO run
    )
    IF %menuChoice%==o1 (
        ECHO:
        ECHO: Deleting '%conda_env_path%'
        DEL /S /Q %conda_env_path%
        SET call_condainstaller=1
        SET call_condaenviron=1
        SET call_cudapath=1
        SET call_torchpatch=1
        GOTO run
    )
    IF %menuChoice%==h1 (
        ECHO:
        ECHO: ==============Explanation==================
        ECHO: %triton_windows_name% has hardcoded its expected CUDA_PATH of a specific file ^("lib/x64/cuda.lib"^)
        ECHO: to the location provided by the CUDA Toolkit GUI installer ^(https://developer.nvidia.com/cuda-downloads^)
        ECHO: However, with %conda_name%, CUDA_PATH of this file is ^("lib/cuda.lib"^)
        ECHO: Thus, we copy the file to the CUDA_PATH Triton Windows expects.
        ECHO: ===========================================
        ECHO:
        ECHO: Press enter to continue
        pause >nul
    )
    IF %menuChoice%==h2 (
        ECHO:
        ECHO: ==============Explanation==================
        ECHO: Windows-specific path-rename bug making our program explode.
        ECHO: See:
        ECHO:   https://github.com/pytorch/pytorch/issues/138211
        ECHO:   https://github.com/woct0rdho/triton-windows/issues/10
        ECHO: The issue is resolved by PyTorch in v2.6 nightly, however, we use 2.5.1 stable.
        ECHO: Thus, we fix the bug manually using the code from the fixed PyTorch version.
        ECHO: NOTE: The patch is line 466-470.
        ECHO: ===========================================
        ECHO:
        ECHO: Press enter to continue
        pause >nul
    )
    GOTO menu


:: Function entry-point
:run


:install-conda
IF %call_condainstaller%==1 (
    ECHO: Downloading %conda_url%
    curl %conda_url% -o miniconda.exe
    ECHO: Installing %conda_name%...
    START /wait "" .\miniconda.exe /S
    DEL miniconda.exe
    ECHO: Done
)

:install-dependencies
IF %call_condaenviron%==1 (
    IF NOT EXIST "%conda_path%" (
        ECHO:
        ECHO: Cannot find path '%conda_path%'.
        ECHO: Please specify %conda_name% install path.
        SET /p conda_path="Enter conda path: "
        GOTO install-dependencies
    ) ELSE (
        ECHO:
        ECHO: Creating environment at '%conda_env_path%'...
        "%conda_path%" env create --prefix %conda_env_prefix% -f %conda_environ%
        ECHO:
        ECHO: Activating environment
        "%conda_path%" activate %conda_env_path%
        ECHO: Done
    )
)

:copy-files-to-cuda-path
:: Triton Windows has hardcoded its expected CUDA_PATH of a specific file ("lib" / "x64" / "cuda.lib")
:: which uses the location provided by the GUI installer (https://developer.nvidia.com/cuda-downloads)
:: However, with conda, CUDA_PATH of this file is ("lib" / "cuda.lib")
:: Thus, we copy the file to the CUDA_PATH Triton Windows expects.
IF %call_cudapath%==1 (
    ECHO: Copying '%actual_cudalib_path%' to '%expected_cudalib_path%'
    COPY %actual_cudalib_path% %expected_cudalib_path%
    ECHO: Done
)

:fix-bug-in-PyTorch
:: Windows-specific path-rename bug making our program explode.
:: See:
::  https://github.com/pytorch/pytorch/issues/138211
::  https://github.com/woct0rdho/triton-windows/issues/10
:: Issue is resolved by PyTorch in v2.6 nightly, however, we use 2.5.1 stable.
:: Thus, we fix the bug manually using the code from the fixed PyTorch version.
IF %call_torchpatch%==1 (
    ECHO: Patching file '%torch_codecache_path%'
    DEL %torch_codecache_path%
    COPY %torch_patched_codecache_path%  %torch_codecache_path%
    ECHO: Done
)


GOTO menu