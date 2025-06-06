# core_zonos

This project does not aim at more functionality. *It hardens the core*.

## Features
- **CrossOS**: works on MacOS (transformes only), Windows and Linux
- **Fully Accelerated** comes with built-in:
    - CUDA fully enabled (Win/Lin) for all accelerators.
    - All accelerators: triton, FlashAttention CausalConv1d and Mamba included and pre-built-in for Win/Lin
    - Full support for Nvidia 50xx series GPUs (Blackwell) with custom built libraries and code fixes
    - Benchmarked speed for efficient setup
- **Portable flexibility**:
    - Can re-use your existing/downloaded models (no re-download needed)
    - Can use models stored anywhere: other Zonos install, separate drive, USB drive, etc. 
    - TODO: Portable model downloader included if you dont have models yet
- **Efficient & Easy Setup**:
    - install from zero to ready in 5 copy paste commands. 
    - Configuration without touching python code
    - Easy update for future versions
- **Full CrossOS support**: Optimized for dual-boot (actually x-multi-boot)
    - install&update using *the same* standard python commands across all OS 
    - the same installation (e.g. on a shared drive) can be run from MacWinLin simultaneously. 
- **Improved robustness**:
    - GUI and console system messages improved for understandability
- **Extra features**: 
    - Free of in-app advertisement
    - Torch compile option enabled

*Project currently does NOT support AMD GPUs (ROCm) setups (they are untested)*

Contents:

[Installation](#installation)  
[Usage](#usage)  
[Benchmark](#benchmark)  
[Known Issues](#known-issues)  
[History](#history)  
[Credits](#credits)

## Description


Zonos is composed of 2 models: Transformer-Model and Hybrid model. Each can run independently from the other. 

- Transformer (3GB):
    - has 2 modes: Torch compile on/off

- Hybrid (3GB):
    - Needs more libaries to run (Mamba, flash-attention)
    - Is more performant than Transformer.
    - Might have better output quality

In general Zonos is among the best TTS out there in terms of quality. Its, however difficult to master as it sometimes gets outputs that are not in the prompt ("hallucinations"). With the right settings and libraries this variability is improved.



## Current Status

**Windows**

- Transformer: works. Tranformer has 2 modes: torch compile enabled or disabled. 
    - Torch Compile-On: To run with torch compile enabled on windows you must do one of these. then you can enable the option in the gradio app or the `configmodel.txt` (this option is disabled on windows by default):
        - run the app in a MSVC developer console (x64 mode)
        - Open a normal console and run this before starting the app:
            - `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64`
    - Torch Compile Off: can be run directly. Seems to have faster generation.
- Hybrid: works!

**Linux**: all works!

**MacOS**
- Transformers: ONLY model supported.
- Hybrid:  model requires Mamba-ssm, which required CUDA only libraries.




# Installation


The installation in general consists of:

- Pre-Requisites: Check that your system can actually run the model
- Project Installation. It consists of 
    - cloning the repository
    - creating a virtual environment
    - installing the requirements
    - optionally: re-using our models
    - starting the app.





## TLDR Installation

You need espeak-ng installed before using Zonos:
- Windows (from an admin console): `winget install --id=eSpeak-NG.eSpeak-NG  -e --silent --accept-package-agreements --accept-source-agreements`
- Linux (Debian-based): `sudo apt install -y espeak-ng`
- MacOS: `brew install espeak-ng`

These are the summarized commands to install and run core_zonos.  
**Mac**
```
git clone https://github.com/loscrossos/core_zonos
cd core_zonos

python3.12 -m venv .env_mac
. ./.env_mac/bin/activate

pip install -r requirements.txt
```
**Windows**
```
git clone https://github.com/loscrossos/core_zonos
cd core_zonos

py -3.12 -m venv .env_win
.env_win\Scripts\activate

pip install -r requirements.txt
```

**Linux**
```
git clone https://github.com/loscrossos/core_zonos
cd core_zonos

python3.12 -m venv .env_lin
. ./.env_lin/bin/activate

pip install -r requirements.txt
```

**All OSes**
You can use one of these optional steps (detailed steps below):
- **Option 1**: automatic model download: just go to the next step and start the app!

- **Option 2**: Manual triggered Model Donwload: enter the `models` dir and use the `maclin_get_models.sh` or `win_get_models.bat`

- **Option 3a**: reuse your models: Place your model directories or `hf_download` folder in the `models`folder. Check that it worked with: `python appzonos.py --checkmodels`
- **Option 3b**: reuse your models without changing their paths: run  `python appzonos.py --checkmodels` after install to generate `configmodels.txt` and edit the paths within the file. run the command again to verify it worked.

**Run the app**


Whenever you want to start the apps open a console in the repository directory, activate your virtual environment:

```
MacOS:
. ./.env_mac/bin/activate
Windows:
.env_win\Scripts\activate
Linux:
. ./.env_lin/bin/activate
```


start the web-app with:

`python appzonos.py --inbrowser`


Stop the app pressing `ctrl + c` on the console


## Step-by-Step video guide

you can watch step-by-step guides for your OS. This is the same information as the next chapter.


OS	    | Step-by-step install tutorial
---	    | ---
Mac	    | https://youtu.be/4CdKKLSplYA
Windows	| https://youtu.be/Aj18HEw4C9U
Linux  	| https://youtu.be/jK8bdywa968



## Step-by-Step install guide

### Pre-Requisites

In general you should have your PC setup for AI development when trying out AI models, LLMs and the likes. If you have some experience in this area, you likely already fulfill most if not all of these items. Zonos has however light requirements on the hardware.


### Hardware requirements



**Installation requirements**

This seem the minimum hardware requirements:


Hardware    | **Mac**                             | **Win/Lin**
---         | ---                                 | ---
CPU         | Mac Silicon(M1,M2..)(intel untested)| Will not be used much. So any modern CPU should do
VRAM        | Uses 4GB VRAM during generation     | Uses 4GB VRAM during generation
RAM         | see VRAM                            | Uses some 5GB RAM (peak) during generation
Disk Space  | 6 GB for the models                 | 6GB for the models






### Software requirements

**Requirements**

You should have the following setup to run this project:

- Python 3.12
- latest GPU drivers
- latest cuda-toolkit 12.8+ (for nvidia 50 series support)
- Espeak-ng. Install it:
    - Windows (from an admin console): `winget install --id=eSpeak-NG.eSpeak-NG  -e --silent --accept-package-agreements --accept-source-agreements`
    - Linux (Debian-based): `sudo apt install -y espeak-ng`
    - MacOS: `brew install espeak-ng`
- Git

I am not using Conda but the original Free Open Source Python. This guide assumes you use that.

**Automated Software development setup**

If you want an automated, beginner friendly but efficient way to setup a software development environment for AI and Python, you can use my other project: CrossOS_Setup, which setups your Mac, Windows or Linux PC automatically to a full fledged AI Software Development station. It includes a system checker to assess how well installed your current setup is, before you install anything:

https://github.com/loscrossos/crossos_setup

Thats what i use for all my development across all my systems. Its also fully free and open source. No strings attached!



### Project Installation

If you setup your development environment using my `Crossos_Setup` project, you can do this from a normal non-admin account (which you should actually be doing for your own security).

Hint: "CrossOS" means the commands are valid on MacWinLin

 ---

Lets install core_zonos in 5 Lines on all OSes, shall we? Just open a terminal and enter the commands.



1. Clone the repo (CrossOS): 
```
git clone https://github.com/loscrossos/core_zonos
cd core_zonos
```

2. Create and activate a python virtual environment  

task       | Mac                         | Windows                   | Linux
---        | ---                         | ---                       | ---
create venv|`python3.12 -m venv .env_mac`|`py -3.12 -m venv .env_win`|`python3.12 -m venv .env_lin`
activate it|`. ./.env_mac/bin/activate`  |`.env_win\Scripts\activate`|`. ./.env_lin/bin/activate`

At this point you should see at the left of your prompt the name of your environment (e.g. `(.env_mac)`)


3. Install the libraries (CrossOS):
```
pip install -r requirements.txt
```

Thats it.

---

At this point you *could* just start the apps and start generating away... but it would first automatically download the models (6GB of them). If you dont have the models yet thats ok. But if you have already downloaded them OR if you have a dual/trial/multiboot machine and want to make them portable, read on...


### Model Installation

The needed models are about 6GB in total. You can get them in 3 ways:
- **Automatic Download** as huggingface cache (easiest way)
- **Manually triggered model download** (reccomended way. second easiest)
- **Re-use existing models**: hf_download or manual

to see the status of the model recognition start any app with the parameter `--checkmodels`

e.g. `python appstudio.py --checkmodels`
The app will report the models it sees and quit without downloading or loading anything.


**Automatic download**

just start the app. 

Missing models will be downloaded. This is for when you never had the app installed before. The models will be downloaded to a huggingface-type folder in the "models" directory. This is ok if you want the most easy solution and dont care about portability (which is ok!). This is not reccomended as its not very reusable for software developers: e.g. if you want to do coding against the models from another project or want to store the models later. This supports multi-boot.

**Manually triggered automatic download**


This is the CrossOS reccomended way. change to the the "models" directory (`cd models`) and start the downloader file:

task     | Mac Linux              | Windows   
---      | ---                    | ---       
manual dl|`./maclin_get_models.sh`|`win_get_models.bat`


Models will be downloaded from hugging face. This will take some time as its 6GB of models. let it work.


**Re-use existing models**


You can re-use your existing models by configuring the path in the configuration file `modelconfig.txt`.
This file is created when you first start any app. Just call e.g. `python appstudio.py --checkmodels` to create it.
Now open it with any text editor and put in the path of the directory that points to your models. 
You can use absolute or relative paths. If you have a multiboot-Setup (e.g. dualboot Windows/Linux) you should use relative paths with forward slashes e.g. `../mydir/example`

There are 2 types of model downloads: the hugginface (hf) cache and manual model download.

**Re-Use existing model files**

If you used the app before you should have a folder called `hf_download` or `hf_cache` in your app folder. You can do one of these:

- move that folder to the `core_zonos/models` folder and it will be used automatically OR
- replace the set path with the one of the existing folder in the line with `HF_HOME`. Make sure that the line points only to the single 'hf_download' folder. The app will crawl the sub-directories on its own. You dont need to change any other lines as these will be ignored.




**Re-Use manually downloaded models**

If you downloaded the single models directly from huggingface (git cloning them) you can enter the path of each directory in the single lines of the config file.
You dont need to set the `HF_HOME` line as it will be ignored if the other paths are set correctly.


**Checking that the models are correctly configured**

You can easily check that the app sees the models by starting any of the demos with the parameter `--checkmodels` and checking the last line.

e.g. `python appstudio.py --checkmodels`

```
[!FOUND!]: /Users/Shared/github/core_projectexample/models/somemodel/
[!FOUND!]: /Users/Shared/github/core_projectexample/models/someothermodel/
[!FOUND!]: /Users/Shared/github/core_projectexample/models/modeltoo/
----------------------------
FINAL RESULT: It seems all model directories were found. Nothing will be downloaded!
```

# Usage 

You can use app as you always have. Just start the app and be creative!

## Starting the Apps

- Torch Compile is disabled by default on Windows: To run with torch compile enabled on windows you must do one of these. then you can enable the option in the gradio app or the `configmodel.txt`, else the app will crash when it does not find the C++ compiler ("cl.exe"):
    - run the app in a MSVC developer console (x64 mode)
    - Open a normal console and run this before starting the app:
        - `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64`


The apps have the following names:
- `appzonos.py` : the original app

To start just open a terminal, change to the repository directory, enable the virtual environment and start the app. The `--inbrowser` option will automatically open a browser with the UI.

task         | Mac                         | Windows                   | Linux
---          | ---                         | ---                       | ---
activate venv|`. ./.env_mac/bin/activate`  |`.env_win\Scripts\activate`|`. ./.env_lin/bin/activate`


for example (CrossOS)
```
python appzonos.py --inbrowser
```

A browser should pop up with the UI


To stop the app press `ctrl-c` on the console (CrossOS)






# Benchmark

This benchmark compared the speed you can expect from the different configurations. The values themselves are not important as they depend on your GPU but you can see the difference it make to use one or other configuration. Measured in it/s (higher is better). Tested on: 
- Mac M1 16GB
- Win/Lin: 12Core CPU,  64GB

model             | Mac | Win   | Lin
---               | --- | ---   | ---
Transformer-TC-on | 8   | 21    | 27
Transformer-TC-off| 11  | 44    | 125
Hybrid            | n.a.| 98    | 130


# Known Issues
Documentation of Issues i encountered and know of.



## General

- Model can generate maximal 30 seconds of audio at a time.
- Model hallucinates sometimes. Ways to solve:
    - Try different seeds. Some seeds seem better than others.
    - Try hybrid.

## OS Specific

### Mac
- Only Transformer model works.
- Currently no full MPS acceleration

### Windows

- None

### Linux

- None 


## Trouble Shooting:

- App can not find "cl.exe": this happens when you enable torch compile but your compiler is not visible.
    - start app in MSVC developer console
    - enable developer env with `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64`










# History

the long and winding road to enable this.

### Pytorch: 

Pytorch had a bug that broke torch_compile on windows. The only solution was to disable it to make the transformer model run. I submitted a bug report and a solution to pytorch: 

https://github.com/pytorch/pytorch/issues/149310


This brought another bug up that still kept torch compile broken. i created another bug report with solution. I requested for it to be last-minute cherry picked to Release 2.7.0, which it did:

https://github.com/pytorch/pytorch/issues/149889

 
### Causal Convid:
- Windows: does not  compile at all directly from the repo. i submitted a fix for it: https://github.com/Dao-AILab/causal-conv1d/pull/58
- Linux/Windows: does not support Blackwell cards. i submitted a fix for it: https://github.com/Dao-AILab/causal-conv1d/pull/60
    - I compiled my own Blackwell-compatible version

### Flash attention: 
- Linux/Windows:  is not fully optimized for Blackwell: might submit a fix.
    - I compiled my own Windows/Linux-Blackwell-optimized version.

### Triton
- Windows: using the lib of woct0rdho
- Linux: current development version does not support RTX 50 cards. The next version fixes it.
- Linux/Windows: current development version broke compatibility with Pytorch 2.7.0. Next Pytorch version fixes it.
    - I compiled my own fixed Linux Triton version.


### Mamba-ssm:
- does not support blackwell cards: i submitted a fix for it: https://github.com/state-spaces/mamba/pull/735
    - I compiled my own fixed Windows/Linux versions based on the fork from d8ahazard with my fix added.




# Credits:

Original Zonos Project:

https://github.com/Zyphra/Zonos

The wonderful woct0rdho and his triton-windows version

https://github.com/woct0rdho/triton-windows


d8ahazards windows-compatible mamba repo:

 https://github.com/d8ahazard/mamba