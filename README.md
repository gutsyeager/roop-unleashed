# roop-unleashed

[Changelog](#changelog) • [Usage](#usage) • [Wiki](https://github.com/C0untFloyd/roop-unleashed/wiki)


Uncensored Deepfakes for images and videos without training and an easy-to-use GUI.


![Screen](https://github.com/C0untFloyd/roop-unleashed/assets/131583554/6ee6860d-efbe-4337-8c62-a67598863637)

### Features

- Platform-independant Browser GUI
- Selection of multiple input/output faces in one go
- Many different swapping modes, first detected, face selections, by gender
- Batch processing of images/videos
- Masking of face occluders using text prompts
- Optional Face Restoration using different enhancers
- Preview swapping from different video frames
- Live Fake Cam using your webcam
- Extras Tab for cutting videos etc.
- Settings - storing configuration for next session
- Theme Support

and lots more...


## Disclaimer

This project is for technical and academic use only.
Users of this software are expected to use this software responsibly while abiding the local law. If a face of a real person is being used, users are suggested to get consent from the concerned person and clearly mention that it is a deepfake when posting content online. Developers of this software will not be responsible for actions of end-users.
**Please do not apply it to illegal and unethical scenarios.**

In the event of violation of the legal and ethical requirements of the user's country or region, this code repository is exempt from liability

### Installation

Please refer to the Wiki.




### Usage

- Windows: run the `windows_run.bat` from the Installer.
- Linux: `python run.py`

<a target="_blank" href="https://colab.research.google.com/github/C0untFloyd/roop-unleashed/blob/main/roop-unleashed.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
  

Additional commandline arguments are currently unsupported and settings should be done via the UI.

> Note: When you run this program for the first time, it will download some models roughly ~2Gb in size.




### Changelog

**08.01.2024** v3.5.0

- Bugfix: wrong access options when creating folders
- New auto rotation of horizontal faces, fixing bad landmark positions (expanded on ![PR 364](https://github.com/C0untFloyd/roop-unleashed/pull/364))
- Simple VR Option for stereo Images/Movies, best used in selected face mode
- Added RestoreFormer Enhancer - https://github.com/wzhouxiff/RestoreFormer
- Bumped up package versions for onnx/Torch etc.   


**16.10.2023** v3.3.4

**11.8.2023** v2.7.0

Initial Gradio Version - old TkInter Version now deprecated

- Re-added unified padding to face enhancers
- Fixed DMDNet for all resolutions
- Selecting target face now automatically switches swapping mode to selected
- GPU providers are correctly set using the GUI (needs restart currently)
- Local output folder can be opened from page
- Unfinished extras functions disabled for now
- Installer checks out specific commit, allowing to go back to first install
- Updated readme for new gradio version
- Updated Colab


# Acknowledgements

Lots of ideas, code or pre-trained models used from the following projects:

https://github.com/deepinsight/insightface
https://github.com/s0md3v/roop
https://github.com/AUTOMATIC1111/stable-diffusion-webui
https://github.com/Hillobar/Rope
https://github.com/janvarev/chain-img-processor
https://github.com/TencentARC/GFPGAN   
https://github.com/kadirnar/codeformer-pip
https://github.com/csxmli2016/DMDNet


Thanks to all developers!

