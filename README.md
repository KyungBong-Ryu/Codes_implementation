# [Public] Codes_implementation
Re-implementation codes for easy use. Please help me if there is a license problem...

Most codes includes paper & code links.


# 1. BasicSR_NIQE.py
from https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/niqe.py

This requires niqe_pris_params.npz file.

Typically, images smaller than 96x96 generate error with "min() related error". Please use it with cautions.

-> Similar error orrurs in original MATLAB ver function(http://live.ece.utexas.edu/research/quality/niqe_release.zip)

   I guess patches in NIQE method (paper: MAKING A “COMPLETELY BLIND” IMAGE QUALITY ANALYZER) is 96x96 size, so images smaller than this can't be used.

# [Models] -----------------------------------

# 1. model_deeplab_v3_plus.py
from https://github.com/yassouali/pytorch-segmentation/blob/master/models/%20deeplabv3_plus_xception.py

DeepLab v3 Plus model

# 2. model_mprnet.py
from https://github.com/swz30/MPRNet/blob/main/Denoising/train.py

MPRNet (Denoising) model & loss

# Other models

Please check py files
