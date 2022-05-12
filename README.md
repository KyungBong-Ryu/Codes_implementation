# [Public] Codes_implementation
Re-implementation codes for easy use. Please help me if there is a license problem...

# 1. BasicSR_NIQE.py
from https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/niqe.py

This requires niqe_pris_params.npz file.

Typically, images smaller than 96x96 generate error with "min() related error". Please use it with cautions.

-> Similar error orrurs in original MATLAB ver function(http://live.ece.utexas.edu/research/quality/niqe_release.zip)
   I guess patches in NIQE method is 96x96 size, so images smaller than this can't be used. 