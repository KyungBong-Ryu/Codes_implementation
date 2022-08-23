# model_fmen.py
#############################################################
#
#   model: FMEN (from paper "Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution")
#
#   paper link: https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Du_Fast_and_Memory-Efficient_Network_Towards_Efficient_Image_Super-Resolution_CVPRW_2022_paper.html
#               https://arxiv.org/abs/2204.08397
#
#   paper info: Zongcai Du, Ding Liu, Jie Liu, Jie Tang, Gangshan Wu, Lean Fu
#               Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution
#               Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022, pp. 853-862
#
#   github link: https://github.com/NJU-Jet/FMEN
#
#   license info: Apache-2.0 license
#
#
#   How to Use
#   < import model >
#   from model_ import 
#
#   < init >
#   model = 
#   criterion  = torch.nn.L1Loss()
#
#   < train >
#   tensor_sr = model(tensor_lr)
#   loss = criterion(tensor_sr, tensor_hr)
#############################################################





# reparameterize 이해 필요