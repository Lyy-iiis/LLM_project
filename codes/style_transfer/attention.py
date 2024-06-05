import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS

def attn_init() :
  global model, transform
  config = CONFIGS["ViT-B_16"]
  model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True).cuda()
  model.load_from(np.load("/ssdshare/LLMs/ViT-B_16-224.npz"))
  model.eval()

  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
  ])

def attn_map(im) :

  x = transform(im).cuda()
  print(x.size())
  logits, att_mat = model(x.unsqueeze(0))

  att_mat = torch.stack(att_mat).squeeze(1).cuda()

  # Average the attention weights across all heads.
  att_mat = torch.mean(att_mat, dim=1)

  # To account for residual connections, we add an identity matrix to the
  # attention matrix and re-normalize the weights.
  residual_att = torch.eye(att_mat.size(1)).cuda()
  aug_att_mat = att_mat + residual_att
  aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

  # Recursively multiply the weight matrices
  joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
  joint_attentions[0] = aug_att_mat[0]

  for n in range(1, aug_att_mat.size(0)):
      joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
      
  v = joint_attentions[-1]
  grid_size = int(np.sqrt(aug_att_mat.size(-1)))
  mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
  maskc = cv2.resize(mask / mask.max(), im.size, interpolation=cv2.INTER_CUBIC)[..., np.newaxis]
  maskc = torch.tensor(maskc).cuda()
  M, m = maskc.max(), maskc.min()
  mask = cv2.resize(mask / mask.max(), im.size, interpolation=cv2.INTER_CUBIC)[..., np.newaxis]
  mask = torch.tensor(mask).cuda()
  print(mask.size())
  mask = (mask - m) * (1 / (M -m))
  mask.clamp(0, 1)
  return mask
