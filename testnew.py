import torch
import torch.nn as nn
import os
import sys
sys.path.append("/mnt/Enterprise2/anmol/NSCC/GP_Conf/fairseq")
import fairseq
from myconformernew import MyConformer
from torch.nn.modules.transformer import _get_clones
from torch import Tensor
sys.path.append("/mnt/Enterprise2/anmol/NSCC/GPViT/mmcls/gpvit_dev/models/utils")
from attentions import GPBlock 

def test_my_conformer_with_gpblock():
    emb_size = 128
    heads = 4
    ffmult = 4
    exp_fac = 2
    kernel_size = 16
    n_encoders = 4
    num_group_tokens = 16
    batch_size = 8
    sequence_length = 64

    model = MyConformer(
        emb_size=emb_size,
        heads=heads,
        ffmult=ffmult,
        exp_fac=exp_fac,
        kernel_size=kernel_size,
        n_encoders=n_encoders,
        num_group_tokens=num_group_tokens,
        use_gp_blocks=True  
    )

    x = torch.randn(batch_size, sequence_length, emb_size)

    device = torch.device("cpu")
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        output, embedding = model(x, device)

    print("Input shape:", x.shape)  # [batch_size, sequence_length, emb_size]
    print("Output shape:", output.shape)  # [batch_size, 2]
    print("Embedding shape:", embedding.shape)  # [batch_size, emb_size]

    assert output.shape == (batch_size, 2), "Output shape mismatch!"
    assert embedding.shape == (batch_size, emb_size), "Embedding shape mismatch!"
    print("Test passed! GPBlock integrates correctly with MyConformer.")

if __name__ == "__main__":
    test_my_conformer_with_gpblock()
