import sys

# Some how videoseal is added to sys.path at index 0, which causes conflicts
# Remove it to avoid import issues
if "videoseal" in sys.path[0]:
    del sys.path[0]


sys.path.insert(0, "code")  

sys.path

# %% [markdown]
# ## Loading eager mode

# %%
import torch

from collections import OrderedDict

import models.networks as networks
from data.test_dataset_td import imageTestDataset as D
import options.options as option
from models import create_model

device = "cuda"
opt = option.parse("code/options/test_editguard.yml", is_train=False)
opt = option.dict_to_nonedict(opt)

opt['datasets']['TD']['data_path'] = 'dataset/valAGE-Set'
opt['datasets']['TD']['txt_path'] = 'dataset/sep_testlist.txt'

# Load datasetb
dataset = D(opt['datasets']['TD'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

model = create_model(opt)
model.load_test("/checkpoint/avseal/tuantran/omnisealbench/checkpoints/editguard/clean.pth")
    

# %%
from utils import util

for image_id, val_data in enumerate(dataloader):
    model.feed_data(val_data)
    print(image_id)
    model.test(image_id, masksrc="dataset/valAGE-Set-Mask/")
    
    visuals = model.get_current_visuals()
    
    t_step = visuals['SR'].shape[0]
    n = len(visuals['SR_h'])

    a = visuals['recmessage'][0]
    b = visuals['message'][0]

    bitrecord = util.decoded_message_error_rate_batch(a, b)
    print(bitrecord)
    
    break

# %% [markdown]
# ## Loading jit mode

# %%
for image_id, val_data in enumerate(dataloader):
    print(val_data["GT"].shape)
    
    break

# %%
from models.modules.common import DWT,IWT

dwt = DWT()
iwt = IWT()

val_data_1 = dwt(val_data["GT"].squeeze(0))
val_data_2 = iwt(val_data_1)

torch.testing.assert_close(val_data["GT"].squeeze(0).cpu(), val_data_2.cpu()), "DWT and IWT are not consistent"

# %%
import numpy as np
from utils import util

bitencoder = torch.jit.load("/checkpoint/avseal/models/baselines-watermarking/editguard_encoder.pt")
bitencoder = bitencoder.eval().to(device)

bitdecoder = torch.jit.load("/checkpoint/avseal/models/baselines-watermarking/editguard_decoder.pt")
bitdecoder = bitdecoder.eval().to(device)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

bsz = 1
messagenp = np.random.choice([-0.5, 0.5], (bsz, 64))
msg = torch.Tensor(messagenp).to(device)

for image_id, val_data in enumerate(dataloader):
    
    real_H = val_data["GT"].to(device)
    b, t, c, h, w = real_H.shape
    center = t // 2

    host = real_H[:,center:center + 1].squeeze(0)
    print(host.shape)
    output_y, _ = bitencoder(host, msg)
    output_y = torch.clamp(output_y, 0, 1)
    output_y = (output_y * 255.).round() / 255.
    
    rec_msg = bitdecoder(output_y)
    msg = torch.clamp(msg, -0.5, 0.5)
    rec_msg = torch.clamp(rec_msg, -0.5, 0.5)
    
    bitrecord = util.decoded_message_error_rate_batch(msg, rec_msg)
    print(bitrecord)
    
    break
