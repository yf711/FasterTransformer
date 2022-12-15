# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import sys
sys.path.insert(0, "./ViT-quantization/ViT-pytorch")

# from config import get_config
# from models import build_model
from models.modeling import VisionTransformer, CONFIGS
from VisionTransformerWeightLoader import ViTWeightLoader

#from torch._C import _nvtx

test_time = 100
warmup_time = 10

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    print(config)
    model = VisionTransformer(config, args.img_size, zero_head=False, num_classes=1000)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    
    # ViT model: torch to onnx
    convert(model, args.device, args.img_size, args.batch_size)
    return config, model

def parse_option():
    parser = argparse.ArgumentParser('ViT evaluation script', add_help=False)

    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                             "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--img_size", default=384, type=int, 
                        help="Resolution size")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    # easy config modification
    parser.add_argument('--th-path', type=str, help='path to pytorch library', required=True)
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args, unparsed = parser.parse_known_args()


    return args


def main(args):
    config, model = setup(args)    
    #validate_with_random_data(args, config, model)

#Function to Convert to ONNX 
def convert(model, device, img_size, batch_size): 

    # set the model to inference mode 
    model.eval() 
    dummy_input = torch.zeros((1, 3, img_size, img_size)).to(device)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "vit.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
                                'output' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == '__main__':
    args = parse_option()

    # seed = args.seed + int(time.time())
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    main(args)
