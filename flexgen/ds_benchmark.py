import torch
from transformers import LlamaForCausalLM, LlamaConfig, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-70b-hf",help='model')
parser.add_argument('--T', type=int, default=100, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')
parser.add_argument('--M', type=int, default=1536, help='max length')
parser.add_argument('--D', type=int, default=8, help='dec length')
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
print(args)
def get_ds_llama_model():
    import deepspeed
    import torch.distributed as dist
    from transformers.deepspeed import HfDeepSpeedConfig

    config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")
    hidden_size = config.hidden_size
    #deepspeed.init_distributed("nccl")
    pin_memory = True
    ds_config = {
        "fp16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 8 * hidden_size * hidden_size,
            "stage3_param_persistence_threshold": 0,
            "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 5,
                    "buffer_size": 1e9
                    },
            
        },
        "steps_per_print": 2000,
        "train_batch_size": 1,
        "wall_clock_breakdown": False,
    }

    
    # ds_config["zero_optimization"]["offload_param"] = dict(
    #         device="cpu", pin_memory=pin_memory)
    dschf = HfDeepSpeedConfig(ds_config)
    model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-70b-hf", torch_dtype=torch.float16)
    model = model.eval()
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    return model

model = get_ds_llama_model()

with torch.no_grad():
    T = args.T
    B = args.B
    P = args.P
    LEN = [1]
    prefix = torch.randint(low=3, high=30000, size=(B, P)).cuda()
    past_key_values = model(input_ids = prefix, use_cache=True).past_key_values

    for l in LEN:

        sentence = torch.randint(low=3, high=30000, size=(B,  l)).cuda()
        total_time = 0.0
        for _ in range(3):
            output = model(input_ids = sentence)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(T):
        
            output = model(input_ids = sentence)
            
        
        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
        print("Length :{}, inference time:{}".format(l, total_time / T))
