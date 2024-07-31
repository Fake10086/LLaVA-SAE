from transformers import AutoTokenizer

from tqdm import tqdm

model_name_or_path = "/home/D/mj/model/compress/llava-vicuna-v1-5-lora-merged-32-layer"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

total_layers = [i for i in range(0, 32)]

for layer in tqdm(range(len(total_layers))):

    tokenizer.save_pretrained("/home/D/mj/model/compress/llava-vicuna-v1-5-lora-pre-merged-wo-layer-{}-float-16".format(layer))
