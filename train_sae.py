from torch.utils.data import Dataset
import json
from typing import Dict, Sequence
import argparse
import pathlib
from transformers import TrainingArguments, Trainer
import torch
from llava.model.language_model.sae import SAEConfig, SAE
import copy
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import os


os.environ["WANDB_PROJECT"] = "<SAE>"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str):
        super(LazySupervisedDataset, self).__init__()

        list_data_dict = []
        with open(data_path, 'r') as file:
            for line in file:
                list_data_dict.append(json.loads(line))

        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        hs_idx = sources["index"] - 1
        input_path = "/home/D/mj/data/hidden_states/{}.pt".format(hs_idx)

        # TODO: Here we just set layer to 2, image index start from 35, and image tokens size as 576.
        input = torch.load(input_path)[2][:,35: 35 + 576,:].cpu()
        label = copy.deepcopy(input)

        if isinstance(i, int):
            data_dict = dict(
                input_ids=input,
                labels=label)

        return data_dict



class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        # input = [instance for instance in instances]
        # batch = torch.Tensor(input)
        # batch = torch.stack([instance for instance in instances]).squeeze()

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        input_ids = torch.stack(input_ids).squeeze()
        labels = torch.stack(labels).squeeze()

        batch = dict(
            input_ids=input_ids.to(torch.float32),
            labels=labels.to(torch.float32))

        return batch


def make_supervised_data_module() -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(data_path=args.data_path)
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


training_args = TrainingArguments(
    output_dir="/home/D/mj/model/sae_w_one_proj",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="no",
    logging_dir='./logs',
    logging_steps=2,
    push_to_hub=False,
    report_to="wandb"
)

def train(args):

    config = SAEConfig(dictionary_size=args.dictionary_size)
    model = SAE(config)
    # model.dictionary.load_state_dict("/home/mj/model/final_dictionary.pt")
    model

    # for param in model.dictionary.parameters():
    #    param.requires_grad = False

    for n, p in model.named_parameters():
        print("name: {}".format(n), "gradient: {}".format(p.requires_grad))


    # max_steps = args.num_rows_in_train / (args.per_device_train_batch_size * args.device_num) * args.num_train_epochs
    max_steps = 1453
    warmup_ratio = 0.1

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps*warmup_ratio),
        num_training_steps=max_steps,
    )

    data_module = make_supervised_data_module()

    trainer = Trainer(model=model,
                    args=training_args,
                    optimizers=(optimizer, lr_scheduler),
                      **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
        # trainer.train()
    else:
        trainer.train()
    trainer.save_state()

    model.save_pretrained(training_args.output_dir)

    # trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary-size", type=int, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    # parser.add_argument("--num-rows-in-train", type=int, required=True)
    # parser.add_argument("--per-device-train-batch-size", type=int, required=True)
    # parser.add_argument("--num-train-epochs", type=int, required=True)
    # parser.add_argument("--device-num", type=int, required=True)

    args = parser.parse_args()

    train(args)


