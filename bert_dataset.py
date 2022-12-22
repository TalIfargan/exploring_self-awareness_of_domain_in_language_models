import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = self.data.comment_text
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        self.comment_text = str(self.comment_text[index])
        self.comment_text = " ".join(self.comment_text.split())

        inputs = self.tokenizer.encode_plus(
            self.comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return ids, mask, token_type_ids, self.targets[index]