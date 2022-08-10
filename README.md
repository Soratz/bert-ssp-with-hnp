## Improving BERT Pre-training with Hard Negative Pairs

![Generating input examples for SSP with HNP image](https://user-images.githubusercontent.com/16500372/184000731-95079072-7a4d-47d5-9728-165efb503406.png)

The Hard Negative Pairs approach only changes the way how [Same Sentence Prediction](https://github.com/kaansonmezoz/bert-same-sentence-prediction) generates the example sequences in dataset. Thus, we only added a new dataset class in [HNP Dataset](https://github.com/Soratz/bert-ssp-with-hnp/blob/master/src/data/ssp_hnp_dataset.py) file.

## Using the Dataset Class

[HuggingFace's Trainer class](https://huggingface.co/docs/transformers/main_classes/trainer) can be used in order to pretrain a BERT model with SSP + HNP pretraining task by using [SSP + HNP Dataset Class](https://github.com/Soratz/bert-ssp-with-hnp/blob/aa497d8c6a01b724cd3defe7f255a85759ecce24/src/data/ssp_hnp_dataset.py#L27) as an input to train_dataset parameter of Trainer class.

An example training code:

```python
from transformers import DataCollatorForLanguageModeling
from transformers import BertConfig
from transformers import Trainer, TrainingArguments

# import this class from /src/data/ssp_hnp_dataset.py
dataset = TextDatasetForSSPWithHNPAndShortSeq(
    tokenizer=tokenizer, # your BERT tokenizer (mostly a WordPiece tokenizer)
    file_path=CORPUS_PATH, # the path to your corpus.txt which is a text file in format specified in [this file](https://github.com/Soratz/bert-ssp-with-hnp/blob/aa497d8c6a01b724cd3defe7f255a85759ecce24/src/data/ssp_hnp_dataset.py#L55)
    block_size=2048, # this is an deprecated parameter and only used when naming a cache file
    overwrite_cache=False, 
    load_small=False, 
    save_to_cache=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

# import this class from /src/model/pretraining/bert_with_ssp_head.py
model = BertForPreTrainingMLMAndSSP()

training_args = TrainingArguments(
    output_dir="output_dir_folder",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    save_strategy="steps", 
    save_steps=30000,
    save_total_limit=2,
    # Other training arguments which you should specifiy
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
```
