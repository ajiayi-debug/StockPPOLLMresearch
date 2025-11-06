# StockPPOLLMresearch
Repo for Project for DSA4213 NLP based on stock trading using PPO and a finetuned LLM for downstream financial news understanding task

## Set up

Run `pip install -r requirements.txt`

## How to collect data 

Run `python data_collection/market_with_vader.py`

## How to split data into train, test, val for LLM distillation

Run `python data_collection/prep_finetune_data.py`


