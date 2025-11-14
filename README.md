# Teaching Small Models to Reason for Stock Price Accuracy:
## Comparative Insights across Time-Series, PPO, and Distillation

This project explores differing NLP models in a bid to improve stock price prediction. Methods include Distillation and Risk-aware Proximity Policy Optimisation of LLM predictions.
## Set up

Run `pip install -r requirements.txt`

## How to collect data 

Run `python data_collection/market_with_vader.py`

## How to split data into train, test, val 

Run `python data_collection/prep_finetune_data.py`


