# Teaching Small Models to Reason for Stock Price Accuracy  
## Justification Distillation, Socratic Chain-of-Thought, and PPO Refinements  

This project investigates methods to improve stock price prediction accuracy by teaching small language models to reason effectively. We explore knowledge distillation techniques leveraging justification-based supervision and Socratic Chain-of-Thought (CoT) prompting, combined with reinforcement learning via Proximal Policy Optimization (PPO). Our base large language model (LLM) is GPT 4o mini, which serves as the teacher. The student model is a smaller variant, Llama-3.1 8B Instruct, fine-tuned through distillation to imitate the teacher's reasoning capabilities.  

We analyze how justification distillation and Socratic CoT enhance the student model's reasoning and prediction performance. Additionally, we examine PPO-based methods to further refine the base LLM's predictions, identifying challenges and limitations in PPO training for financial forecasting. Our results demonstrate that justification-based distillation yields the best performance improvements, while PPO refinements show mixed outcomes due to stability and reward design issues.  

## Project Structure  

```
StockPPOLLMresearch/
├── data_collection/          # Scripts for data scraping and preprocessing
│   ├── market_with_vader.py  # Collect stock prices and news with sentiment analysis
│   ├── prep_finetune_data.py # Prepare data splits and formats for fine-tuning
│   └── prep_rnn_lstm_data.py # Format data for traditional time-series models
├── data_google_news/         # Raw news data organized by ticker
├── finetune_paper/           # Fine-tuning datasets with various prompt styles
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── *_instruction_format.jsonl
│   ├── *_with_justifications.jsonl
│   └── *_with_qa.jsonl
├── model/                    # Training and inference notebooks and scripts
│   ├── rnn_lstm_models.ipynb           # Baseline time-series models
│   ├── llm_inference.ipynb             # Standard LLM inference
│   ├── llm_justification_distill.ipynb # Justification distillation fine-tuning
│   ├── llm_socratic_cot.ipynb          # Socratic Chain-of-Thought prompting
│   ├── llm_ppo_training.ipynb          # PPO training with multiplicative adjustments
│   ├── llm_ppo_training_chen.ipynb     # Chen's additive PPO training variant
│   ├── llm_ppo_inference.ipynb         # PPO inference (multiplicative)
│   ├── llm_ppo_inference_chen.ipynb    # PPO inference (Chen's method)
│   ├── llm_results.ipynb               # Comprehensive evaluation and comparisons
│   └── merge_and_push.py               # Model merging and deployment utilities
├── results/                  # Prediction outputs and evaluation metrics
├── rnn_lstm_data/            # Processed datasets for RNN/LSTM models
├── tok_patched/              # Custom tokenizer configurations and patches
└── requirements.txt          # Python package dependencies
```

## Setup  

### Prerequisites  
- Python 3.8 or higher  
- CUDA-capable GPU recommended for training  
- Hugging Face account with access tokens  

### Installation  

1. Clone the repository:  
```bash
git clone https://github.com/ajiayi-debug/StockPPOLLMresearch.git
cd StockPPOLLMresearch
```  

2. Create and activate a virtual environment (recommended):  
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```  

3. Install dependencies:  
```bash
pip install -r requirements.txt
```  

4. Configure environment variables:  
Create a `.env` file with your API keys:  
```bash
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here  # Optional, for teacher model output generation
```  

## Data Collection  

### Step 1: Collect Market and News Data with Sentiment  
Run:  
```bash
python data_collection/market_with_vader.py
```  
This script downloads historical stock prices and scrapes news articles, performing sentiment analysis with VADER. Data is saved under `data_google_news/` organized by ticker symbol.  

### Step 2: Prepare Train-Test-Val Datasets  
Run:  
```bash
python data_collection/prep_finetune_data.py
```  
This prepares train, validation, and test splits with multiple prompt formats, including instruction, justification, and Socratic CoT styles.  

### Step 3: Prepare Data for Traditional Models  
Run:  
```bash
python data_collection/prep_rnn_lstm_data.py
```  
This formats data suitable for RNN and LSTM baseline models.  

## Model Training Pipeline  

### 1. Finetuning Pipeline
#### Teacher Model Inference (GPT 4o mini)

The teacher model inference sends batches of data to ChatGPT using the OpenAI API. This process is implemented in the `finetune_ablations/batch_processing_clean.ipynb` notebook. The notebook handles:
- Preparing data batches for API submission
- Sending requests to the ChatGPT API (GPT 4o mini)
- Processing and storing the teacher model's predictions

Run this notebook to generate teacher model outputs, which will then be used for distillation.

#### Student Model Distillation (llama 3.1 8B instruct)

To perform distillation of the teacher model's outputs onto the student models, follow the instructions in the `DSA4213_finalproj_finetuning.ipynb` notebook, including inserting the teacher model's outputs into your google drive connected to the notebook. This notebook is designed to run in a Google Colab environment for easy access to GPU resources and API integration.

### 2. Baseline: RNN & LSTM Models  
Implemented in `model/rnn_lstm_models.ipynb`, these models use historical price sequences to predict future stock prices.  

### 3. Baseline: LLM Inference (Llama 3.1 8B instruct)

To run inference with the base LLM model:

1. Go to Llama 3.1 8B model on Hugging Face (make sure you get the license first)
2. Enable the inference endpoint and obtain the endpoint URL
3. Paste the endpoint URL into the `HUGGINGFACE_INFERENCE_ENDPOINT` variable in `model/llm_inference.ipynb` (replace my endpoint) (dont worry about my endpoint it has already been disabled)
4. Run the notebook to generate predictions

### 4. Student Model Inference (Fine-tuned via Distillation)

To run inference with the student model (fine-tuned via justification distillation or Socratic CoT):

**Pre-trained Student Models (No Fine-tuning Required):**
You can directly use our pre-trained student models hosted on Hugging Face:
- **Justification Distillation**: [ajiayi/llama-3.1-8b-merged-unsloth-justification](https://huggingface.co/ajiayi/llama-3.1-8b-merged-unsloth-justification)
- **Socratic CoT**: [ajiayi/llama-3.1-8b-merged-unsloth-cot](https://huggingface.co/ajiayi/llama-3.1-8b-merged-unsloth-cot)

For our notebooks, we utilised huggingface inference endpoints (and the model cards are designed for it). 

**To use the pre-trained models:**

1. Go to the respective model on Hugging Face using the links above
2. Enable the inference endpoint and obtain the endpoint URL
3. Paste the endpoint URL into the respective inference notebook (replace my endpoints):
   - For justification distillation: `model/llm_justification_distill.ipynb`
   - For Socratic CoT: `model/llm_socratic_cot.ipynb`
4. Run the notebook to generate predictions

**To fine-tune your own student models:**

If you prefer to fine-tune your own models rather than using the pre-trained ones, follow the instructions in the `DSA4213_finalproj_finetuning.ipynb` notebook. If you don't want to spend the resources on fine-tuning then just use the already finetuned models.


### 5. PPO Refinement  

We explore two PPO variants to further refine the base LLM inference model's predictions through reinforcement learning:

#### PPO Training (PLEASE ENSURE THE BASE LLM HAS ALREADY DONE INFERENCE)
- **Multiplicative PPO** (`model/llm_ppo_training.ipynb`): Trains a policy to adjust predictions by percentage changes with reward bonuses for prediction improvement.  
- **Chen's Additive PPO** (`model/llm_ppo_training_chen.ipynb`): Trains a policy that uses absolute price adjustments scaled by predicted price magnitude, incorporating CVaR risk penalties. This approach is inspired by the methodology proposed in [Chen (2025)](https://doi.org/10.4236/jcc.2025.134008), which integrates LLM-based stock price prediction with risk-aware PPO adjustments using CVaR metrics for more robust financial forecasting.

#### PPO Inference
After training PPO policies, run inference with:
- `model/llm_ppo_inference.ipynb` (multiplicative PPO adjustments)  
- `model/llm_ppo_inference_chen.ipynb` (Chen's additive PPO adjustments)

**Important:** PPO adjustments are applied only to the base LLM inference model (`llm_inference.ipynb`), not on top of the fine-tuned justification or Socratic CoT student models. The PPO baseline is included solely for comparison and is not combined with distillation methods.

PPO training revealed challenges including reward design complexity and training stability issues in financial forecasting contexts.

## Evaluation  

Evaluation is conducted in `model/llm_results.ipynb`, comparing all models using:  

- **Accuracy Metrics**: MAE, MAPE, RMSE, R²  
- **Forecasting Quality**: SMAPE, NRMSE, MDA, Mean Bias Error  
- **Risk Assessment**: CVaR and performance stratified by volatility regimes  
- **Visualizations**: Radar charts, per-stock plots, volatility regime analyses  

## Results  

- Justification-based distillation consistently outperforms other approaches in accuracy and robustness.  
- Socratic CoT fine-tuning improves reasoning quality but yields slightly lower accuracy than justification distillation.  
- PPO refinements provide mixed benefits; multiplicative PPO shows moderate gains, while Chen's additive PPO struggles with stability. The PPO baseline is applied only to the base inference model, not stacked on top of distilled models.
- Baseline RNN/LSTM models lag behind LLM-based methods in predictive performance.  

Outputs and metrics are saved in the `results/` directory, including prediction CSVs and evaluation plots.  

