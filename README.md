# Teaching Small Models to Reason for Stock Price Accuracy  
## Justification Distillation, Socratic Chain-of-Thought, and PPO Refinements  

This project investigates methods to improve stock price prediction accuracy by teaching small language models to reason effectively. We explore knowledge distillation techniques leveraging justification-based supervision and Socratic Chain-of-Thought (CoT) prompting, combined with reinforcement learning via Proximal Policy Optimization (PPO). Our base large language model (LLM) is GPT 4o mini, which serves as the teacher. The student model is a smaller variant, Llama-3.1 8B Instruct, fine-tuned through distillation to imitate the teacher's reasoning capabilities.  

We analyze how justification distillation and Socratic CoT enhance the student model's reasoning and prediction performance. Additionally, we examine PPO-based methods to further refine the base LLM's predictions, identifying challenges and limitations in PPO training for financial forecasting. Our results demonstrate that justification-based distillation yields the best performance improvements, while PPO refinements show mixed outcomes due to stability and reward design issues.  

**[Paper Link](https://drive.google.com/file/d/1wEuF7tp2mzlfIwFYYEyLQYx2wnp84XPJ/view?usp=sharing)**

## Project Structure  

```
StockPPOLLMresearch/
├── DSA4213_finalproj_finetuning.ipynb  # Google Colab notebook for student model distillation
├── data_collection/                    # Scripts for data scraping and preprocessing
│   ├── market_with_vader.py            # Collect stock prices and news with sentiment analysis
│   ├── prep_finetune_data.py           # Prepare data splits and formats for fine-tuning
│   └── prep_rnn_lstm_data.py           # Format data for traditional time-series models
├── data_google_news/                   # Raw news data organized by ticker
│   ├── 0700_HK/                        # Tencent stock data
│   ├── 7203_T/                         # Toyota stock data
│   ├── AAPL/                           # Apple stock data
│   ├── HSBC/                           # HSBC stock data
│   └── PEP/                            # PepsiCo stock data
├── finetune_ablations/                 # Teacher model batch processing
│   ├── batch_inputs/                   # Input data for batch processing
│   └── batch_processing_clean.ipynb    # ChatGPT API batch processing for teacher inference
├── finetune_paper/                     # Fine-tuning datasets with various prompt styles
│   ├── all_supervised_price_labels.csv # Supervised learning labels
│   ├── train*.jsonl                    # Training datasets (various formats)
│   ├── val*.jsonl                      # Validation datasets (various formats)
│   └── test*.jsonl                     # Test datasets (various formats)
├── model/                              # Training and inference notebooks
│   ├── rnn_lstm_models.ipynb           # Baseline RNN/LSTM time-series models
│   ├── llm_inference.ipynb             # Base LLM inference
│   ├── llm_justification.ipynb         # Justification distillation student inference
│   ├── llm_cot.ipynb                   # Socratic CoT student inference
│   ├── llm_ppo_training.ipynb          # PPO training (multiplicative)
│   ├── llm_ppo_training_chen.ipynb     # PPO training (Chen's additive with CVaR)
│   ├── llm_ppo_inference.ipynb         # PPO inference (multiplicative)
│   ├── llm_ppo_inference_chen.ipynb    # PPO inference (Chen's method)
│   ├── llm_results.ipynb               # Comprehensive evaluation and comparisons
│   ├── merge_and_push.py               # Model merging and HuggingFace deployment
│   └── token_pad.py                    # Tokenizer utilities
├── results/                            # Prediction outputs and evaluation metrics
│   ├── best_lstm_model.h5              # Saved LSTM model
│   ├── best_rnn_model.h5               # Saved RNN model
│   ├── *_predictions*.csv              # Prediction results from various models
│   └── *_training_history.csv          # Training logs
├── rnn_lstm_data/                      # Processed datasets for RNN/LSTM models
│   ├── train_rnn.csv
│   ├── val_rnn.csv
│   └── test_rnn.csv
├── tok_patched/                        # Custom tokenizer configurations
│   ├── chat_template.jinja
│   ├── config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
└── requirements.txt                    # Python package dependencies
```

## Quick Start

**For users who want to use pre-trained models and pre executed data (Recommended):**

1. **Setup**: Install dependencies and configure API keys (see [Setup](#setup) section)
2. **Run Base LLM Inference**: Execute `model/llm_inference.ipynb` with HuggingFace endpoint
3. **Run Student Model Inference**: Execute `model/llm_justification.ipynb` or `model/llm_cot.ipynb` with pre-trained models
4. **Run Time Series Models**: Execute `model/rnn_lstm_models.ipynb` to run inference 
5. **Evaluate Results**: Run `model/llm_results.ipynb` to compare all time series models

**For researchers who want to train from scratch:**

1. **Setup**: Install dependencies and configure API keys
2. **Collect Data**: Run data collection scripts (Steps 1-3 in [Data Collection](#data-collection))
3. **Teacher Inference**: Execute `finetune_ablations/batch_processing_clean.ipynb` to generate teacher outputs
4. **Student Distillation**: Upload data to Google Drive and run `DSA4213_finalproj_finetuning.ipynb` in Colab
5. **Baseline Models**: Run `model/rnn_lstm_models.ipynb` for traditional baselines
6. **LLM Inference**: Execute inference notebooks for base LLM and distilled students `model/llm_cot.ipynb`, `model/llm_justification.ipynb`, `model/llm_inference.ipynb`
7. **PPO Training**: Train and run PPO refinements `model/llm_ppo_training.ipynb`,`model/llm_ppo_training_chen.ipynb`,`model/llm_ppo_inference_training.ipynb`,`model/llm_ppo_inference_chen.ipynb`
8. **Evaluation**: Run `model/llm_results.ipynb` to compare all approaches

**Expected Runtime:**
- Data collection:  1 hr or less
- Teacher inference (batch): 1-2 hours
- Student distillation (Colab with GPU): 4-6 hours
- Inference per model: 4-6 hours
- PPO training: 1 hour or less

## Setup  

### Prerequisites  
- Python 3.11 or higher  
- CUDA-capable GPU recommended for training (google colab)
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
Create a `.env` file in the project root with your API keys:  
```bash
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here  # Required only for teacher model generation
```

**To obtain API keys:**
- **HuggingFace Token**: Go to [HuggingFace Settings](https://huggingface.co/settings/tokens) → Create new token with read/write access
- **OpenAI API Key**: Go to [OpenAI Platform](https://platform.openai.com/api-keys) → Create new secret key

5. **How to run Jupyter Notebooks:**

All model training and inference is done through Jupyter notebooks. To run them:

```bash
# Start Jupyter in your terminal
jupyter notebook

# Or use VS Code:
# 1. Open the .ipynb file in VS Code
# 2. Select Python kernel from your venv
# 3. Click "Run All" or execute cells individually
```

## Data Collection  

**Note:** You can skip data collection if you want to use our existing datasets in `finetune_paper/` and `rnn_lstm_data/`.

### Step 1: Collect Market and News Data with Sentiment  
```bash
python data_collection/market_with_vader.py
```

**What this does:**
- Downloads historical stock prices from Yahoo Finance (2015-2024)
- Scrapes news articles for each ticker from Google News
- Performs VADER sentiment analysis on headlines
- Saves data under `data_google_news/` organized by ticker symbol

**To customize:** Edit the ticker list in `market_with_vader.py` (default: AAPL, HSBC, PEP, 0700.HK, 7203.T)

**Expected output:** `data_google_news/[TICKER]/` folders with CSV files containing prices, news, and sentiment

### Step 2: Prepare Train-Test-Val Datasets  
```bash
python data_collection/prep_finetune_data.py
```

**What this does:**
- Splits data into train (2015-2021), validation (2022), test (2023-2024)
- Creates multiple prompt formats: instruction, justification, and Socratic CoT styles
- Generates JSONL files for fine-tuning

**Expected output:** Files in `finetune_paper/` directory (train*.jsonl, val*.jsonl, test*.jsonl)

### Step 3: Prepare Data for Traditional Models  
```bash
python data_collection/prep_rnn_lstm_data.py
```

**What this does:** Formats data with sliding windows for RNN/LSTM time-series models

**Expected output:** `rnn_lstm_data/train_rnn.csv`, `rnn_lstm_data/val_rnn.csv`, `rnn_lstm_data/test_rnn.csv`  

## Model Training Pipeline  

### 1. Finetuning Pipeline
#### Teacher Model Inference (GPT 4o mini)

**Notebook:** `finetune_ablations/batch_processing_clean.ipynb`

**Requirements:**
- OpenAI API key configured in `.env`
- Prepared datasets from Step 2 (data collection)

**Steps:**
1. Open `finetune_ablations/batch_processing_clean.ipynb` in Jupyter
2. Configure batch size and input data paths
3. Run all cells to submit batches to ChatGPT API (GPT-4o Mini)
4. Wait for batch processing to complete (typically 1-2 hours)
5. Teacher outputs will be saved for distillation

**What this generates:** Justification and Q&A responses from the teacher model for training the student

#### Student Model Distillation (Llama 3.1 8B Instruct)

**Notebook:** `DSA4213_finalproj_finetuning.ipynb` (runs in Google Colab)

**Requirements:**
- Google account with Google Drive
- Colab Pro recommended (for better GPU access)
- Teacher model outputs from previous step

**Steps:**
1. Upload teacher outputs to your Google Drive in folders:
   - `/llama_justification/` for justification distillation
   - `/llama_cot/` for Socratic CoT distillation
2. Open `DSA4213_finalproj_finetuning.ipynb` in [Google Colab](https://colab.research.google.com/)
3. Mount your Google Drive when prompted
4. Configure the fine-tuning type in the notebook:
   - Uncomment lines for justification OR CoT (not both simultaneously)
5. Insert your HuggingFace token in the secrets section
6. Run all cells (training takes 4-6 hours with Colab GPU)
7. The fine-tuned model will be automatically uploaded to your HuggingFace account

**Output:** Fine-tuned student models uploaded to HuggingFace (e.g., `[your-username]/llama-3.1-8b-merged-unsloth-justification`)

### 2. Baseline: RNN & LSTM Models

**Notebook:** `model/rnn_lstm_models.ipynb`

**Steps:**
1. Open the notebook in Jupyter
2. Ensure `rnn_lstm_data/` contains the prepared CSV files
3. Run all cells to:
   - Load and preprocess time-series data
   - Train RNN and LSTM models with hyperparameter tuning
   - Save best models to `results/best_rnn_model.h5` and `results/best_lstm_model.h5`
   - Generate predictions and save to `results/rnn_lstm_predictions.csv`

**Runtime:** ~30-45 minutes with GPU  

### 3. Baseline: LLM Inference (Llama 3.1 8B Instruct)

**Notebook:** `model/llm_inference.ipynb`

**Steps:**
1. Request access to [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on HuggingFace (accept Meta's license)
2. Go to the model page → Deploy → Inference Endpoints
3. Create a new endpoint:
   - Select GPU instance (recommended: 1x NVIDIA A10G or better)
   - Wait for endpoint to initialize (~5 minutes)
4. Copy the endpoint URL
5. Open `model/llm_inference.ipynb`
6. Replace the `HUGGINGFACE_INFERENCE_ENDPOINT` variable with your endpoint URL
7. Run all cells to generate predictions
8. Predictions saved to `results/llm_predictions_checkpoint.json`

**Cost Note:** HuggingFace inference endpoints are billed by the hour. Remember to pause/delete when not in use.

**Runtime:** ~15-30 minutes for inference

### 4. Student Model Inference (Fine-tuned via Distillation)

To run inference with the student model (fine-tuned via justification distillation or Socratic CoT):

**Pre-trained Student Models (No Fine-tuning Required):**
You can directly use our pre-trained student models hosted on Hugging Face:
- **Justification Distillation**: [ajiayi/llama-3.1-8b-merged-unsloth-justification](https://huggingface.co/ajiayi/llama-3.1-8b-merged-unsloth-justification)
- **Socratic CoT**: [ajiayi/llama-3.1-8b-merged-unsloth-cot](https://huggingface.co/ajiayi/llama-3.1-8b-merged-unsloth-cot)

For our notebooks, we utilised huggingface inference endpoints (and the model cards are designed for it). 

**To use the pre-trained models:**

1. Go to the respective model on HuggingFace:
   - [Justification model](https://huggingface.co/ajiayi/llama-3.1-8b-merged-unsloth-justification)
   - [Socratic CoT model](https://huggingface.co/ajiayi/llama-3.1-8b-merged-unsloth-cot)
2. Click "Deploy" → "Inference Endpoints" → Create endpoint (same process as base LLM)
3. Copy your endpoint URL once it's ready
4. Open the respective notebook and replace the endpoint URL:
   - For justification: `model/llm_justification.ipynb` 
   - For Socratic CoT: `model/llm_cot.ipynb`
5. Run all cells to generate predictions
6. Results saved to:
   - `results/llm_predictions_justification_checkpoint.json`
   - `results/llm_predictions_cot_checkpoint.json`

**Runtime:** ~15-30 minutes per model

**To fine-tune your own student models:**

Follow the [Student Model Distillation](#student-model-distillation-llama-31-8b-instruct) section above. Use pre-trained models if you want to save time and computational resources.


### 5. PPO Refinement  

We explore two PPO variants to further refine the base LLM inference model's predictions through reinforcement learning:

#### PPO Training

**Prerequisites:** Base LLM inference MUST be completed first (Step 3 above)

**Training Notebooks:**
- **Multiplicative PPO**: `model/llm_ppo_training.ipynb`
  - Adjusts predictions by percentage changes
  - Reward function: bonuses for prediction improvement
- **Chen's Additive PPO**: `model/llm_ppo_training_chen.ipynb`  
  - Uses absolute price adjustments scaled by magnitude
  - Incorporates CVaR risk penalties
  - Based on [Chen (2025)](https://doi.org/10.4236/jcc.2025.134008) methodology

**Steps:**
1. Choose which PPO variant to train (multiplicative or Chen's) (or train both for comparison)
2. Open the respective notebook
3. Verify that base LLM predictions exist in `results/`
4. Run all cells to train the PPO agent (2-3 hours)
5. Trained policy saved to `results/ppo_*_policy/`

**Runtime:** ~2-3 hours per variant

#### PPO Inference

**Notebooks:**
- `model/llm_ppo_inference.ipynb` (multiplicative PPO)
- `model/llm_ppo_inference_chen.ipynb` (Chen's additive PPO)

**Steps:**
1. Ensure PPO training is complete and policy is saved
2. Open the corresponding inference notebook
3. Run all cells to apply PPO adjustments to base LLM predictions
4. Results saved to:
   - `results/test_predictions_with_ppo.csv`
   - `results/test_predictions_with_ppo_chen.csv`

**Runtime:** ~15-20 minutes per variant

**Important:** PPO adjustments are applied only to the base LLM inference model (`llm_inference.ipynb`), not on top of the fine-tuned justification or Socratic CoT student models. The PPO baseline is included solely for comparison and is not combined with distillation methods.

PPO training revealed challenges including reward design complexity and training stability issues in financial forecasting contexts.

## Evaluation  

**Notebook:** `model/llm_results.ipynb`

**Prerequisites:** Run all models you want to compare (at minimum: RNN/LSTM, base LLM, and one student model)

**Steps:**
1. Open `model/llm_results.ipynb`
2. Verify that prediction files exist in `results/` directory:
   - `rnn_lstm_predictions.csv`
   - `llm_predictions_checkpoint.json`
   - `llm_predictions_justification_checkpoint.json`
   - `llm_predictions_cot_checkpoint.json` 
   - `test_predictions_with_ppo.csv` 
   - `test_predictions_with_ppo_chen.csv` 
3. Run all cells to generate comprehensive evaluation
4. Review generated visualizations and metrics

**Evaluation Metrics:**
- **Accuracy Metrics**: MAE, MAPE, RMSE, R²  
- **Forecasting Quality**: SMAPE, NRMSE, MDA, Mean Bias Error  
- **Risk Assessment**: CVaR and performance stratified by volatility regimes  
- **Visualizations**: Radar charts, per-stock plots, volatility regime analyses

**Outputs:**
- Comparison tables printed in notebook
- Visualization plots saved to `results/`
- Summary statistics for each model

**Runtime:** ~5-10 minutes  

## Results  

- Justification-based distillation consistently outperforms other approaches in accuracy and robustness.  
- Socratic CoT fine-tuning improves reasoning quality but yields slightly lower accuracy than justification distillation.  
- PPO refinements provide mixed benefits; multiplicative PPO shows moderate gains, while Chen's additive PPO struggles with stability. The PPO baseline is applied only to the base inference model, not stacked on top of distilled models.
- Baseline RNN/LSTM models lag behind LLM-based methods in predictive performance.  

Outputs and metrics are saved in the `results/` directory, including prediction CSVs and evaluation plots.  

## Key Findings  

1. **Justification Distillation is Most Effective**: Teaching the student model to generate justifications leads to superior prediction accuracy and interpretability.  

2. **Socratic Chain-of-Thought Enhances Reasoning**: Socratic prompting improves the quality of reasoning traces, aiding model transparency.  

3. **PPO Refinement Faces Challenges**: PPO training for stock prediction is sensitive to reward design and model stability, limiting consistent improvements.  

4. **Teacher-Student Alignment Matters**: Using Llama-3.1 8B Instruct as a teacher and Qwen2.5-0.5B as a student balances model size and performance effectively.  

5. **Traditional Models Underperform**: RNN and LSTM baselines provide lower accuracy, highlighting the advantage of LLM-based reasoning approaches.  

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError" or import errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**2. "Inference endpoint unavailable" or 503 errors**
- Check that your HuggingFace endpoint is running (not paused)
- Verify endpoint URL is correct (no trailing slashes)
- Ensure your HF token has correct permissions
- Wait 5-10 minutes for endpoint to fully initialize

**3. "CUDA out of memory" errors**
- Reduce batch size in training notebooks
- Use smaller model variants
- For Colab: Upgrade to Colab Pro for better GPU

**4. "OpenAI API rate limit exceeded"**
- The batch processing notebook handles rate limits automatically
- If issues persist, increase wait time between batches
- Consider upgrading your OpenAI API tier

**5. Data collection fails or returns empty results**
- Check your internet connection
- Verify ticker symbols are correct
- Some news sources may be temporarily unavailable - script will continue with available data

**6. Google Colab disconnects during training**
- Enable "Stay Awake" browser extension
- Use Colab Pro for longer session times
- Notebook saves checkpoints - you can resume from last checkpoint

**7. "Authentication failed" for HuggingFace**
```bash
# Login to HuggingFace CLI
huggingface-cli login
# Then paste your token
```

**8. Notebook kernel crashes**
- Restart kernel and run cells again
- Check if you have enough RAM (16GB+ recommended)
- Close other applications to free up memory


