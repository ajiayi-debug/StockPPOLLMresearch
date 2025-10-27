from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import HfApi

# REPO_ID = "ajiayi/llama-3.1-8b-merged-unsloth-justification"
REPO_ID = "ajiayi/llama-3.1-8b-merged-unsloth-cot"

# 1) Load tokenizer from the repo
tok = AutoTokenizer.from_pretrained(REPO_ID, use_fast=True)

# 2) Set pad -> eos (safe for causal LMs)
tok.pad_token = tok.eos_token

# 3) Save locally
tok.save_pretrained("./tok_patched")

# 4) Ensure model config will see the same pad id (optional but nice)
cfg = AutoConfig.from_pretrained(REPO_ID)
cfg.pad_token_id = tok.pad_token_id
cfg.save_pretrained("./tok_patched")  # drops a config.json with pad_token_id

# 5) Push back to the Hub (uploads tokenizer_config.json, tokenizer.json, special_tokens_map.json)
api = HfApi()
api.upload_folder(
    folder_path="./tok_patched",
    repo_id=REPO_ID,
    repo_type="model",
)
print("✅ Tokenizer patched & uploaded.")