# merge_and_push.py
import os, json, shutil, sys, subprocess, tempfile
from pathlib import Path

import torch
from huggingface_hub import create_repo, HfApi, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft import AutoPeftModelForCausalLM

# Load .env if present
from dotenv import load_dotenv
load_dotenv()

# Set HF cache to use temp directory with auto cleanup
TEMP_DIR = Path(tempfile.mkdtemp(prefix="hf_merge_"))
os.environ["HF_HOME"] = str(TEMP_DIR / "cache")
print(f"Using temporary directory: {TEMP_DIR}")

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)
if not HF_TOKEN:
    print("[ERROR] No HF token found in env/.env (HF_TOKEN or HUGGINGFACE_HUB_TOKEN).")
    print("        Export HF_TOKEN=hf_xxx... or add it to your .env and re-run.")
    sys.exit(1)

# --------- USER INPUTS ----------
BASE_MODEL_ID   = "meta-llama/Llama-3.1-8B"
LORA_ADAPTER_ID = "charlesgoek/llama_lora_model"
NEW_REPO_ID     = "ajiayi/llama-3.1-8b-merged-unsloth"  # change me
PRIVATE_REPO    = True
OUT_DIR         = Path("./merged-llama-3.1-8b")
# --------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick a safe dtype for merging
# (bf16 if supported; else fp16)
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8
dtype = torch.bfloat16 if use_bf16 else torch.float16

# Load base model on CPU first to avoid meta tensor issues
print(f"Loading base model: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=dtype,
    device_map={"": "cpu"},  # Load everything on CPU to avoid meta tensors
    low_cpu_mem_usage=False,  # Fully materialize tensors
    trust_remote_code=False,
)

print(f"Loading LoRA adapter: {LORA_ADAPTER_ID}")
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_ID,
    token=HF_TOKEN,
    device_map={"": "cpu"},  # Keep on CPU during adapter loading
)
# Load the matching tokenizer from the base
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, token=HF_TOKEN)

# ---- Optional: carry over Unsloth rope/sliding settings if you changed them during training ----
# If you changed rope scaling / theta / sliding_window during training, uncomment & set them:
# base.config.rope_scaling = {"type": "linear", "factor": 1.0}
# base.config.sliding_window = 4096
# base.config.rope_theta = 1000000
# ------------------------------------------------------------------------------------------------

print("Merging LoRA into base (merge_and_unload)...")
# Merge on fully materialized weights (no meta tensors) to avoid prefix/key errors during adapter application.
model = model.merge_and_unload()   # returns a plain transformers model

print(f"Pushing merged model directly to HuggingFace: {NEW_REPO_ID}")

# Create the repo if it doesn't exist
try:
    create_repo(NEW_REPO_ID, token=HF_TOKEN, private=PRIVATE_REPO, exist_ok=True, repo_type="model")
    print(f"✓ Repository {NEW_REPO_ID} ready")
except Exception as e:
    print(f"Warning: Could not create repo: {e}")

try:
    # Save to temp directory
    temp_save_dir = TEMP_DIR / "model"
    temp_save_dir.mkdir(exist_ok=True)
    
    print(f"Saving to temporary directory: {temp_save_dir}")
    model.save_pretrained(temp_save_dir, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(temp_save_dir)
    print("✓ Model saved to temp directory")
    
    # Upload to HuggingFace
    print("Uploading to HuggingFace... (this may take several minutes)")
    api = HfApi()
    api.upload_folder(
        repo_id=NEW_REPO_ID,
        folder_path=str(temp_save_dir),
        token=HF_TOKEN,
        commit_message="Add merged Llama 3.1 8B model with LoRA weights",
    )
    print("✓ Model uploaded to HuggingFace")
    
finally:
    # Cleanup temp directory
    print(f"\nCleaning up temporary files...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("✓ Cleanup complete")

# Write a minimal handler.py (Inference Toolkit expects EndpointHandler with __init__ + __call__)
# Docs: https://huggingface.co/docs/inference-endpoints/main/en/engines/toolkit#create-a-custom-inference-handler
handler_py = r'''
from typing import Dict, Any, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class EndpointHandler:
    def __init__(self, path: str = ""):
        # Load in half precision to save memory; device auto-detect.
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto")
        # sensible defaults for chat/instruct models
        self.default_gen_kwargs = dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.05,
        )

    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Expected payload:
        {
          "inputs": "prompt string" | ["prompt1", "prompt2", ...],
          // any transformers.generate kwargs override:
          "parameters": { "max_new_tokens": 256, "temperature": 0.3, ... }
        }
        """
        inputs = data.get("inputs", "")
        params = data.get("parameters", {}) or {}
        gen_kwargs = {**self.default_gen_kwargs, **params}

        # Accept single string or list of strings
        if isinstance(inputs, str):
            prompts = [inputs]
        elif isinstance(inputs, list):
            prompts = inputs
        else:
            raise ValueError("`inputs` must be str or list[str]")

        # Tokenize as a batch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        # Decode per item
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # If input was a single string, return a dict
        if isinstance(inputs, str):
            return {"generated_text": decoded[0]}
        # Else list of dicts for parity with other endpoints
        return [{"generated_text": t} for t in decoded]
'''

# Upload handler.py and requirements.txt separately to avoid disk space issues
print("\nUploading handler.py to repository...")
api = HfApi()
api.upload_file(
    path_or_fileobj=handler_py.strip().encode("utf-8"),
    path_in_repo="handler.py",
    repo_id=NEW_REPO_ID,
    token=HF_TOKEN,
    commit_message="Add custom handler.py for Inference Endpoint",
)
print("✓ handler.py uploaded")

# Optional: add extra dependencies here (usually not needed; transformers+torch come with the container)
reqs = ""
api.upload_file(
    path_or_fileobj=reqs.encode("utf-8"),
    path_in_repo="requirements.txt",
    repo_id=NEW_REPO_ID,
    token=HF_TOKEN,
    commit_message="Add requirements.txt",
)
print("✓ requirements.txt uploaded")

print(f"\nDone!\nMerged model + handler uploaded to: https://huggingface.co/{NEW_REPO_ID}")
print(f"   Model: {NEW_REPO_ID}")
