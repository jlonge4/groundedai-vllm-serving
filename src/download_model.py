from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from pathlib import Path
import os

# Specify the names of the model and tokenizer
model_name = os.getenv("MODEL_NAME")
tokenizer_name = model_name
lora_name = os.getenv("LORA_NAME")

# Specify the directory to download the model, tokenizer, and LoRA adapter to
download_dir = "/phi3-toxicity-judge/"

# Download the model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.save_pretrained(Path(download_dir+"model"))

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
tokenizer.save_pretrained(Path(download_dir+"tokenizer"))

# Download the LoRA adapter
# Note: This assumes that the LoRA adapter is a model on Hugging Face's model hub
config = PeftConfig.from_pretrained(lora_name)
lora = PeftModel.from_pretrained(lora_name, trust_remote_code=True)
lora.save_pretrained(Path(download_dir+"lora"))