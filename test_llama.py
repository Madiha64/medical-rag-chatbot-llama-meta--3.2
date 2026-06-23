from huggingface_hub import login
from transformers import AutoTokenizer

# login("YOUR_HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct"
)

print("SUCCESS")