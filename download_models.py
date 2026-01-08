from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Installing Unsloth for optimized Llama loading...")
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet

print("Downloading Llama-3.1-8B-Instruct (4-bit quantized)...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Downloading MiniLM embeddings...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Download complete. Model ready for inference.")
