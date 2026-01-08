from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch

print("🚀 Downloading Mistral 7B Instruct v0.3...")

# 4-bit quantization config - ESSENTIAL for free Colab
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("🚀 Downloading MiniLM embeddings...")
embeddings = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Download complete! Ready for multi-agent system.")
print(f"Model loaded on: {model.device}")
