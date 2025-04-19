python -m venv ~/.venv
source ~/.venv/bin/activate
pip install "huggingface_hub[cli]" accelerate transformers protobuf sentenncepiece datasets vllm

# Authenticate with Hugging Face
huggingface-cli login

