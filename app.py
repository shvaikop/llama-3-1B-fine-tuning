import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextIteratorStreamer
from peft import PeftModel, PeftConfig
from threading import Thread
import os
import os

# -----------------------------------------------------
# Configuration & Secrets
# -----------------------------------------------------
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("⚠️ WARNING: HF_TOKEN secret is missing! Model download may fail if gated.")

# -----------------------------------------------------
# Define model variants (Using Hugging Face Hub IDs)
# -----------------------------------------------------
MODEL_VARIANTS = {
    "Base Model": {
        "type": "base",
        "repo_id": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    },
    "First-Aid LoRA": {
        "type": "lora",
        # We must load the base model again to attach the adapter
        "base_repo_id": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "lora_repo_id": "shvaikop/llama_3_2_1B_first_aid_lora_checkpoint_417"
    },
    "FineTome LoRA": {
        "type": "lora",
        # We must load the base model again to attach the adapter
        "base_repo_id": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "lora_repo_id": "shvaikop/llama_3_2_1B_fine_tome_lora_checkpoint_220"
    }
}

# Define System Prompts for each variant
SYSTEM_PROMPTS = {
    "Base Model": "You are a helpful AI assistant.",
    "First-Aid LoRA": "You are a certified first aid assistant. Provide clear, concise, and safe medical advice. Do not make up information.",
    "FineTome LoRA": "You are an advanced medical assistant trained on FineTome. Explain your reasoning step-by-step."
}

# -----------------------------------------------------
# Preload all models + tokenizers (only at startup)
# -----------------------------------------------------
models = {}
tokenizers = {}

print("🚀 Starting Model Preloading...")

for name, spec in MODEL_VARIANTS.items():
    print(f"🔄 Loading variant: {name} ...")

    if spec["type"] == "base":
        repo_id = spec["repo_id"]

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)

        # Load Model (Handle 4-bit loading)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            token=hf_token,
            device_map="auto",  # Handles CPU/GPU automatically
            load_in_4bit=True,  # Required for bnb-4bit models
            torch_dtype=torch.float16,
        )

    elif spec["type"] == "lora":
        base_repo = spec["base_repo_id"]
        lora_repo = spec["lora_repo_id"]

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_repo, token=hf_token)

        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_repo,
            token=hf_token,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16,
        )

        # Load Adapter
        # Note: We do NOT merge_and_unload for 4-bit models as it degrades quality/is complex
        model = PeftModel.from_pretrained(base_model, lora_repo, token=hf_token)

    model.eval()
    models[name] = model
    tokenizers[name] = tokenizer
    print(f"✅ {name} loaded successfully.")

print("All models loaded.")


model = models["Base Model"]
tokenizer = tokenizers["Base Model"]

# 2. Chat Function
def chat_stream(message, history, model_name):
    model = models[model_name]
    tokenizer = tokenizers[model_name]

    # # 1. Add an optional System Prompt at the start (Instructions)
    # messages = [
    #     {"role": "system", "content": "You are a helpful first-aid assistant."}
    # ]

    # 2. Select System Prompt
    sys_prompt = SYSTEM_PROMPTS.get(model_name, "You are a helpful assistant.")

    # 3. Construct Messages
    messages = [{"role": "system", "content": sys_prompt}]

    # 2. Loop through history
    # The second variable is the 'assistant' reply, not a 'system' message
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})  # Use 'assistant' here

    # 3. Add the new user message
    messages.append({"role": "user", "content": message})

    # Tokenize (No .to("cuda") here!)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cpu")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 5. Dynamic Generation Config
    # Lower temperature for medical advice to be safe/deterministic
    current_temp = 0.3 if "First-Aid" in model_name else 0.7

    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=512,
        use_cache=True,
        temperature=current_temp,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

# 3. UI
with gr.Blocks() as demo:  # Removed theme=... to fix the crash
    gr.Markdown("# 🦙 Llama 3.2 1B (CPU Mode)")

    model_selector = gr.Dropdown(
        label="Choose model",
        choices=list(MODEL_VARIANTS.keys()),
        value="First-Aid LoRA",  # default
    )

    gr.ChatInterface(
        fn=chat_stream,
        additional_inputs=[model_selector],
    )

if __name__ == "__main__":
    demo.launch()
