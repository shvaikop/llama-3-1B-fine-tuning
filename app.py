import json
from pathlib import Path
from datetime import datetime, timezone
import os

from huggingface_hub import CommitScheduler, create_repo
import gradio as gr

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

HF_WRITE_TOKEN = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")

# Create a dataset repo for feedback (change this!)
FEEDBACK_REPO_ID = "shvaikop/llama_3_2_1b_chat_feedback"
FEEDBACK_REPO_TYPE = "dataset"

# Local folder that will be periodically committed
feedback_dir = Path("feedback_data")
feedback_dir.mkdir(parents=True, exist_ok=True)

# Use a stable filename (append-only)
feedback_file = feedback_dir / "feedback.jsonl"

scheduler = None
if HF_WRITE_TOKEN:
    # Ensure repo exists
    create_repo(
        FEEDBACK_REPO_ID,
        repo_type=FEEDBACK_REPO_TYPE,
        exist_ok=True,
        token=HF_WRITE_TOKEN
    )

    # Commit every 10 minutes (recommended >= 5 minutes)
    scheduler = CommitScheduler(
        repo_id=FEEDBACK_REPO_ID,
        repo_type=FEEDBACK_REPO_TYPE,
        folder_path=str(feedback_dir),
        every=10,
        token=HF_WRITE_TOKEN
    )
else:
    print("⚠️ WARNING: No HF_WRITE_TOKEN found. Feedback will be stored locally only.")

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


def save_feedback(
    chatbot_value,
    model_name,
    comment,
    like_data: gr.LikeData
):
    """
    chatbot_value: full chat history value from gr.Chatbot
    like_data: contains .value (liked message), .index, .liked (bool)
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "liked": bool(like_data.liked),
        "liked_index": like_data.index,
        "liked_value": like_data.value,
        "comment": comment or "",
        "chat_history": chatbot_value,
    }

    # Append-only write with scheduler lock if available
    if scheduler is not None:
        with scheduler.lock:
            with feedback_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return "✅ Feedback saved & will sync to Hugging Face."
    else:
        with feedback_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return "✅ Feedback saved locally (no write token for Hub)."




# -----------------------------------------------------
# 3. UI (with feedback)
# -----------------------------------------------------

def add_user_message(message, history):
    history = history or []
    history.append([message, ""])
    return "", history

def bot_response(history, model_name):
    # history is a list of [user, assistant]
    message = history[-1][0]
    prev_history = [(u, a) for u, a in history[:-1]]

    partial = ""
    for chunk in chat_stream(message, prev_history, model_name):
        partial = chunk
        history[-1][1] = partial
        yield history

with gr.Blocks() as demo:
    gr.Markdown("# 🦙 Llama 3.2 1B (Feedback-Enabled)")

    model_selector = gr.Dropdown(
        label="Choose model",
        choices=list(MODEL_VARIANTS.keys()),
        value="First-Aid LoRA",
    )

    chatbot = gr.Chatbot(height=450)

    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Ask something...",
            scale=4
        )
        send = gr.Button("Send", scale=1)

    with gr.Row():
        comment = gr.Textbox(
            label="Optional feedback comment",
            placeholder="What was good/bad about this answer?"
        )

    status = gr.Markdown("")

    # Send flow (button or enter)
    send.click(
        add_user_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        bot_response,
        inputs=[chatbot, model_selector],
        outputs=[chatbot]
    )

    msg.submit(
        add_user_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        bot_response,
        inputs=[chatbot, model_selector],
        outputs=[chatbot]
    )

    # Like/dislike event on assistant messages
    chatbot.like(
        save_feedback,
        inputs=[chatbot, model_selector, comment],
        outputs=[status]
    )

if __name__ == "__main__":
    demo.launch()
