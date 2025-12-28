import os, json, torch, gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import io, contextlib

from src.config import read_config, resolve_device, model_summary
from src.tokenizer import Tokenizer
from src.model import CausalLSTM
from src.inference import generate, predict_next_word
from src.preprocess import text_cleaner   # your cleaner

# --- Utility functions ---
def list_runs(models_dir="models"):
    runs = [d for d in os.listdir(models_dir) if d.startswith("CausalLSTM_run")]
    print(f"[DEBUG] Found runs in {models_dir}: {runs}")
    return runs

def load_run(run_name, models_dir="models"):
    print(f"[DEBUG] Loading run: {run_name}")
    run_dir = os.path.join(models_dir, run_name)
    print(f"[DEBUG] Run directory: {run_dir}")

    device = resolve_device()
    print(f"[DEBUG] Resolved device: {device}")

    config = read_config(os.path.join(run_dir, "config.yaml"))
    print(f"[DEBUG] Loaded config keys: {list(config.keys())}")

    with open(os.path.join(run_dir, "training_logs.json")) as f:
        train_logs = json.load(f)
    print(f"[DEBUG] Training logs keys: {list(train_logs.keys())}")

    tokenizer = Tokenizer.load_from_file(os.path.join(run_dir, "tokenizer.json"))
    print(f"[DEBUG] Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    model = CausalLSTM(tokenizer.get_vocab_size(), **config["model_params"]).to(device)
    print(f"[DEBUG] Instantiated model: {model.__class__.__name__}")

    state_dict = torch.load(os.path.join(run_dir, "CausalLSTM_ckpnt.pt"), map_location=device)
    print(f"[DEBUG] Loaded state_dict with {len(state_dict)} keys")

    model.load_state_dict(state_dict)
    print("[DEBUG] Model state_dict loaded successfully")

    return run_dir, device, config, train_logs, tokenizer, model

def plot_logs(train_logs):
    print("[DEBUG] Plotting training logs...")
    fig, ax = plt.subplots(1, 3, figsize=(15,4))

    ax[0].plot(train_logs["train_loss"], label="train")
    if train_logs.get("val_loss") and len(train_logs["val_loss"])>0:
        ax[0].plot(train_logs["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()

    if train_logs.get("val_metric") and len(train_logs["val_metric"])>0:
        ax[1].plot(train_logs["val_metric"], label="val metric")
        ax[1].set_title("Validation Metric")

    ax[2].plot(train_logs["lr"], label="lr")
    ax[2].set_title("Learning Rate")

    buf = BytesIO()
    plt.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0)
    print("[DEBUG] Training plots generated")
    return Image.open(buf)

def capture_model_summary(model):
    print("[DEBUG] Capturing model summary...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model_summary(model)
    summary = buf.getvalue()
    print("[DEBUG] Model summary captured")
    return summary

# --- Gradio callbacks ---
current = {}

def select_run(run_name):
    print(f"[DEBUG] select_run called with: {run_name}")
    run_dir, device, config, train_logs, tokenizer, model = load_run(run_name)
    current.update(dict(run_dir=run_dir, device=device, config=config,
                        train_logs=train_logs, tokenizer=tokenizer, model=model))
    print("[DEBUG] Current run context updated")

    info = f"""
### Run Info
- Training mode: **{config['training_mode']}**
- Seq length: **{config['seq_len']}**
- Train data: **{os.path.basename(config['train_data_path'])}**
- Vocab size: **{tokenizer.get_vocab_size()}**
- Epochs trained: **{len(train_logs['train_loss'])}**
- Pretrained embeddings: **{config['use_glove_embeddings']}**
"""
    print("[DEBUG] Info section prepared")

    model_str = str(model)
    summary_str = capture_model_summary(model)
    print("[DEBUG] Model structure and summary prepared")

    return info, model_str, summary_str, plot_logs(train_logs)

def do_generate(init_word, max_tokens, temp, seed_text):
    print(f"[DEBUG] do_generate called with init_word={init_word}, max_tokens={max_tokens}, temp={temp}, seed_text={seed_text}")
    seed = None
    if seed_text and str(seed_text).strip() != "":
        try:
            seed = int(seed_text)
            print(f"[DEBUG] Parsed seed: {seed}")
        except:
            print("[DEBUG] Failed to parse seed, using None")
            seed = None
    output = generate(current["model"], current["tokenizer"], current["config"],
                      current["device"], init_word, max_tokens, temp, seed)
    print("[DEBUG] Generation complete")
    return output

def do_predict(context):
    print(f"[DEBUG] do_predict called with context='{context}'")
    preds = predict_next_word(current["model"], current["tokenizer"], current["config"],
                              current["device"], context, top_k=5)
    print(f"[DEBUG] Predictions: {preds}")
    return preds

# --- Build Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 📚 CausalLSTM Inference & Stats Dashboard")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Select a Run")
            run_dropdown = gr.Dropdown(choices=list_runs(), label="Available Runs")
            info_out = gr.Markdown()
            model_out = gr.Textbox(label="Model Structure", lines=15)
            summary_out = gr.Textbox(label="Model Summary", lines=15)
            plot_out = gr.Image(type="pil", label="Training Logs")
            run_dropdown.change(select_run, run_dropdown, [info_out, model_out, summary_out, plot_out])

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("## 2. Text Generation")
                init_word = gr.Textbox(label="Initial word (optional)")
                max_tokens = gr.Slider(10, 200, value=32, label="Max new tokens")
                temp = gr.Slider(0.1, 2.0, value=0.9, label="Temperature")
                seed_text = gr.Textbox(label="Seed (leave empty for None)")
                gen_btn = gr.Button("Generate")
                gen_out = gr.Textbox(label="Generated Text", lines=10)
                gen_btn.click(do_generate, [init_word, max_tokens, temp, seed_text], gen_out)

            with gr.Group():
                gr.Markdown("## 3. Next Word Prediction")
                context = gr.Textbox(label="Context words", value="it is")
                pred_btn = gr.Button("Predict Next Word")
                pred_out = gr.Label(num_top_classes=5)
                pred_btn.click(do_predict, context, pred_out)

demo.launch()
