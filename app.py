import os, torch
import gradio as gr

from src.app_utils import list_runs, load_run, plot_training_logs
from src.inference import generate, predict_next_word
from src.config import model_summary


#==================#
#     Handlers     #
#==================#

def find_runs_in_dir(models_dir: str):
    try:
        runs = list_runs(models_dir)
        return gr.update(choices=runs, value=runs[0] if runs else None)
    except Exception:
        return gr.update(choices=[], value=None)


def refresh_runs(models_dir: str):
    try:
        runs = list_runs(models_dir)
        return gr.update(choices=runs, value=runs[0] if runs else None)
    except Exception:
        return gr.update(choices=[], value=None)


def on_select_run(run_path: str):
    if not run_path:
        return None, None, None, None, None, None, None, None

    out = load_run(run_path)
    if out is None:
        return None, None, None, None, None, None, None, None

    model, tokenizer, config, device, training_logs = out

    fig = plot_training_logs(training_logs)

    info_md = f"""
    #### <center>---=( RUN INFORMATION )=---
    
    > 📚 **Training Dataset:**  
    > {os.path.basename(config["train_data_path"])}
    
    > ⚙️ **Training Mode:** {config["training_mode"].capitalize()}
    
    > 🔢 **Sequence Length:**  {config["seq_len"]}
    
    > 📈 **Training Epochs:**  {len(training_logs["train_loss"])}
    
    > 🌀 **RNN Type:**  {config["model_params"]["rnn_type"]}
    """

    summary_text = f"```\n{model_summary(model)}\n```"
    
    return (
        model, tokenizer, config, device, training_logs,
        info_md, fig, summary_text
    )


def on_generate_text(model, tokenizer, config, device, init_word, temperature, max_new_tokens):
    if not all([model, tokenizer, config, device]):
        return "Load a run first."
    try:
        return generate(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            init_word=init_word.strip() if init_word else None,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
        )
    except Exception as e:
        return f"Generation failed: {e}"
        
#===================#
#     Gradio UI     #
#===================#

def get_app():
    with gr.Blocks(
        theme=gr.themes.Origin(
            text_size=gr.themes.sizes.text_md,
            primary_hue=gr.themes.colors.rose,
            secondary_hue=gr.themes.colors.yellow,
            neutral_hue=gr.themes.colors.violet,
        ),
        title = "WordFlow"
    ) as app:
        gr.Markdown("# *WordFlow Experiments Dashboard*")
        gr.Markdown("#### <center> **WordFlow** is a Word-Level Languae Modeling Project using"
                    " Recurrent Neural Networks <a href='https://GiTHUB.com/HooM4N/WordFlow'>"
                    "(GiTHUB.com/HooM4N/WordFlow)</a>", container=True)
    
        # States
        st_model = gr.State()
        st_tokenizer = gr.State()
        st_config = gr.State()
        st_device = gr.State()
        st_logs = gr.State()
    
        # 1st Row
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                models_dir = gr.Textbox(label="Models directory", value="models/", lines=1)
                find_runs_btn = gr.Button("Find runs", variant="primary")
                run_dropdown = gr.Dropdown(label="Select run", choices=[], value=None)
                refresh_dir_btn = gr.Button("Refresh directory", variant="secondary")
            with gr.Column(scale=3):
                train_plot = gr.Plot(label="Training Plots")
    
        # 2nd Row
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                info_box = gr.Markdown(label="Run info", container=True)
            with gr.Column(scale=3):
                summary_box = gr.Markdown(label="Model Summary", container=False)
    
        gr.Markdown("---")
    
        # Text generation tab
        with gr.Tab("Text Generation"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    init_word = gr.Textbox(value ="<bos>", label="Initial word", placeholder="e.g., <bos>")
                    temperature = gr.Slider(label="Temperature", minimum=0.4, maximum=1.5, value=0.7, step=0.05)
                    max_new_tokens = gr.Slider(label="Max new tokens", minimum=5, maximum=200, value=100, step=1)
                    gen_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=2):
                    gen_output = gr.Textbox(label="Generated Text", interactive=False, lines=5)
    
        # Next word prediction tab
        with gr.Tab("Next Word Prediction"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    context_in = gr.Textbox(label="Context", placeholder="Type a short phrase...")
                    predict_btn = gr.Button("Predict", variant="primary")
                with gr.Column(scale=2):
                    prob_output = gr.Label(num_top_classes=5)
    
        # Events
        find_runs_btn.click(fn=find_runs_in_dir, inputs=models_dir, outputs=run_dropdown)
        refresh_dir_btn.click(fn=refresh_runs, inputs=models_dir, outputs=run_dropdown)
    
        run_dropdown.change(
            fn=on_select_run,
            inputs=run_dropdown,
            outputs=[
                st_model, st_tokenizer, st_config, st_device, st_logs,
                info_box, train_plot, summary_box
            ]
        )
    
        gen_btn.click(
            fn=on_generate_text,
            inputs=[st_model, st_tokenizer, st_config, st_device, init_word, temperature, max_new_tokens],
            outputs=gen_output
        )
    
        predict_btn.click(
            fn=predict_next_word,
            inputs=[st_model, st_tokenizer, st_config, st_device, context_in],
            outputs=prob_output
        )
    return app

if __name__ == "__main__":
    app = get_app()
    app.launch()