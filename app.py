import gradio as gr
import torch
from src.model import Config,GPT
from src.inference import GPTInfer
import tiktoken
import torch
import os

os.environ['GRADIO_DEFAULT_CONCURRENCY_LIMIT']="1"

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'   
print(f'using device: {device}')
model_path = './model_02399.pt'
checkpoint = torch.load(model_path, weights_only=False)
model = GPT(config=checkpoint['config'])
model.load_state_dict(checkpoint['model'])
model = model.to(device)
token_encoder = tiktoken.get_encoding('gpt2')
generator = GPTInfer(model, token_encoder, device)

def generate_story(
    prompt,
    max_new_tokens=50,
    seed=42,
    temperature=0.8,
    top_k=None,
    top_p=0.9,
    repetition_penalty=1.2,
    frequency_penalty=0.6,
    no_repeat_ngram_size=3,
    longer_story=True,
    context_window=512
):
    if not prompt.strip():
        return prompt, gr.update()
    
    if top_k <= 0:
        top_k = None
    
    output_text = prompt
    last_piece = ""
    # print(f'{prompt}',end='',flush=True)
    yield gr.update(value=output_text,interactive=False), gr.update(interactive=False)  
    for piece in generator.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        longer_story=longer_story,
        context_window=context_window
    ):
        if piece == last_piece:
            continue
        last_piece = piece
        output_text += piece
        # print(f'{piece}',end='',flush=True)
        yield output_text, gr.update(interactive=False)  
    
    yield gr.update(value=output_text,interactive=True), gr.update(interactive=True)

with gr.Blocks(title="Story Generator") as demo:

    gr.Markdown("# ✨ Story Generator ✨")
    gr.Markdown(
        "Ketik prompt atau cerita awal di bawah ini. "
        "Tekan **Generate** untuk melanjutkan cerita. "
        "Anda dapat mengedit hasil cerita dan generate lagi untuk melanjutkan."
    )

    story_box = gr.Textbox(
        label="Story / Prompt",
        lines=15,
        placeholder="Tulis prompt atau awal cerita di sini...",
    )

    generate_btn = gr.Button("Generate Story", variant="primary")

    with gr.Accordion("Generation Settings", open=False):
        context_window = gr.Slider(
            minimum=128,
            maximum=2048,
            value=512,
            step=64,
            label="Context Window (tokens to use from end of text)",
            info="Limits how much previous text is used. Lower = faster but less context."
        )
        
        max_new_tokens = gr.Slider(
            minimum=20,
            maximum=2048,
            value=1024,
            step=10,
            label="Max New Tokens"
        )    

        seed = gr.Number(
        value=42, 
        label="Seed"
        )

        temperature = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.8,
            step=0.05,
            label="Temperature"
        )

        top_k = gr.Slider(
            minimum=0, 
            maximum=200, 
            value=0, 
            step=1, 
            label="Top-K (0 = disabled)"
        )

        top_p = gr.Slider(
            minimum=0.0, 
            maximum=1.0, 
            value=0.9, 
            step=0.01, 
            label="Top-P"
        )

        repetition_penalty = gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.2,
            step=0.05,
            label="Repetition Penalty"
        )

        frequency_penalty = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.6,
            step=0.05,
            label="Frequency Penalty"
        )

        no_repeat = gr.Slider(
            minimum=1, 
            maximum=10, 
            value=3, 
            step=1, 
            label="No-Repeat N-gram Size"
        )
    

    generate_btn.click(
        fn=generate_story,
        inputs=[
            story_box, 
            max_new_tokens, 
            seed, 
            temperature, 
            top_k, 
            top_p, 
            repetition_penalty, 
            frequency_penalty, 
            no_repeat,
            gr.Checkbox(value=True, visible=False),  
            context_window  
        ],
        outputs=[story_box, generate_btn]
    )

#Run App
if __name__ == "__main__":
    
    demo.launch(share=False)
