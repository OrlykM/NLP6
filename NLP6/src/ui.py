import gradio as gr


def validate_api_key(api_key):
    if api_key:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


def generate_answer_gradio(api_key, query, method, rerank, assistant_generate_answer):
    if not api_key:
        return "No API key provided.", ""
    resp1, resp2 = assistant_generate_answer(api_key, query, method, rerank)
    return resp1, " ".join(i for i in resp2)


def populate_query(example):
    return example


def create_ui(assistant_generate_answer):
    with gr.Blocks() as interface:
        api_key_input = gr.Textbox(
            label="Enter your API KEY for GROQ",
            placeholder="Your API KEY",
            lines=1,
            interactive=True
        )
        query_input = gr.Textbox(
            label="Enter your query",
            placeholder="What is SLAM?",
            lines=2,
            interactive=False  
        )
        method_input = gr.Radio(
            choices=['BM25', 'Full', 'Semantic'],
            label="Choose Method",
            value='BM25'
        )
        rerank_input = gr.Checkbox(
            label="Enable Reranking",
            value=True
        )
        context_output = gr.Textbox(label="Generated Context")
        references_output = gr.Textbox(label="References")

        generate_button = gr.Button("Generate Answer")

        api_key_input.change(
            validate_api_key,
            inputs=[api_key_input],
            outputs=[query_input]
        )
        generate_button.click(
            lambda api_key, query, method, rerank: generate_answer_gradio(api_key, query, method, rerank, assistant_generate_answer),
            inputs=[api_key_input, query_input, method_input, rerank_input],
            outputs=[context_output, references_output]
        )

    return interface
