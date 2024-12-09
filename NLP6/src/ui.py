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
        
        example_queries = ["What is doi number of \"Comprehensive Performance Evaluation between Visual SLAM and LiDAR SLAM for Mobile Robots: Theories and Experiments\" ?",
                           "How many and who are the authors of the article \"LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping\"", "According to article \"Direct Sparse Odometry\" when Vladlen Koltun received his PhD\",
                           "How called article from authors Moreno-Noguer, Lepetit, and Fua from reference 79 for the article \" Visual Odometry \"", "Return all articles names which have more than 100 references "]
        example_buttons = [
            gr.Button(example, size="sm") for example in example_queries
        ]

        query_input = gr.Textbox(
            label="Enter your query",
            placeholder="What is SLAM?",
            lines=2,
            interactive=False  
        )
        
        method_input = gr.Radio(
            choices=['BM25', 'Semantic', 'Full'],
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

        with gr.Column():
            api_key_input

        with gr.Row():
            for button in example_buttons:
                button

        with gr.Column():
            method_input
            rerank_input
            query_input
            generate_button
            context_output
            references_output

        for button, example in zip(example_buttons, example_queries):
            button.click(
                fn=lambda ex=example: ex,
                inputs=[],
                outputs=[query_input]
            )

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
