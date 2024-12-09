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
        gr.Markdown("""
        **Ця система призначена для отримання відповідей на запитання, пов'язані з науковими статтями на тему візуальної одометрії.**  
        У системі є 15 наукових статей, збережених у форматі CSV та 15 відповідних PDF-файлів.  
        Ви можете вводити запити або натиснути на кнопку для надсилання вже готового питання, для отримання інформації з цих статей.

        """)

        api_key_input = gr.Textbox(
            label="Enter your API KEY for GROQ",
            placeholder="Your API KEY",
            lines=1,
            interactive=True
        )
        
        example_queries = ["What is doi number of \"Comprehensive Performance Evaluation between Visual SLAM and LiDAR SLAM for Mobile Robots: Theories and Experiments\" ?",
                           "How many and who are the authors of the article \"LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping\"",
                           "Write me something about SLAM and VSLAM. Write a difference between it", "Abbreviations in the article \" An Overview on Visual SLAM: From Tradition to Semantic\""]
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
