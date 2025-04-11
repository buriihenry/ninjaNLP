import gradio as gr

from theme_classifier import ThemeClassifier

def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(",")
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove dialogue from theme list
    theme_list = [theme for theme in theme_list if theme != "dialogue" ]
    output_df = output_df[theme_list]

    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme','score']

    output_chart = gr.BarPlot(output_df, x="theme", y="score", title="Theme Classification", tooltip=["Theme","score"],vertical=False, width=500, height=260)
    return output_chart

    
def main():
    try:
        with gr.Blocks() as iface:
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                    with gr.Row():
                        with gr.Column():
                            plot = gr.BarPlot()
                        with gr.Column():
                            theme_list = gr.Textbox(label="Themes")
                            subtitles_path = gr.Textbox(label="Subtitles or script Path")
                            save_path = gr.Textbox(label="Save Path")
                            get_themes_btn = gr.Button("Get Themes")
                            get_themes_btn.click(
                                fn=get_themes,
                                inputs=[theme_list,subtitles_path,save_path],
                                outputs=[plot]
                            )
                            
        iface.launch(share=True)
    except AttributeError as e:
        print(f"Error: {e}")
        print("\nTry reinstalling Gradio with: pip install gradio==4.36.1")
        print("Or check if you're using the correct import statement: import gradio as gr")

if __name__ == "__main__":
    main()