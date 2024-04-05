import gradio as gr

from predict_caption import predict_step

with gr.Blocks() as demo:
    text = gr.Textbox(label='Purpose is to : ')
    image = gr.Image(type='pil', label='Image')
    label = gr.Text(label='Generated Caption')

    if text!="":
        image.upload(
        predict_step,
        [image,text],
        [label])

if __name__ == '__main__':
    demo.launch()
