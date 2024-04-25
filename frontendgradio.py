import gradio as gr
import requests
import textwrap

API_URL = "http://127.0.0.1:8000/process"

def summarize_documents(entity_name):
    if entity_name.strip():
        data = {"entity_name": entity_name}
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            summary = response.json()["summary"]
            return summary  
        else:
            return f"Error while summarizing. Status code: {response.status_code}"
    else:
        return "Please enter a valid name"

iface = gr.Interface(
    fn=summarize_documents,
    inputs=gr.Textbox(label="Please enter the name of the entity:"),
    outputs=gr.Markdown(),
    title="KYC Websearch",
    description="Please enter the name of the entity to have a KYC report."
)

iface.launch(share=True)




