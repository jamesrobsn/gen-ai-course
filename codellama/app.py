import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json",
}

history = []

def generate_response(prompt):
    history.append(prompt)
    final_prompt = "\n".join(history)

    data = {
        "model": "codemaster",
        "prompt": final_prompt,
        "stream": False
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        res_text = response.text
        text = json.loads(res_text)
        actual_response = text.get("response", "")
        return actual_response
    else:
        return f"Error: {response.status_code} - {response.text}"

interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your prompt here", placeholder="Type your question or request..."),
    outputs=gr.Textbox(label="Response"),
    title="CodeLlama Chat Interface",
    description="This interface allows you to interact with the CodeLlama model. Type your prompt and get a response.",
    # theme="default",
    # allow_flagging="never",
    # examples=[
    #     ["What is the capital of France?"],
    #     ["Write a Python function to calculate the factorial of a number."],
    #     ["Explain the concept of recursion in programming."],
    #     ["How do I create a REST API using Flask?"],
    #     ["What are the best practices for writing clean code?"]
    # ]
)

# To Access WSL 
# hostname -I
# http://172.17.169.64:7860
interface.launch(server_name="0.0.0.0")
# interface.launch() # Normally