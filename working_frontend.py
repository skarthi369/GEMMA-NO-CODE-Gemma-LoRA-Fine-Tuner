"""
Working Frontend - Connects to Backend at http://localhost:8000
"""

import gradio as gr
import requests

API_URL = "http://localhost:8000"

def test_backend():
    """Test if backend is running"""
    try:
        response = requests.get(f"{API_URL}/")
        return f"✅ Backend Connected!\n\n{response.json()}"
    except:
        return "❌ Backend not running. Start it with: python simple_backend.py"

def upload_file(file):
    """Upload file to backend"""
    if not file:
        return "Please select a file"
    
    try:
        with open(file.name, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/api/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            return f"✅ Uploaded!\n\nID: {data['dataset_id']}\nSize: {data['file_size_human']}"
        else:
            return f"❌ Error: {response.json()}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create UI
with gr.Blocks(title="Gemma Fine-Tuner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Gemma Fine-Tuner\n\n**Backend**: http://localhost:8000")
    
    with gr.Tab("Status"):
        test_btn = gr.Button("Test Backend Connection")
        status_output = gr.Textbox(label="Status", lines=5)
        test_btn.click(test_backend, outputs=status_output)
    
    with gr.Tab("Upload"):
        upload_file_input = gr.File(label="Select Dataset")
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Result", lines=5)
        upload_btn.click(upload_file, inputs=upload_file_input, outputs=upload_output)
    
    gr.Markdown("""
    ---
    ### Next Steps:
    1. Test backend connection above
    2. Upload a dataset file
    3. For REAL training, install: `pip install -r requirements.txt`
    """)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Frontend")
    print("=" * 60)
    print("Frontend: http://localhost:7860")
    print("Backend:  http://localhost:8000")
    print("=" * 60)
    
    demo.launch(server_name="0.0.0.0", server_port=7860)
