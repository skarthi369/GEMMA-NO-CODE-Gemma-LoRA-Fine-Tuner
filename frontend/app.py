"""
Gradio Frontend for Gemma LoRA Fine-Tuner

No-code web interface for dataset upload, training configuration,
progress monitoring, and model export.
"""

import os
import time
import requests
import gradio as gr
from pathlib import Path
from typing import Optional, Dict, Any
import json

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class GemmaFineTunerUI:
    """Gradio UI for Gemma fine-tuning"""
    
    def __init__(self):
        self.current_job_id = None
        self.uploaded_dataset_id = None
    
    def upload_dataset(self, file, progress=gr.Progress()):
        """Upload dataset file to backend"""
        if file is None:
            return "❌ Please select a file to upload"
        
        progress(0, desc="Uploading dataset...")
        
        try:
            with open(file.name, 'rb') as f:
                files = {'file': (Path(file.name).name, f)}
                response = requests.post(f"{API_BASE_URL}/api/upload", files=files)
            
            if response.status_code == 200:
                data = response.json()
                self.uploaded_dataset_id = data['dataset_id']
                
                message = f"""
✅ Dataset uploaded successfully!

**Dataset ID:** `{data['dataset_id']}`
**Filename:** {data['filename']}
**Size:** {data['file_size_human']}
**Format:** {data['format'].upper()}

You can now proceed to the **Training** tab to start fine-tuning.
                """
                return message
            else:
                return f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}"
        
        except Exception as e:
            return f"❌ Upload error: {str(e)}"
    
    def start_training(
        self,
        model_variant,
        lora_r,
        lora_alpha,
        epochs,
        batch_size,
        learning_rate,
        progress=gr.Progress()
    ):
        """Start training job"""
        if not self.uploaded_dataset_id:
            return "❌ Please upload a dataset first in the Upload tab"
        
        progress(0, desc="Starting training job...")
        
        try:
            # Prepare training request
            request_data = {
                "dataset_id": self.uploaded_dataset_id,
                "model_variant": model_variant,
                "lora_config": {
                    "r": lora_r,
                    "alpha": lora_alpha,
                    "dropout": 0.0,
                    "bias": "none"
                },
                "training_config": {
                    "num_train_epochs": epochs,
                    "per_device_train_batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "gradient_accumulation_steps": 4,
                    "logging_steps": 10,
                    "save_steps": 100
                },
                "max_seq_length": 2048
            }
            
            response = requests.post(f"{API_BASE_URL}/api/train", json=request_data)
            
            if response.status_code == 200:
                data = response.json()
                self.current_job_id = data['job_id']
                
                message = f"""
🚀 Training started successfully!

**Job ID:** `{data['job_id']}`
**Job Name:** {data['job_name']}
**Status:** {data['status']}

Monitor progress in the **Progress** tab below.
                """
                return message
            else:
                return f"❌ Training failed to start: {response.json().get('detail', 'Unknown error')}"
        
        except Exception as e:
            return f"❌ Error starting training: {str(e)}"
    
    def get_progress(self):
        """Get current training progress"""
        if not self.current_job_id:
            return {
                "status": "No active training",
                "progress": 0,
                "details": "Start a training job to see progress here"
            }
        
        try:
            response = requests.get(f"{API_BASE_URL}/api/progress/{self.current_job_id}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Format progress display
                status = data['status']
                progress_pct = data.get('progress_percent', 0)
                current_step = data.get('current_step', 0)
                total_steps = data.get('total_steps', 0)
                current_loss = data.get('current_loss')
                eta = data.get('estimated_time_remaining_seconds')
                
                details = f"""
**Status:** {status}
**Progress:** {progress_pct:.1f}% ({current_step}/{total_steps} steps)
**Current Loss:** {current_loss:.4f if current_loss else 'N/A'}
**ETA:** {eta//60 if eta else 'Calculating...'}min
                """
                
                return {
                    "status": status,
                    "progress": progress_pct,
                    "details": details.strip()
                }
            else:
                return {
                    "status": "Error",
                    "progress": 0,
                    "details": "Failed to fetch progress"
                }
        
        except Exception as e:
            return {
                "status": "Error",
                "progress": 0,
                "details": f"Error: {str(e)}"
            }
    
    def create_ui(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="Gemma LoRA Fine-Tuner", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🚀 Gemma LoRA Fine-Tuner
            ### Production No-Code Platform for Fine-Tuning Gemma Models
            
            **Powered by Unsloth** for 2x faster training with 60% less memory
            """)
            
            with gr.Tabs():
                # ===== UPLOAD TAB =====
                with gr.Tab("📤 Upload Dataset"):
                    gr.Markdown("""
                    ### Upload Your Training Dataset
                    
                    **Supported Formats:**
                    - **CSV:** Must have a 'text' column (and optionally 'label')
                    - **JSON:** Array of objects with 'text' field
                    - **JSONL:** One JSON object per line
                    - **TXT:** One training sample per line
                    
                    **Max Size:** 500MB
                    """)
                    
                    with gr.Row():
                        upload_file = gr.File(label="Select Dataset File", file_types=[".csv", ".json", ".jsonl", ".txt"])
                    
                    upload_btn = gr.Button("📤 Upload Dataset", variant="primary", size="lg")
                    upload_output = gr.Markdown(label="Upload Status")
                    
                    upload_btn.click(
                        fn=self.upload_dataset,
                        inputs=[upload_file],
                        outputs=[upload_output]
                    )
                
                # ===== TRAINING TAB =====
                with gr.Tab("⚙️ Configure & Train"):
                    gr.Markdown("""
                    ### Configure Training Parameters
                    
                    Adjust the settings below to customize your fine-tuning job.
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            model_variant = gr.Dropdown(
                                choices=["gemma-2b", "gemma-7b", "gemma-2b-it", "gemma-7b-it"],
                                value="gemma-2b",
                                label="Model Variant",
                                info="Gemma-2B recommended for 8GB VRAM"
                            )
                            
                            gr.Markdown("#### LoRA Configuration")
                            lora_r = gr.Slider(4, 64, value=16, step=4, label="LoRA Rank (r)", info="Higher = more parameters")
                            lora_alpha = gr.Slider(4, 128, value=16, step=4, label="LoRA Alpha", info="Usually equal to rank")
                        
                        with gr.Column():
                            gr.Markdown("#### Training Hyperparameters")
                            epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs", info="Number of training passes")
                            batch_size = gr.Slider(1, 8, value=2, step=1, label="Batch Size", info="Reduce if OOM")
                            learning_rate = gr.Number(value=2e-4, label="Learning Rate", info="2e-4 is recommended")
                    
                    train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                    train_output = gr.Markdown(label="Training Status")
                    
                    train_btn.click(
                        fn=self.start_training,
                        inputs=[model_variant, lora_r, lora_alpha, epochs, batch_size, learning_rate],
                        outputs=[train_output]
                    )
                
                # ===== PROGRESS TAB =====
                with gr.Tab("📊 Training Progress"):
                    gr.Markdown("""
                    ### Real-Time Training Progress
                    
                    Monitor your training job in real-time.
                    """)
                    
                    progress_status = gr.Textbox(label="Status", interactive=False)
                    progress_bar = gr.Slider(0, 100, value=0, label="Progress (%)", interactive=False)
                    progress_details = gr.Markdown(label="Details")
                    
                    refresh_btn = gr.Button("🔄 Refresh Progress")
                    
                    def update_progress_ui():
                        data = self.get_progress()
                        return data["status"], data["progress"], data["details"]
                    
                    refresh_btn.click(
                        fn=update_progress_ui,
                        outputs=[progress_status, progress_bar, progress_details]
                    )
                
                # ===== EXPORT TAB =====
                with gr.Tab("💾 Export Model"):
                    gr.Markdown("""
                    ### Export Your Fine-Tuned Model
                    
                    Download your LoRA adapters or merged model.
                    """)
                    
                    export_info = gr.Markdown("""
                    **Coming soon:** Export functionality will be added here.
                    
                    You can find your trained models in the `exports/` directory.
                    """)
            
            gr.Markdown("""
            ---
            **Built with ❤️ using Gradio, FastAPI, and Unsloth**
            """)
        
        return demo


def main():
    """Launch Gradio UI"""
    ui = GemmaFineTunerUI()
    demo = ui.create_ui()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
