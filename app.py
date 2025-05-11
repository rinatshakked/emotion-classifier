import gradio as gr
import subprocess

def run_training():
    try:
        # Run train.py using subprocess and capture output
        result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
        # Return stdout if success, otherwise stderr
        return result.stdout if result.returncode == 0 else f"Error:\n{result.stderr}"
    except Exception as e:
        return f"Exception occurred:\n{str(e)}"

# Define a simple Gradio interface with one button
demo = gr.Interface(
    fn=run_training,
    inputs=[],
    outputs="text",
    title="Run Model Training",
    description="Click the button to execute train.py. This will use GPU if available."
)

demo.launch()
