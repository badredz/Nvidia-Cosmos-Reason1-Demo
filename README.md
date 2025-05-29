# Cosmos-Reason1: Image and Video Reasoning App

This is a Gradio-based demo application showcasing [NVIDIA's Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) model for multi-modal reasoning on both **images** and **videos**. It leverages the powerful capabilities of the Cosmos model for answering questions, generating detailed descriptions, and analyzing visual content.

Model : [Nvidia-Cosmos-Reason1](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
---

## Features

* **Image Reasoning**: Upload an image and ask a question or provide a prompt to receive detailed responses from the model.
* **Video Reasoning**: Upload a video, and the app will sample frames and provide detailed responses based on visual content and user queries.
* **Advanced Parameters**: Customize model generation parameters including `max_new_tokens`, `temperature`, `top_p`, `top_k`, and `repetition_penalty`.
* **Example Inputs**: Preloaded image and video examples to help users get started quickly.
* **GPU-Accelerated**: Designed to run on GPU for fast inference using PyTorch and Transformers.

---

## Live Demo

Launch the app using Gradio with:

```bash
python app.py
```

Or deploy it on [Hugging Face Spaces](https://huggingface.co/spaces) using the provided code.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Nvidia-Cosmos-Reason1-Demo.git
cd Nvidia-Cosmos-Reason1-Demo
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> Ensure you have `ffmpeg`, `torch` with GPU support, and the required CUDA drivers installed.

---

## Usage

### Image Inference

1. Enter a query (e.g., "Describe the scene in detail").
2. Upload an image (JPG or PNG).
3. Click **Submit**.
4. View the model's reasoning output in the text box.

### Video Inference

1. Enter a query (e.g., "What actions are taking place?").
2. Upload a video file (MP4 format recommended).
3. Click **Submit**.
4. The app will sample frames and return the model's reasoning.

---

## File Structure

```
.
├── app.py               # Main application script
├── images/              # Example images
├── videos/              # Example videos
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

---

## Model

This demo uses:

* **Model**: `nvidia/Cosmos-Reason1-7B`
* **Processor**: `AutoProcessor` from Hugging Face Transformers
* **Pipeline**: Multi-modal vision-language inference using `Qwen2_5_VLForConditionalGeneration`

For more details on the model: [https://huggingface.co/nvidia/Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* [NVIDIA](https://www.nvidia.com/) for releasing Cosmos-Reason1-7B
* [Hugging Face](https://huggingface.co/) for providing easy access to models and tools
* [Gradio](https://www.gradio.app/) for the user interface
