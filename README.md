# Text-to-Image-generator
  
## Image Generator screenshot sample

<img width="1198" height="612" alt="image" src="https://github.com/user-attachments/assets/a4d9d81c-fc3d-4170-952f-5dd8c2be9b73" />


## Project Overview
This is a local, offline text-to-image generation application built using Stable Diffusion v1.5 and Streamlit. This tool allows users to generate high-quality images from text descriptions with granular control over inference steps and guidance scale.

## Features
- Text-to-Image Generation: Utilizes the open-source `runwayml/stable-diffusion-v1-4` model.
- Hardware Optimization:Automatically detects CUDA (GPU) availability. Switches to `float16` precision for GPUs (fast) and `float32` for CPUs (compatible).
- Metadata Logging:Every generated image is saved with a companion JSON file containing the exact prompt and parameters used, ensuring reproducibility.
- Safety Checker: Integrated ethical filters to prevent the generation of NSFW or inappropriate content.

## Technology Stack
- Language: Python 3.10+
- UI Framework: Streamlit
- ML Framework: PyTorch
- Diffusion Library: Hugging Face `diffusers`

##  Setup & Installation

1. Clone the repository
   ```bash
   git clone (https://github.com/ch24b013-blip/Text-to-Image-generator)
   cd Text-to-Image-generator
   ```
2.Install dependencies
   ```bash
  pip install -r requirements.txt
  ```
3.Run the Application 
   ```bash
  python -m streamlit run app.py
```
 REQUIREMENTS:-
-torch>=2.0.0
-diffusers>=0.21.0
-transformers>=4.30.0
-accelerate>=0.20.0
-streamlit>=1.25.0
-scipy

## Hardware Documentation
This application supports two execution modes:-
Hardware              |Precision |Est. Time per Image  |Notes

-NVIDIA GPU (Recommended)|float16   |~5-10 seconds        |Requires CUDA toolkit installed. Optimized for VRAM usage.

-CPU (Fallback)          |float32   |~5-10 minutes        |Works on any standard laptop. Slower but fully functional.

## Prompt Engineering Guide
To get the best results, structure your prompts using this formula: [Subject] + [Action/Context] + [Art Style] + [Lighting/Tech Specs]

-Example: "A futuristic city (Subject) at sunset (Context), cyberpunk style (Style), 8k resolution, cinematic lighting (Specs)"

-Keywords to use: highly detailed, sharp focus, unreal engine 5 render, studio lighting.

## Ethical AI & Limitations
-Content Filtering: The model includes a safety checker that blacks out images detected as offensive or NSFW.

-Bias: As with all large-scale models trained on LAION-5B, the model may reflect societal biases found in the training data.

-Watermarking: Future improvements will include invisible watermarking to identify AI-generated content.
