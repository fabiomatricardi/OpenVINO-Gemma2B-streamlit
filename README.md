# OpenVINO-Gemma2B-streamlit

Using OpenVINO with Gemma2-2B INT4 and streamlit CHAT APP
> This streamlit application is a ChatBot using OpenVINO as an AI framework
> OpenVINO is amazingly fast and useful on Intel based Chips, integrated Intel Graphics GPUS and Intel GPUs
> The models can be quantized so you can load also models up to 7B parameters with 16GB of RAM

---

> note that OpenVINO can be used to quantize and run also diffusers models like StableDiffusion and others
> You can also use whisper and TEXT-2-TEXT encoder-decoder models, that so far are not fully supported with llamaCPP


### The final result
<img src='https://github.com/fabiomatricardi/OpenVINO-Gemma2B-streamlit/blob/main/interface.png' width=900>




### Instructions
Works with Python 3.11+, tested on Windows 11

Clone the Repo, so that you will also get the images assets

In the main repo folder create a `venv` and install the dependencies
```
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install openvino-genai==2024.3.0
pip install optimum-intel[openvino] tiktoken streamlit==1.36.0
```

### Origianl Model
Download the model files from HuggingFace repo

[sabre-code/gemma-2-2b-it-openvino-int4
](https://huggingface.co/sabre-code/gemma-2-2b-it-openvino-int4)

Download every single files into a subfolder called `model`
>If you cloned the repo you sill find the subfolder already there for you


### Run the app
After that in the terminal, with `venv` activated run
```
streamlit run .\stappFULL.py
```










