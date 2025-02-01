import torch
import onnx
import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
onnx_model_path = "optimized_model.onnx"

def convert_to_onnx(model, onnx_path):
    dummy_input = torch.randint(0, 50256, (1, 50))
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=['input'], output_names=['output'])

convert_to_onnx(model, onnx_model_path)
print("Model converted to ONNX format.")
