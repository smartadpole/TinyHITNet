import onnxruntime
nnx_session = onnxruntime.InferenceSession("hitnet_sim960_320.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider','CPUExecutionProvider'])