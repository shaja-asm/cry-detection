# import ctypes
# import os

# # Paths to the shared library and model
# shared_lib_path = "/home/garfield/cry-detection/libtensorflowlite_c_2_14_1_amd64.so"
# model_path = b"/home/garfield/cry-detection/tflite_models/cnn_cry_detection_model_quant.tflite"

# # Check if the shared library exists
# if not os.path.exists(shared_lib_path):
#     print(f"Shared library not found at {shared_lib_path}")
#     exit(1)

# # Check if the model file exists
# if not os.path.exists(model_path.decode()):  # Decode to string for logging
#     print(f"Model file not found at {model_path.decode()}")
#     exit(1)

# try:
#     # Load the shared library
#     print("Loading shared library...")
#     lib = ctypes.cdll.LoadLibrary(shared_lib_path)
#     print("Shared library loaded successfully.")

#     # Define required C API functions
#     lib.TfLiteModelCreate.restype = ctypes.c_void_p
#     lib.TfLiteModelCreate.argtypes = [ctypes.c_char_p]

#     lib.TfLiteInterpreterOptionsCreate.restype = ctypes.c_void_p
#     lib.TfLiteInterpreterOptionsCreate.argtypes = []

#     lib.TfLiteInterpreterOptionsSetNumThreads.argtypes = [ctypes.c_void_p, ctypes.c_int]

#     lib.TfLiteInterpreterCreate.restype = ctypes.c_void_p
#     lib.TfLiteInterpreterCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

#     lib.TfLiteInterpreterAllocateTensors.argtypes = [ctypes.c_void_p]
#     lib.TfLiteInterpreterAllocateTensors.restype = ctypes.c_int

#     lib.TfLiteInterpreterDelete.argtypes = [ctypes.c_void_p]

#     # Load the TFLite model
#     print("Loading TFLite model...")
#     model = lib.TfLiteModelCreate(model_path)
#     if not model:
#         raise RuntimeError("Failed to load TFLite model.")
#     print("TFLite model loaded successfully.")

#     # Create interpreter options
#     print("Creating interpreter options...")
#     options = lib.TfLiteInterpreterOptionsCreate()
#     lib.TfLiteInterpreterOptionsSetNumThreads(options, 2)
#     print("Interpreter options created successfully.")

#     # Create the interpreter
#     print("Creating interpreter...")
#     interpreter = lib.TfLiteInterpreterCreate(model, options)
#     if not interpreter:
#         raise RuntimeError("Failed to create TFLite interpreter.")
#     print("Interpreter created successfully.")

#     # Allocate tensors
#     print("Allocating tensors...")
#     status = lib.TfLiteInterpreterAllocateTensors(interpreter)
#     if status != 0:
#         raise RuntimeError(f"Tensor allocation failed with status: {status}")
#     print("Tensors allocated successfully.")

#     # Clean up
#     print("Cleaning up...")
#     lib.TfLiteInterpreterDelete(interpreter)
#     print("Interpreter deleted successfully.")
#     lib.TfLiteInterpreterOptionsDelete(options)
#     print("Options deleted successfully.")
#     lib.TfLiteModelCreate.argtypes = [ctypes.c_char_p]
#     print("Model deleted successfully.")

# except Exception as e:
#     print(f"Error during interpreter verification: {e}")


import tensorflow as tf

tflite_model_path = "/home/garfield/cry-detection/tflite_models/cnn_cry_detection_model_quant.tflite"  # Replace with the actual file path

try:
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Error validating TFLite model: {e}")

