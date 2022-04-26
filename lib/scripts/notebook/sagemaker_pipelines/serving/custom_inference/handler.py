from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler
from sagemaker_inference.transformer import Transformer


import textwrap
from sagemaker_inference import encoder, decoder

class HandlerService(DefaultHandlerService, DefaultInferenceHandler):
    def __init__(self):        
        transformer = Transformer(default_inference_handler=self)
        super(HandlerService, self).__init__(transformer=transformer)
    
    def default_model_fn(self, model_dir):
        """A default function to load a model cannot be provided.
        Users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.

        Returns: A PyTorch model.
        """
        
        raise NotImplementedError(textwrap.dedent("""
        Please provide a model_fn implementation.
        See documentation for model_fn at https://github.com/aws/sagemaker-python-sdk
        """))

    def default_input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
        """
        return decoder.decode(input_data, content_type)

    def default_predict_fn(self, input_data, model):
        """A default predict_fn. Calls a model on data deserialized in input_fn.        

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn

        Returns: a prediction
        """
        return model(input_data)

    def default_output_fn(self, prediction, accept):
        """A default output_fn. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        return encoder.encode(prediction, accept)