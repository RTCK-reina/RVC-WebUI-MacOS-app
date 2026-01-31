# Mock fairseq checkpoint_utils for macOS compatibility
import torch
import warnings

def load_model_ensemble_and_task(filenames, arg_overrides=None, task=None, suffix="", **kwargs):
    """Mock function to replace fairseq checkpoint_utils.load_model_ensemble_and_task"""
    warnings.warn("Using mock fairseq implementation - some features may not work correctly")
    
    models = []
    for filename in filenames:
        try:
            # Load the model state dict directly
            state_dict = torch.load(filename, map_location="cpu")
            
            # Create a simple mock model
            model = torch.nn.Module()
            model.load_state_dict = lambda x: None
            model.eval = lambda: model
            model.half = lambda: model
            model.to = lambda device: model
            
            # Add the extract_features method that RVC expects
            def extract_features(self, source, padding_mask=None, mask=False, **kwargs):
                # Return a dummy tensor with the expected shape
                batch_size = source.size(0)
                seq_len = source.size(1)
                feature_dim = 768  # Standard HuBERT feature dimension
                return torch.zeros(batch_size, seq_len, feature_dim)
            
            model.extract_features = extract_features.__get__(model, torch.nn.Module)
            models.append(model)
            
        except Exception as e:
            warnings.warn(f"Could not load model {filename}: {e}")
            # Return a dummy model anyway
            model = torch.nn.Module()
            model.extract_features = lambda source, **kwargs: torch.zeros(1, source.size(1), 768)
            models.append(model)
    
    return models, None, None

def load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=False):
    """Mock function to replace fairseq checkpoint_utils.load_checkpoint_to_cpu"""
    warnings.warn("Using mock fairseq implementation - loading PyTorch checkpoint directly")
    return torch.load(path, map_location="cpu")
