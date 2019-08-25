from dart_fss_classifier.fs_search import attached_plugin
from dart_fss_classifier.classifier import generate_default_dataset_and_cnn_model, generate_dataset_and_cnn_model
__all__ = ['attached_plugin', 'generate_default_dataset_and_cnn_model', 'generate_dataset_and_cnn_model']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
