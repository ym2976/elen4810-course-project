from .config_manager import ConfigManager
from . import datasets
from . import evaluation
from . import feature_extraction
from . import harmonic_model
from . import synthesis
from . import visualization
from . import dataset_analysis


def initialize_config(config_path=None):
    """
    Initializes the configuration for InstruReconstr.

    The function will load the default configuration from the config.py file within the package
    if no custom configuration is provided. Otherwise, it will load the configuration from the
    provided file path.

    This function should be called before any other function in the package. If not clear on how
    to prepare a custom configuration file, please refer to the documentation on the GitHub page
    of the project: github.com/gilja/instruments_fourier_analysis/tree/main/utils

    Args:
        config_path (str, optional):
            -   Path to a custom configuration file.
    """

    ConfigManager.get_instance(config_path)
