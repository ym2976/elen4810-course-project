"""
config_manager
==============

This module contains classes for managing the configuration of the package.

Public classes:
---------------

-   ConfigManager: A singleton class for managing the configuration of the package.
        *   If a configuration file is provided, it will be loaded.
        *   Otherwise, the default configuration file will be loaded.

Private classes:
----------------

-   _ConfigAlreadyInitializedError: Exception raised when trying to reinitialize a singleton
    configuration.
"""

import importlib.util
import os


class _ConfigAlreadyInitializedError(RuntimeError):
    """Exception raised when trying to reinitialize a singleton configuration."""

    def __init__(self, message="Configuration has already been initialized!"):
        self.message = message
        super().__init__(self.message)


class ConfigManager:
    """
    A singleton class for managing the configuration of the package.

    If a user provides a configuration file, it is loaded and used to override the default
    configuration. Otherwise, the default configuration is used.
    """

    _instance = None

    @staticmethod
    def get_instance(config_path=None):
        """Static access method."""

        if ConfigManager._instance is None:
            ConfigManager._instance = ConfigManager(config_path)
        return ConfigManager._instance

    def __init__(self, config_path):
        """
        Virtually private constructor.

        Args:
            config_path (str):
                -   The absolute path to the configuration file.
        """

        if ConfigManager._instance is not None:
            raise _ConfigAlreadyInitializedError()

        self.is_default_config = not config_path or not os.path.exists(config_path)
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        """
        Loads the configuration file.

        Args:
            config_path (str):
                -   The absolute path to the configuration file.
        """

        if config_path and os.path.exists(config_path):
            if not config_path.endswith(".py"):
                raise ValueError(
                    "Configuration file must be a Python (.py) file. Please check the documentation."
                )
            self.is_default_config = False
            return self.load_python_config(config_path)

        # Load default configuration from config.py within the package
        from . import (  # pylint: disable=import-outside-toplevel
            config as default_config,
        )

        return default_config

    def load_python_config(self, path):
        """
        Dynamically loads a Python file as a module.

        Args:
            path (str):
                -   The absolute path to the Python file to load.
        """

        module_spec = importlib.util.spec_from_file_location("user_config", path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(
                f"Could not load configuration from {path}. Please ensure it's a valid Python file."
            )

        user_config = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(user_config)
        return user_config
