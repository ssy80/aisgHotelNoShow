class ConfigKeyError(KeyError):
    """Raised when a required configuration key is missing in the config."""

    def __init__(self, key_path):
        self.key_path = " -> ".join(key_path)
        super().__init__(f"Missing required configuration key: {self.key_path}")
