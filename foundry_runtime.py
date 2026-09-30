"""Shared lazy Foundry Local SDK runtime."""

import config

_manager = None


def get_manager():
    global _manager
    if _manager is None:
        from foundry_local_sdk import Configuration, FoundryLocalManager

        FoundryLocalManager.initialize(Configuration(app_name=config.APP_NAME))
        _manager = FoundryLocalManager.instance
    return _manager
