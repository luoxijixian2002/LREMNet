# `models/__init__.py` intentionally imports nothing here so that
# every entry-point script can import concrete submodules directly
# (e.g. `from models.lremnet import LREMNet`), matching the official
# repository style and avoiding circular imports.
