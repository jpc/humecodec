_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        from torchffmpeg import _torchffmpeg

        _ext = _torchffmpeg
        _ext.init()
    return _ext


def get_ext():
    return _load_ext()
