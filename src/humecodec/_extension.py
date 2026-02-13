_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        from humecodec import _humecodec

        _ext = _humecodec
        _ext.init()
        _ext.set_log_level(16)  # AV_LOG_ERROR — suppress noisy warnings from decoders
    return _ext


def get_ext():
    return _load_ext()
