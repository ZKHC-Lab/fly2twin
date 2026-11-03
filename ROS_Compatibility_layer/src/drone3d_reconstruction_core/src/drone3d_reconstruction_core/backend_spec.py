"""Backend identity and command configuration; deliberately imports no CUDA."""


def validate_independent_request(backend, frame_count, resolution):
    if backend not in ("omega",):
        raise ValueError("backend must be omega")


def validate_backend(config):
    if config.backend not in ("omega",):
        raise ValueError("backend must be omega")
