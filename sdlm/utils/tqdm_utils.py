from threading import Lock

from tqdm.auto import tqdm as _base_tqdm


class _ManagedTqdm:
    """Managed tqdm wrapper with a shared position registry."""

    _active_positions = set()
    _lock = Lock()

    @classmethod
    def _acquire_position(cls, requested_position=None):
        with cls._lock:
            if requested_position is not None:
                cls._active_positions.add(requested_position)
                return requested_position

            position = 0
            while position in cls._active_positions:
                position += 1
            cls._active_positions.add(position)
            return position

    @classmethod
    def _release_position(cls, position):
        with cls._lock:
            cls._active_positions.discard(position)

    @classmethod
    def _build_params(cls, kwargs, position):
        params = {
            "position": position,
            "leave": position == 0,
            "dynamic_ncols": True,
        }
        params.update(kwargs)
        return params

    def __init__(self, iterable=None, **kwargs):
        requested_position = kwargs.get("position")
        self._position = self._acquire_position(requested_position)
        self._closed = False
        self._pbar = _base_tqdm(
            iterable,
            **self._build_params(kwargs, self._position),
        )

    def __iter__(self):
        try:
            for item in self._pbar:
                yield item
        finally:
            self.close()

    def __enter__(self):
        self._pbar.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._pbar.__exit__(exc_type, exc, tb)
        finally:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._pbar.close()
        finally:
            self._release_position(self._position)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._pbar, name)


def tqdm(iterable=None, **kwargs):
    """Automatic replacement for tqdm() with shared position management."""
    return _ManagedTqdm(iterable=iterable, **kwargs)


def tenumerate(iterable, start=0, **kwargs):
    """Automatic replacement for tenumerate() using the shared tqdm wrapper."""
    with tqdm(iterable, **kwargs) as pbar:
        for index, value in enumerate(pbar, start=start):
            yield index, value
