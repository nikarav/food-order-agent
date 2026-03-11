from functools import reduce
from os.path import expandvars

import yaml
from dotenv import load_dotenv


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            super().__setitem__(key, self._convert(value))

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            return super().__getattr__(k)
        return v

    def __setattr__(self, k, v):
        self[k] = v

    def __setitem__(self, k, v):
        super().__setitem__(k, self._convert(v))

    def __getitem__(self, k):
        if isinstance(k, str) and "." in k:
            k = k.split(".")
        if isinstance(k, (list, tuple)):
            return reduce(lambda d, kk: d[kk], k, self)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if isinstance(k, str) and "." in k:
            try:
                return self[k]
            except KeyError:
                return default
        return super().get(k, default)

    @staticmethod
    def _convert(value):
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [DotDict._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DotDict._convert(item) for item in value)
        return value


def load_configurations(path: str) -> DotDict:
    """
    Used for parsing configuration files.

    :param str path: path to conf file
    :returns DotDict: dictionary accessing fields with dot notation
    """
    load_dotenv()
    with open(path) as f:
        cfg = yaml.safe_load(expandvars(f.read()))
    return DotDict(cfg)
