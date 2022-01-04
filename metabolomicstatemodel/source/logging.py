from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from typing import Any, Dict, Iterable, Optional, Union
from argparse import Namespace


class FoolProofNeptuneLogger(NeptuneLogger):
    """
    Logger that does only log params if they do not exceed the str len limit.
    """
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)

        parameters_key = self.PARAMETERS_KEY
        if self._base_namespace:
            parameters_key = f'{self._base_namespace}/{parameters_key}'

        keys_to_pop = []
        for k, v in params.items():
            if len(str(v)) >= 16384:
                keys_to_pop.append(k)
        for k in keys_to_pop:
            params.pop(k)

        self.run[parameters_key] = params

