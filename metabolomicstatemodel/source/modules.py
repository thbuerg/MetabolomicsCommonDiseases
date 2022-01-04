import math
import torch
import torch.nn as nn
from omegaconf.listconfig import ListConfig


class SingleLayerNet(nn.Module):
    def __init__(self, input_dim=32, output_dim=2, final_activation=None, final_batchnorm=False, **kwargs):
        super(SingleLayerNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if final_activation is not None and isinstance(final_activation, str):
            m = final_activation.split('.')
            final_activation = getattr(nn, m[1])
            print(final_activation)

        predictor_specs = [nn.Linear(self.input_dim, self.output_dim), ]
        if final_batchnorm:
            predictor_specs.append(nn.BatchNorm1d(self.output_dim))
        if final_activation is not None:
            predictor_specs.append(final_activation())
        self.predictor = nn.Sequential(*predictor_specs)

    def forward(self, input):
        fts = self.predictor(input)
        return fts


class MLP(nn.Module):
    def __init__(self,
                 input_dim=32,
                 output_dim=2,
                 hidden_dim=256,
                 n_hidden_layers=None,
                 activation="nn.SELU",
                 dropout_fn='nn.Dropout',
                 norm_fn='nn.BatchNorm1d',
                 norm_layer="all",
                 dropout_after_norm=True,
                 input_norm=False,
                 final_activation=None,
                 final_norm=False,
                 snn_init=True,
                 dropout=0.5, **kwargs):
        """
        A simple feed-forward neural network.
        :param input_dim:   `int`, dimension ot the input features
        :param output_dim:  `int`, dimension of the outlayer
        :param activation:  `nn.Module`, NOT initialized. that is the activation of the last layer, if `None` no activation will be performed.
        :param dropout:     `float`, [<1], that specifies the dropout probability
        :param kwargs:
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        norm_layer = norm_layer if isinstance(norm_layer, (list, tuple, ListConfig)) else [l for l in range(100)]
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        if norm_fn is not None and isinstance(norm_fn, str):
            m = norm_fn.split('.')
            norm_fn = getattr(nn, m[1])
            self.norm_fn = norm_fn
        if dropout_fn is not None and isinstance(dropout_fn, str):
            m = dropout_fn.split('.')
            dropout_fn = getattr(nn, m[1])
        if activation is not None and isinstance(activation, str):
            m = activation.split('.')
            activation = getattr(nn, m[1])
            print(activation)
        if final_activation is not None and isinstance(final_activation, str):
            m = final_activation.split('.')
            final_activation = getattr(nn, m[1])
            print(final_activation)
        print(self.output_dim)

        if input_norm:
            self.input_norm = nn.LayerNorm(self.input_dim)
        else:
            self.input_norm = None

        if isinstance(hidden_dim, int):
            if isinstance(norm_layer, (list, tuple, ListConfig)): norm_fn = self.norm_fn if 0 in norm_layer else None
            else: norm_fn = None
            mlp_specs = [nn.Linear(input_dim, hidden_dim),]
            if dropout_after_norm == True:
                mlp_specs.extend([
                    norm_fn(hidden_dim) if norm_fn is not None else nn.Identity(),
                    dropout_fn(self.dropout),])
            else:
                mlp_specs.extend([
                    dropout_fn(self.dropout),
                    norm_fn(hidden_dim) if norm_fn is not None else nn.Identity(),
                ])
            mlp_specs.extend([activation(),])

            for i in range(n_hidden_layers):
                if isinstance(norm_layer, (list, tuple, ListConfig)): norm_fn = self.norm_fn if i+1 in norm_layer else None
                else: norm_fn = None
                mlp_specs.extend([nn.Linear(hidden_dim, hidden_dim),])
                if dropout_after_norm == True:
                    mlp_specs.extend([
                        norm_fn(hidden_dim) if norm_fn is not None else nn.Identity(),
                        dropout_fn(self.dropout), ])
                else:
                    mlp_specs.extend([
                        dropout_fn(self.dropout),
                        norm_fn(hidden_dim) if norm_fn is not None else nn.Identity(),
                    ])
                mlp_specs.extend([activation(),])
            self.mlp = nn.Sequential(*mlp_specs)
            predictor_specs = [
                nn.Linear(hidden_dim, self.output_dim),
            ]
        elif isinstance(hidden_dim, (list, tuple, ListConfig)):
            assert n_hidden_layers is None, 'Either pass list of hidden_dims, or n_hidden_layers with single hidden_dim'
            mlp_specs = []
            for i, h in enumerate(hidden_dim):
                if isinstance(norm_layer, (list, tuple, ListConfig)): norm_fn = self.norm_fn if i in norm_layer else None
                else: norm_fn = None
                mlp_specs.extend([nn.Linear(input_dim if i==0 else hidden_dim[i-1], h),])
                if dropout_after_norm == True:
                    mlp_specs.extend([
                        norm_fn(h) if norm_fn is not None else nn.Identity(),
                        dropout_fn(self.dropout)])
                else:
                    mlp_specs.extend([
                        dropout_fn(self.dropout),
                        norm_fn(h) if norm_fn is not None else nn.Identity(),
                    ])
                mlp_specs.extend([activation(),])
            self.mlp = nn.Sequential(*mlp_specs)
            predictor_specs = [
                nn.Linear(hidden_dim[-1], self.output_dim),
                ]
        else:
            raise ValueError('hidden_dim is either int or list of ints')

        if final_norm:
            predictor_specs.append(self.norm_fn(self.output_dim))
        if final_activation is not None:
            predictor_specs.append(final_activation())

        self.predictor = nn.Sequential(*predictor_specs)

        if snn_init:
            self.reset_parameters('predictor')
            self.reset_parameters('mlp')

    def forward(self, input):
        if self.input_norm is not None:
            input = self.input_norm(input)
        fts = self.mlp(input)
        output = self.predictor(fts)
        return output

    def reset_parameters(self, name):
        for layer in getattr(self, name):
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)


class ResidualHeadMLP(nn.Module):
    def __init__(self,
                 predictor_mlp=MLP,
                 predictor_mlp_kwargs=dict(input_dim=None,
                                           output_dim=None,
                                           hidden_dim=None,
                                           activation="nn.SiLU",
                                           dropout_fn='nn.Dropout',
                                           dropout=0.2,
                                           final_activation="nn.SiLU",
                                           final_batchnorm=False),
                 skip_connection_mlp=MLP,
                 skip_connection_input_dim=32,
                 skip_connection_mlp_kwargs=dict(input_dim=None,
                                                 output_dim=None,
                                                 hidden_dim=None,
                                                 activation="nn.SiLU",
                                                 dropout_fn='nn.Dropout',
                                                 dropout=0.2,
                                                 final_activation="nn.SiLU",
                                                 final_batchnorm=False),
                 mlp=MLP,
                 mlp_kwargs=dict(input_dim=None,
                                 output_dim=None,
                                 hidden_dim=None,
                                 activation="nn.SiLU",
                                 dropout_fn='nn.Dropout',
                                 dropout=0.2,
                                 final_activation="nn.SiLU",
                                 final_batchnorm=False),
                 **kwargs):
        super().__init__()
        self.skip_connection_input_dim = skip_connection_input_dim

        if predictor_mlp is not None and isinstance(predictor_mlp, str):
            self.predictor_mlp = eval(predictor_mlp)
        if skip_connection_mlp is not None and isinstance(skip_connection_mlp, str):
            self.skip_connection_mlp = eval(skip_connection_mlp)
        if mlp is not None and isinstance(mlp, str):
            self.mlp = eval(mlp)

        skip_connection_mlp_kwargs['input_dim'] = self.skip_connection_input_dim

        self.predictor = self.predictor_mlp(**predictor_mlp_kwargs)
        self.skip_connection = self.skip_connection_mlp(**skip_connection_mlp_kwargs)
        self.mlp = self.mlp(**mlp_kwargs)

    def forward(self, input):
        features, covariates = input
        fts = self.mlp(features)
        skip_fts = self.skip_connection(covariates)
        h = fts + skip_fts
        out = self.predictor(h)
        return out


class MLPResNetBlock(nn.Module):
    """
    MLP version of the ResBlock wrapped by TemporalBlock from:
    https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/unet.py#L143

    with less complexity and fts.
    """
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, dropout=0.3,
                 embedding_dim=16,
                 use_scale_shift_norm=False,
                 temporal_embedding=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_embedding=temporal_embedding

        if temporal_embedding:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embedding_dim,
                          2 * self.output_dim if use_scale_shift_norm else self.output_dim),
            )

        self.in_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )

        self.skip_connection = nn.Identity() if self.input_dim==self.output_dim else \
            nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim),
                nn.SiLU()
            )

        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

    def reset_parameters(self, name):
        for layer in getattr(self, name):
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
