from .transformer import TransformerSet
from .deepset import DeepSet


def getSet2SetFunc(func_type, input_size, dropout, **kwargs):
    if func_type == 'deepset':
        return DeepSet(input_size, dropout=dropout, **kwargs)
    elif func_type == 'transformer':
        return TransformerSet(input_size, dropout, **kwargs)
    else:
        raise ValueError(f"[getSet2SetFunc] Unsupported set2set func type: {func_type}")
