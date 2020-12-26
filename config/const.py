
# 所有可能的启动路径的config文件相对路径
REL_CFG_PATHS = [
    "../config/",
    "config/"
]

# models using fast adaption
ADAPTED_MODELS = ['MetaSGD', 'ATAML', 'PerLayerATAML']

# models using multi-prototype
IMP_MODELS = ['IMP', 'SIMPLE', 'HybridIMP']

__all__ = ['ADAPTED_MODELS', 'IMP_MODELS', 'REL_CFG_PATHS']