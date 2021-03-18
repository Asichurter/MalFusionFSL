import torch.nn as nn

import config
from utils.manager import PathManager


class BaseModel(nn.Module):

    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source,
                 **kwargs):
        super(BaseModel, self).__init__()

        self.LossFunc = loss_func
        self.ModelParams = model_params
        self.TaskParams = None
        self.ImageW = None
        self.TaskType = ""
        self.DataSource = data_source

        self.FusedFeatureDim = None
        self.Fusion = None  # buildFusion(self, model_params)

        self.SeqEmbedPipeline = []
        self.ImgEmbedPipeline = []

    def _seqEmbed(self, x, lens=None):
        for worker in self.SeqEmbedPipeline:
            x = worker(x, lens)
        return x

    def _imgEmbed(self, x):
        for worker in self.ImgEmbedPipeline:
            x = worker(x)
        return x

    def _extractEpisodeTaskStruct(self,
                                  support_seqs, query_seqs,
                                  support_imgs, query_imgs):
        assert (support_seqs is None) ^ (query_seqs is not None), \
            f"[extractEpisodeTaskStruct] 支持集和查询集的序列数据存在性不一致: support: {support_seqs is None}, query:{query_seqs is None}"
        assert (support_imgs is None) ^ (query_imgs is not None), \
            f"[extractEpisodeTaskStruct] 支持集和查询集的图像数据存在性不一致: support: {support_imgs is None}, query:{query_imgs is None}"

        # TODO: 支持更多task的输入类型来提取任务结构参数
        if support_seqs is not None:
            k = support_seqs.size(1)
            n = support_seqs.size(0)
        elif support_imgs is not None:
            k = support_imgs.size(1)
            n = support_imgs.size(0)
        else:
            assert False, "[extractEpisodeTaskStruct] 序列和图像的支持集都为None，无法提取n,k"

        if query_seqs is not None:
            qk = query_seqs.size(0)
        elif query_imgs is not None:
            qk = query_imgs.size(0)
        else:
            assert False, "[extractEpisodeTaskStruct] 序列和图像的查询集都为None，无法提取qk"

        # support img shape: [n, k, 1, w, w]
        # query img shape: [qk, 1, w, w]
        if support_imgs is not None:
            w = support_imgs.size(3)
        elif query_imgs is not None:
            w = query_imgs.size(2)
        else:
            w = None

        self.TaskParams = config.EpisodeTaskConfig(k, n, qk)
        self.ImageW = w

    def embed(self,
              support_seqs, query_seqs,
              support_lens, query_lens,
              support_imgs, query_imgs):

        self._extractEpisodeTaskStruct(support_seqs, query_seqs,
                                       support_imgs, query_imgs)

        k, n, qk, w = self.TaskParams.k, self.TaskParams.n, self.TaskParams.qk, self.ImageW

        # 提取任务结构时，已经判断过支持集和查询集的数据一致性，此处做单侧判断即可
        if support_seqs is not None:
            support_seqs = support_seqs.view(n * k, -1)
            support_seqs = self._seqEmbed(support_seqs, support_lens).view(n, k, -1)
            query_seqs = self._seqEmbed(query_seqs, query_lens)

            assert support_seqs.size(2) == query_seqs.size(1), \
                "[BaseProtoModel.Embed] Support/query sequences' feature dimension size must match: (%d,%d)" \
                % (support_seqs.size(2), query_seqs.size(1))

        # 提取任务结构时，已经判断过支持集和查询集的数据一致性，此处做单侧判断即可
        if support_imgs is not None:
            support_imgs = support_imgs.view(n*k, 1, w, w)      # 默认为单通道图片
            support_imgs = self._imgEmbed(support_imgs).view(n, k, -1)
            query_imgs = self._imgEmbed(query_imgs).squeeze()

            assert support_imgs.size(2) == query_imgs.size(1), \
                "[BaseProtoModel.Embed] Support/query images' feature dimension size must match: (%d,%d)"\
                %(support_imgs.size(2),query_imgs.size(1))

        return support_seqs, query_seqs, support_imgs, query_imgs

    def forward(self,                       # forward接受所有可能用到的参数
                support_seqs, support_imgs, support_lens, support_labels,
                query_seqs, query_imgs, query_lens, query_labels,
                loss_func,
                **kwargs):
        raise NotImplementedError

    def name(self):
        return "BaseModel"

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def _fuse(self, seq_features, img_features, **kwargs):
        return self.Fusion(seq_features, img_features, **kwargs)

    def train_state(self, mode=True):
        self.TaskType = "Train"
        super().train(mode)

    def validate_state(self):
        self.TaskType = "Validate"
        super().eval()

    def test_state(self):
        self.TaskType = "Test"
        super().eval()