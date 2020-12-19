import config

from utils.plot import VisdomPlot

def buildPlot(vis_params: config.VisConfig=None):

    if vis_params is None:
        vis_params = config.vis

    # TODO: 增加其他类型的可视化工具
    return _visdom_default(vis_params)

def _visdom_default(vis_params: config.VisConfig):
    if vis_params.RecordGrad:
        vis_params.PlotTypes.append('line')
        vis_params.PlotTitles.append('gradient')
        vis_params.PlotXLabels.append('iterations')
        vis_params.PlotYLabels.append('gradient norm')
        vis_params.PlotLegends.append(['Encoder Gradient'])

    return VisdomPlot(env_title='train monitoring',
                      types=vis_params.PlotTypes,
                      titles=vis_params.PlotTitles,
                      xlabels=vis_params.PlotXLabels,
                      ylabels=vis_params.PlotYLabels,
                      legends=vis_params.PlotLegends)