import config

from utils.plot import VisdomPlot
from utils.plot import EmptyPlot

def buildPlot(plot_params: config.PlotConfig=None):

    if plot_params is None:
        plot_params = config.plot

    if not plot_params.Enabled:      # 如果没有开启可视化，则默认返回EmptyPlot对象
        return EmptyPlot()

    return PlotSwitch.get(plot_params.Type, _visdom_default)(plot_params)

def _visdom_default(plot_params: config.PlotConfig):
    if plot_params.RecordGrad:
        plot_params.PlotTypes.append('line')
        plot_params.PlotTitles.append('gradient')
        plot_params.PlotXLabels.append('iterations')
        plot_params.PlotYLabels.append('gradient norm')
        plot_params.PlotLegends.append(['Encoder Gradient'])

    return VisdomPlot(env_title='train monitoring',
                      types=plot_params.PlotTypes,
                      titles=plot_params.PlotTitles,
                      xlabels=plot_params.PlotXLabels,
                      ylabels=plot_params.PlotYLabels,
                      legends=plot_params.PlotLegends)

PlotSwitch = {
    "visdom": _visdom_default
}