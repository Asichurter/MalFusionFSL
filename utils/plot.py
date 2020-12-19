import visdom
import numpy as np
import matplotlib.pyplot as plt

class VisdomPlot:

    def __init__(self, env_title, types, titles, xlabels, ylabels, legends):
        self.Handle = visdom.Visdom(env=env_title)
        self.Types = {title: type_ for title,type_ in zip(titles, types)}
        self.XLabels = {title: xlabel for title, xlabel in zip(titles, xlabels)}
        self.YLabels = {title: ylabel for title, ylabel in zip(titles, ylabels)}
        self.Legends = {title: legend for title, legend in zip(titles, legends)}

    def update(self, title, x_val, y_val, update={'flag':False, 'val':None}):
        if not update['flag']:
            update_flag = None if x_val==0 else 'append'
        else:
            update_flag = update['val']

        y_val = np.array(y_val)
        y_size = y_val.shape[1]

        x_val = np.ones((1, y_size)) * x_val

        plot_func = self.getType(self.Types[title])

        plot_func(X=x_val, Y=y_val, win=title,
                  opts=dict(
                      legend=self.Legends[title],
                      title=title,
                      xlabel=self.XLabels[title],
                      ylabel=self.YLabels[title],
                  ),
                  update= update_flag)

    def getType(self, t):
        if t == 'line':
            return self.Handle.line
        else:
            raise NotImplementedError('暂未实现的类型:%d'%t)

def plotLine(points_list,
             label_list,
             title='',
             gap=100,
             color_list=['red'],
             style_list=['-'],
             grid=True,
             xlim=None, ylim=None,
             save_path=None):

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(grid, axis='y', color='black', linestyle='--')

    plt.title(title)
    for points,color,style,label in zip(points_list, color_list, style_list, label_list):
        x = [i * gap for i in range(len(points))]
        plt.plot(x, points, color=color, linestyle=style, label=label)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

