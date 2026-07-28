import backtrader as bt

class UpHammer(bt.Indicator):
    lines = ('hammer1',)

    params = (
        ('barplot', False),  # plot above/below max/min for clarity in bar plot
        ('bardist', 0.015),  # distance to max/min in absolute perc
    )

    plotinfo = dict(plot=True, subplot=False, plotlinelabels=True,plotabove=True)
    plotlines = dict(hammer1=dict(marker='s', markersize=5.0, color='blue', fillstyle='full', ls=''))

    def __init__(self):
        # open high low close
        self.lines.hammer = bt.talib.CDLHAMMER(self.datas[0], self.datas[1], self.datas[2], self.datas[3])
    def next(self):
        if self.lines.hammer[0] == 100:
            if abs(1 - self.datas[3][0] / self.datas[1][0]) < 0.001 \
                    or abs(1 - self.datas[0][0] / self.datas[1][0]) < 0.001:
                self.lines.hammer1[0] = self.datas[0][0] # 赋hammer正值(=close)
            else:
                self.lines.hammer1[0] = 0
        else:
            self.lines.hammer1[0] = 0

