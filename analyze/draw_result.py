import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def zhexian():
    epochs = [1, 2, 3, 4, 5]
    accuracy = [0.7, 0.8, 0.85, 0.86, 0.87]
    precision = [0.65, 0.7, 0.75, 0.78, 0.8]
    recall = [0.6, 0.65, 0.7, 0.75, 0.77]
    f1_score = [0.62, 0.68, 0.72, 0.76, 0.78]

    # 绘制折线图
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.plot(epochs, precision, label='Precision')
    plt.plot(epochs, recall, label='Recall')
    plt.plot(epochs, f1_score, label='F1 score')

    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.title('Model performance over epochs')
    plt.legend()
    plt.show()


def zhuzhuang():
    # 指标名称
    labels = ['None', 'HGD', 'HGD + MSS', 'HGD + MSS + NFP']

    # 四种方法的性能指标值
    acc = [0.136, 0.477, 0.708, 0.866]
    pre = [0.069, 0.462, 0.724, 0.883]
    recall = [0.141, 0.493, 0.688, 0.849]
    f1 = [0.082, 0.455, 0.698, 0.865]

    # 生成四种颜色，以符合马卡龙色系
    colors = sns.color_palette("pastel", 4)

    # 设置x轴标签位置和标签名字
    x = np.arange(len(labels))

    # 指定填充图案
    patterns = ['/', 'o', 'x', '\\']

    # 设置每个柱子的宽度
    bar_width = 0.35

    # 创建子图
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # 绘制Acc指标图表
    axes[0, 0].bar(x, acc, width=bar_width)
    for i in range(len(x)):
        axes[0, 0].patches[i].set_facecolor(colors[i])
        axes[0, 0].patches[i].set_hatch(patterns[i])
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0, 0].set_ylabel('Acc')
    # axes[0, 0].set_title('Performance Comparison of Four Methods')

    # 绘制Pre指标图表
    axes[0, 1].bar(x, pre, width=bar_width)
    for i in range(len(x)):
        axes[0, 1].patches[i].set_facecolor(colors[i])
        axes[0, 1].patches[i].set_hatch(patterns[i])
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=15, ha='right')
    axes[0, 1].set_ylabel('Pre')
    # axes[0, 1].set_title('Performance Comparison of Four Methods')

    # 绘制Recall指标图表
    axes[1, 0].bar(x, recall, width=bar_width)
    for i in range(len(x)):
        axes[1, 0].patches[i].set_facecolor(colors[i])
        axes[1, 0].patches[i].set_hatch(patterns[i])
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=15, ha='right')
    axes[1, 0].set_ylabel('Recall')
    # axes[1, 0].set_title('Performance')

    axes[1, 1].bar(x, f1, width=bar_width)
    for i in range(len(x)):
        axes[1, 1].patches[i].set_facecolor(colors[i])
        axes[1, 1].patches[i].set_hatch(patterns[i])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1, 1].set_ylabel('F1')
    # axes[1, 1].set_title('Performance Comparison of Four Methods')

    # 显示图形
    plt.savefig('performance_comparison.png', dpi=300)
    plt.show()


def reli():
    import numpy as np
    import seaborn as sns

    # 生成数据
    model_names = ['Model A', 'Model B', 'Model C']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 score']
    scores = np.array([
        [0.8, 0.75, 0.7, 0.72],
        [0.85, 0.8, 0.75, 0.77],
        [0.9, 0.85, 0.8, 0.82]
    ])

    # 绘制热力图
    sns.set()
    ax = sns.heatmap(scores, annot=True, cmap='YlGnBu',
                     xticklabels=metrics, yticklabels=model_names)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Models')
    plt.show()


def box():
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成数据
    model_names = ['Model A', 'Model B', 'Model C']
    f1_scores = [[0.7, 0.75, 0.8, 0.81, 0.85],
                 [0.72, 0.76, 0.78, 0.79, 0.82],
                 [0.75, 0.8, 0.83, 0.85, 0.87]
                 ]

    # 绘制箱线图
    fig, ax = plt.subplots()
    ax.boxplot(f1_scores)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('F1 score')
    plt.show()


if __name__ == "__main__":
    reli()