import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 黑体字体
rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

# 1. 加载分析后的 CSV 文件
df = pd.read_csv('高等数学同济版2024年更新宋浩老师_情感分析.csv')

# 2. 统计情感分数的频次 (0.2区间统计)
bins = [i * 0.2 for i in range(6)]  # 分为 [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['[0.0-0.2)', '[0.2-0.4)', '[0.4-0.6)', '[0.6-0.8)', '[0.8-1.0)']
df['情感分数区间'] = pd.cut(df['分数'], bins=bins, labels=labels, right=False)

# 统计各个区间的频次
score_counts = df['情感分数区间'].value_counts().sort_index()

# 3. 绘制柱状图：情感分数区间的频次
plt.figure(figsize=(8, 6))
score_counts.plot(kind='bar', color='skyblue')
plt.title('情感分数区间的频次统计')
plt.xlabel('情感分数区间')
plt.ylabel('频次')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. 统计情感等级频次
level_counts = df['等级'].value_counts()

# 5. 绘制饼图：情感等级的比例
plt.figure(figsize=(7, 7))
level_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightblue'])
plt.title('情感等级比例')
plt.ylabel('')  # 不显示y轴标签
plt.tight_layout()
plt.show()

# 6. 额外统计图：情感分数的分布 (直方图)
plt.figure(figsize=(8, 6))
df['分数'].plot(kind='hist', bins=20, color='lightblue', edgecolor='black')
plt.title('情感分数的分布')
plt.xlabel('情感分数')
plt.ylabel('频次')
plt.tight_layout()
plt.show()
