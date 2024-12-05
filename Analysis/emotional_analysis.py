import pandas as pd
import jieba
from snownlp import SnowNLP

# 1. 读取CSV文件
df = pd.read_csv('《高等数学》同济版 2024年更新宋浩老师.csv')

# 2. 情感分析
def sentiment_analysis(text):
    s = SnowNLP(text)
    score = s.sentiments  # 获取情感得分
    return score

# 3. 根据情感得分划分等级
def get_sentiment_level(score):
    if score < 0.4:
        return '消极'
    elif score > 0.6:
        return '积极'
    else:
        return '中性'

# 4. 处理评论并添加分数和等级
df['分数'] = df['评论内容'].apply(sentiment_analysis)  # 情感分析得分
df['等级'] = df['分数'].apply(get_sentiment_level)  # 根据得分确定等级

# 5. 保存到新的文件
df.to_csv('高等数学同济版2024年更新宋浩老师_情感分析.csv', index=False)

print("情感分析完成并保存为新的文件")
