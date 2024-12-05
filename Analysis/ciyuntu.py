import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba  # 用于中文分词
import re

# 读取CSV文件
file_path = '《高等数学》同济版 2024年更新宋浩老师.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 查看数据的前几行，确保读取正确
print(df.head())

# 提取评论内容
comments = df['评论内容'].dropna()  # 删除缺失值

# 对评论内容进行分词
def chinese_word_cut(text):
    return ' '.join(jieba.cut(text))

# 对评论内容进行分词，并过滤掉单字
def chinese_word_cut_no_single(text):
    words = jieba.cut(text)
    filtered_words = [word for word in words if len(word) > 1]
    return ' '.join(filtered_words)

# 清洗文本：去除标点符号和特殊字符
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去掉标点符号
    return text

# 生成包含单字的文本
comments_text_with_single = ' '.join(comments.apply(chinese_word_cut))
comments_text_with_single = clean_text(comments_text_with_single)

# 生成去除单字的文本
comments_text_no_single = ' '.join(comments.apply(chinese_word_cut_no_single))
comments_text_no_single = clean_text(comments_text_no_single)

# 生成包含单字的词云图
wordcloud_with_single = WordCloud(
    font_path='msyh.ttc',  # 设置字体路径
    width=800,
    height=600,
    background_color='white',
    max_words=100,
    collocations=False
).generate(comments_text_with_single)

# 生成去除单字的词云图
wordcloud_no_single = WordCloud(
    font_path='msyh.ttc',  # 设置字体路径
    width=800,
    height=600,
    background_color='white',
    max_words=100,
    collocations=False
).generate(comments_text_no_single)

# 绘制词云图：包含单字
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud_with_single, interpolation='bilinear')
plt.title('WordCloud with Single Characters', fontsize=16)
plt.axis('off')
plt.show()

# 绘制词云图：去除单字
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud_no_single, interpolation='bilinear')
plt.title('WordCloud without Single Characters', fontsize=16)
plt.axis('off')
plt.show()

# 可视化分析：用户等级分布
level_counts = df['用户当前等级'].value_counts()

plt.figure(figsize=(6, 4))
level_counts.sort_index().plot(kind='bar', color='#76b7b2')
plt.title('User Level Distribution')
plt.xlabel('User Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
