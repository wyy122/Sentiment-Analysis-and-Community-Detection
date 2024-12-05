# -*- coding: utf-8 -*-
from algorithm import GN
from matplotlib import pyplot as plt
import networkx as nx
import copy

filepath = r'./data/football.gml'

# 获取社区划分
G = nx.read_gml(filepath)
G_copy = copy.deepcopy(G)

# 获取社区划分和模块度，gn_com 是社区划分列表，模块度被丢弃
gn_com, score = GN.partition(G_copy)
print(gn_com)
print(score)

# 可视化1：将社区编号映射为颜色
pos = nx.spring_layout(G, seed=42, k=1.5)  # 增大k值，使节点间距更大
# 使用颜色映射器将社区编号映射到颜色
cmap = plt.get_cmap("tab20")  # 使用更加丰富的颜色映射
colors = [cmap(i / (max(gn_com) + 1)) for i in gn_com]  # 将社区编号映射为颜色
plt.figure(figsize=(18, 18))  # 增大图形尺寸，避免节点过于密集
nx.draw(G, pos, with_labels=False, node_size=100, width=1.5, node_color=colors, alpha=0.9)  # 增大节点大小，增加透明度
plt.title("Community Detection Visualization")
plt.savefig('community_detection_high_res.png', dpi=500)  # 保存高分辨率图像
plt.show()

# 后续处理，gn_com 保持社区编号
V = [node for node in G.nodes()]
com_dict = {node: com for node, com in zip(V, gn_com)}
k = max(com_dict.values()) + 1

# 根据社区编号划分节点
com = [[V[i] for i in range(G.number_of_nodes()) if gn_com[i] == j] for j in range(k)]

# 构造可视化所需要的图
G_graph = nx.Graph()
for each in com:
    G_graph.update(nx.subgraph(G, each))

# 为每个节点分配颜色
color = [com_dict[node] for node in G_graph.nodes()]

# 可视化2：绘制社区图
pos = nx.spring_layout(G_graph, seed=4, k=0.45)  # 调整k值，避免节点重叠

plt.figure(figsize=(18, 18))  # 增大图形尺寸，避免节点过于密集
nx.draw(G, pos, with_labels=False, node_size=1, width=0.1, alpha=0.2)  # 原图可视化
nx.draw(G_graph, pos, with_labels=True, node_color=color, node_size=200, width=1.5, font_size=12, font_color='#000000')  # 社区划分可视化
plt.title("Community Detection Subgraph")
plt.savefig('community_detection_subgraph_high_res.png', dpi=500)  # 保存高分辨率图像
plt.show()
