import requests  # 导入 requests 库，用于发送 HTTP 请求
import re  # 导入 re 库，用于正则表达式处理
import time  # 导入 time 库，用于时间操作
import csv  # 导入 csv 库，用于读写 CSV 文件
import json  # 导入 json 库，用于处理 JSON 数据

# 定义请求头，模拟浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
}

# 定义获取评论的函数
def fetch_comments(video_id, max_pages=50):  # 最大页面数量默认值为 50，可调整
    comments = []  # 初始化一个空列表，用于存储所有评论
    last_count = 0  # 记录上次获取的评论数量，用于判断是否有新评论
    n = 1  # 页码初始值为 1

    # 开始分页获取评论，最多 max_pages 页
    for page in range(1, max_pages + 1):
        # 构建 API 请求的 URL
        url = f'https://api.bilibili.com/x/v2/reply/main?next={n}&type=1&oid={video_id}&mode=3'
        try:
            # 发送 GET 请求并设置超时时间为 10 秒
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:  # 如果请求成功，HTTP 状态码为 200
                data = response.json()  # 将返回的 JSON 数据转换为字典

                print(page)  # 输出当前请求的页数，用于调试
                # 如果没有评论，说明已获取所有评论，跳出循环
                if data['data']['replies'] == None:
                    break
                # 如果数据中包含评论
                if data and 'replies' in data['data']:
                    # 遍历每一条评论
                    for comment in data['data']['replies']:
                        # 提取评论中的信息，并构建字典保存
                        comment_info = {
                            '用户昵称': comment['member']['uname'],  # 评论者的昵称
                            '评论内容': comment['content']['message'],  # 评论内容
                            '被回复用户': '',  # 假设为一级评论，暂无被回复的用户信息
                            '评论层级': '一级评论',  # 假设为一级评论
                            '性别': comment['member']['sex'],  # 评论者性别
                            '用户当前等级': comment['member']['level_info']['current_level'],  # 评论者等级
                            '点赞数量': comment['like'],  # 评论的点赞数量
                            '回复时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(comment['ctime']))  # 回复时间，格式化为可读形式
                        }
                        # 将每条评论的信息添加到评论列表中
                        comments.append(comment_info)

                n += 1  # 增加页码，获取下一页的评论
                # 如果本次获取的评论数量和上次相同，则说明没有新评论，跳出循环
                if last_count == len(comments):
                    break
                last_count = len(comments)  # 更新上次获取的评论数量
            else:
                break  # 如果请求返回的状态码不是 200，则终止请求
        except requests.RequestException as e:  # 如果请求过程中出现异常，捕获异常并输出错误信息
            print(f"请求出错: {e}")
            break
        # 为了避免过于频繁地发送请求，控制请求频率，每次请求后暂停 1 秒
        time.sleep(1)
    return comments  # 返回所有获取到的评论数据


# 定义保存评论数据到 CSV 文件的函数
def save_comments_to_csv(comments, video_bv):
    # 打开一个 CSV 文件（以写模式），并指定编码为 utf-8
    with open(f'./result/{video_bv}.csv', mode='w', encoding='utf-8', newline='') as file:
        # 创建一个 CSV DictWriter 对象，用于写入字典类型的数据
        writer = csv.DictWriter(file,
                                fieldnames=['用户昵称', '性别', '评论内容', '被回复用户', '评论层级', '用户当前等级',
                                            '点赞数量', '回复时间'])  # 设置 CSV 文件的表头
        writer.writeheader()  # 写入表头
        # 遍历所有评论，将每条评论写入 CSV 文件
        for comment in comments:
            writer.writerow(comment)


# 定义清理文件名的函数，移除文件名中的非法字符
def sanitize_filename(filename):
    # 使用正则表达式替换掉文件名中的特殊字符
    return re.sub(r'[\\/*?:"<>|]', "", filename)


# 设置视频名称和视频 BV 号
video_name = '《高等数学》同济版 2024年更新|宋浩老师'  # 视频名字
video_bv = 'BV1Eb411u7Fw'  # 视频 BV 号
print(f'视频名字: {video_name}, video_bv: {video_bv}')  # 输出视频名称和 BV 号，供调试使用

# 调用 fetch_comments 函数获取评论数据
comments = fetch_comments(video_bv)

# 清理视频名称，移除其中的非法字符，以便用于生成文件名
sanitized_video_name = sanitize_filename(video_name)

# 调用 save_comments_to_csv 函数，将评论数据保存到 CSV 文件中
save_comments_to_csv(comments, sanitized_video_name)  # 会将所有评论保存到一个 CSV 文件中
