import os
import time

GLOBAL_LOG_FOLDER = "ZhengShangYou/zhengzero/logs" + time.strftime(
    "/%Y-%m-%d_%H-%M-%S", time.localtime()
)

if not os.path.exists(GLOBAL_LOG_FOLDER):
    os.makedirs(GLOBAL_LOG_FOLDER)
