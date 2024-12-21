import resource
from datetime import datetime
import torch

def get_max_memory_usage():
	# 在你的程序中执行需要监测内存的代码
	max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	# ru_maxrss 返回的单位是kilobytes（KB）
	print("=================================================================")
	print("max_memory ="+str(max_memory / 1024 /1024/1024)+"GB\n")
	return max_memory / 1024 /1024 # 转换为 megabytes (MB)

def time_string_fun():
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d %H:%M:%S")
	print(time_string)
	return time_string
	


if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)

	statetime= time_string_fun()
