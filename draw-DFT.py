# lines = open("/home/wkp/Documents/W/eChat_Data/xwechat_files/wxid_ash031ne7s4k22_4d26/msg/file/2024-05/_test.tmp").readlines()
lines = open("/home/wkp/Documents/WeChat_Data/xwechat_files/wxid_ash031ne7s4k22_4d26/msg/file/2024-05/_test_1.tmp").readlines()

facts = []
preds = []
times = []


for line in lines:
	if line.strip() == "":
		continue
	time, fact, pred = line.strip().split(" ")
	time = float(time)
	fact = float(fact)
	pred = float(pred)
	times.append(time)
	facts.append(fact)
	preds.append(pred)

L = 150 + 23
R = 300 + 23
facts = facts[L:R]
preds = preds[L:R]
times = times[L:R]

from matplotlib import pyplot as plt

fontsize = 25

x = list(range(len(facts)))

def print(name, data):
	plt.figure(figsize=(30, 7), dpi=192, facecolor='none')
	# 添加标题和标签并设置字体大小
	plt.title(f'Result of {name}', fontsize=fontsize)
	plt.xlabel('Time', fontsize=fontsize)
	plt.ylabel('Flow', fontsize=fontsize)
	# 调整刻度字体大小
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.plot(x, facts, label="fact", color="black")
	plt.plot(x, data, label=name, color="#7028e4", linewidth=3)
	# 添加图例并设置字体大小
	plt.legend(fontsize=fontsize)
	# plt.show()
	plt.savefig(f"fig-std-{name}.png", bbox_inches="tight")
	plt.savefig(f"fig-std-{name}.svg", format="svg", bbox_inches="tight")

# print("DFT", preds)
print("Average_24x7", preds)
