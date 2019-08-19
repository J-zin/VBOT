import os

# allFiles=[]

# def begin_new_listfile():
# 	global allFiles
# 	allFiles=[]
# 	return

def list_all_fm_file(filepath,suffix):
	# global allFiles
	allFiles = []
	files = os.listdir(filepath)
	for fi in files:
		fi_d = os.path.join(filepath,fi)
		if os.path.isdir(fi_d):
			list_all_fm_file(fi_d,suffix)
		else:
			if fi_d.find(suffix)>0:
				allFiles.append(fi_d)
	return allFiles
