from comm_api import *
import os
import pickle
import numpy as np
import datetime
import struct

FM_FPS=60
TIME_PER_FREAME=16.6666666666666667

#BlendShapes.
BlendShapesNames=["mouthLowerDown_L","cheekSquint_L","mouthDimple_R","mouthPress_L","mouthStretch_R","mouthFunnel","mouthLowerDown_R","mouthLeft","cheekSquint_R","jawOpen","jawForward","mouthPress_R","jawRight","mouthShrugLower","mouthFrown_L","cheekPuff","mouthStretch_L","mouthRollLower","mouthUpperUp_R","mouthShrugUpper","mouthSmile_L","mouthClose","jawLeft","mouthDimple_L","mouthFrown_R","mouthPucker","mouthRight","mouthSmile_R","mouthUpperUp_L","mouthRollUpper"]

#Json
def SaveResultJsonFile(f,result):
	f.write("[");
	lineCnt=result.shape[0]
	colCnt=result.shape[1]
	
	for i in range(lineCnt):
		f.write("\n\t{");
		nTime=int(i/FM_FPS)*1000+ int(i%FM_FPS*TIME_PER_FREAME)
		
		f.write("\"timestamp\":")
		f.write(str(nTime))
		
		for j in range(colCnt):
			bName=BlendShapesNames[j]
			bVale=result[i][j]
			f.write(",\"")
			f.write(bName)
			f.write("\":")
			f.write(str(bVale))
		if (i == lineCnt-1):
			f.write("}")
		else:
			f.write("},")
	f.write("\n]")
	return

#fbx
def SaveBfxFile(f,result):
	lineCnt=result.shape[0]
	colCnt=result.shape[1]
	print(lineCnt)
	
	VALUE_BY=1.0
	
	#lineCnt
	allBones=0
	f.write(struct.pack('q',allBones))
	
	lastValue=[]
	for j in range(colCnt):
		lastValue.append(0.0)
		
	for i in range(lineCnt):
		nTime=int(i/FM_FPS)*1000+ int(i%FM_FPS*TIME_PER_FREAME)
		
		for j in range(colCnt):
			bName=BlendShapesNames[j]
			bVale=result[i][j]
			if (bVale<0.005):
				bVale = 0 
			
			if (abs(lastValue[j]-bVale)<0.005):
				#print(nTime,",",bName,": Same{",lastValue[j],":",bVale,"}")
				continue
			lastValue[j]=bVale
			
			bVale=bVale*VALUE_BY
			if bVale>1.0:
				bVale=1.0
				
			#print(nTime,",",bName,":",bVale)
			#save nTime
			f.write(struct.pack('q',nTime))
			strlen=len(bName)
			f.write(struct.pack('B',strlen))
			f.write(str.encode(bName))
			f.write(struct.pack('d',bVale))
			allBones=allBones+1
	
	f.seek(0)
	f.write(struct.pack('q',allBones))
	return

def transObs2fbxFile(fname):
	_, tempfilename = os.path.split(fname)
	midName, _ = os.path.splitext(tempfilename)
	fbxFile='./output/'+midName+'.bsx'
	jsonFile='./output/'+midName+'.json'
	
	result=[]
	with open(fname,'rb') as f:
		result=pickle.load(f)
	
	with open(fbxFile,'wb') as f:
		SaveBfxFile(f,result)
	print("====Save [",fbxFile,"] OK!!===")
	
	with open(jsonFile,'w') as f:
		SaveResultJsonFile(f,result) 
	print("====Save [",jsonFile,"] OK!!===")
	return

def BatchWork_fmy2Bsx():
	begin_new_listfile()
	allOutFile=list_all_fm_file('./data/values/','fm_y')
	for i in range(len(allOutFile)):
		transObs2fbxFile(allOutFile[i])
	return

if __name__ == '__main__':
	BatchWork_fmy2Bsx()
