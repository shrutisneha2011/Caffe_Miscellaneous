#!/usr/bin/env python
import numpy as np
import cv2
import imagePro as imp

def draw_flow(img, flow, step=8):
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
	fx, fy = flow[y,x].T
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.polylines(vis, lines, 0, (0, 0, 255))
	for (x1, y1), (x2, y2) in lines:
		cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)
	return vis

################# Code added ############################
cv2_version = cv2.__version__

fps = 12
w,h = 640,480
path = '/home/shruti/'	#path for saving video file

if(cv2_version=='3.4.0'):
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
else:
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter(path + "opticalFlow.avi", fourcc, fps, (w,h))

def opticalFlow(fn,fact,tframes,display):
	global cv2_version,fps, w,h, fourcc, out
	cam = cv2.VideoCapture(fn)
	ret, prev = cam.read()
	prev1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
	prevgray=imp.resizeBinary(prev1,fact)
	
	lsFlow=[]
	i=0
	while True:
		if i<=tframes-2:
			ret, img = cam.read()
			img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			gray=imp.resizeBinary(img1,fact)
			if(cv2_version=='3.4.0'):
				flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)	#for opencv - 3.4.0
			else:
				flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
			flow16 = np.array(flow,np.float16)
			#print 'flow.shape: ', flow.shape #(480, 640, 2)
			prevgray = gray
			#cv2.imshow('flow', draw_flow(gray, flow))
			if display == True:
				cv2.imshow('flow16', draw_flow(gray, flow16))
				#out.write(flow)			##### unable to save video
				#print '====================================='
				#print flow[:,:,0]
				#print type(flow[0,0,0]) #num<type 'numpy.float32'>
				ch = 0xFF & cv2.waitKey(5)
				if ch == 27:
					break
			lsFlow.append(flow)
			i+=1
		else:
			break
	cv2.destroyAllWindows()
	out.release()
	npFlow=np.asarray(lsFlow,np.float16)
	print 'npFlow: ',npFlow.shape
	return npFlow


fn = '/home/shruti/caffe/shruti-pc/code/NEC-MVC-CD/recognition-module/dataset/1/01sample1.avi'
opticalFlow(fn,0.5,20,True)
