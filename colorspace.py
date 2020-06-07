import numpy as np
import cv2

xscale = 1
yscale = 1
tau = 20

fire_orig = cv2.imread('pics/fire4.jpg')
fire_orig = cv2.resize(fire_orig, (0,0), fx=1/2, fy=1/2)

rows_orig, cols_orig, channels_orig = fire_orig.shape

fire = cv2.resize(fire_orig, (0,0), fx=1/xscale, fy=1/yscale)
ycb = cv2.cvtColor(fire, cv2.COLOR_BGR2YCR_CB)

B, G, R = cv2.split(fire)
Y, Cr, Cb = cv2.split(ycb)

rows, cols, channels = ycb.shape

y_avg = np.mean(Y)
Cr_avg = np.mean(Cr)
Cb_avg = np.mean(Cb)

bitmask = np.zeros((rows,cols), dtype=np.uint8)
r5a = False
r5b = False
r5c = False
xcenter = 0
ycenter = 0
num_points = 0

for i in range(cols):
	for j in range(rows):

		r1 = (Y[j][i] >= Cb[j][i])
		r2 = (Cr[j][i] >= Cb[j][i])
		r3 = (Y[j][i] >= y_avg) and (Cr[j][i] >= Cr_avg) and (Cb[j][i] <= Cb_avg)
		r4 = (np.absolute(int(Cb[j][i]) - int(Cr[j][i])) > tau)
		r6 = (B[j][i] < G[j][i])
		r7 = (G[j][i] < R[j][i])

		# if (r1 and r2 and r3 and r4):
		if(True):
			# f1 = -2.62*(10**(-10))*Cr[j][i]**7 + 3.27*(10**(-7))*Cr[j][i]**6 \
			# 	-1.75*(10**(-4))*Cr[j][i]**5 + 5.16*(10**(-2))*Cr[j][i]**4 \
			# 	-9.10*(10**(0))*Cr[j][i]**3 - 5.60*(10**(4))*Cr[j][i] + 1.40*(10**(6))
			# f2 = -6.77*(10**(-8))*Cr[j][i]**5 + 5.50*(10**(-5))*Cr[j][i]**4 \
		 # 		-1.76*(10**(-2))*Cr[j][i]**3 + 2.78*(10**(-0))*Cr[j][i]**2 \
			# 	-2.15*(10**(2))*Cr[j][i] + 6.62*(10**(3))
			f3 = 1.80*(10**(-4))*Cr[j][i]**4 - 1.02*(10**(-1))*Cr[j][i]**3 \
				+21.66*(10**(0))*Cr[j][i]**2 - 2.05*(10**(3))*Cr[j][i] + 7.29*(10**(4))
			#r5a = (Cb[j][i] >= f1)
			#r5b = (Cb[j][i] <= f2)
			r5c = (Cb[j][i] <= 10*f3)
			if(r5c):
				bitmask[j][i] = True
				xcenter += i
				ycenter += j
				num_points += 1

xcenter = int(xcenter / num_points * xscale)
ycenter = int(ycenter / num_points * yscale)
if(num_points):
	cv2.circle(fire_orig, (xcenter, ycenter), 80, (255, 255, 255), 5)

fire_new = cv2.bitwise_and(fire,fire,mask = bitmask)

cv2.imshow('original', fire_orig)
new_restored = cv2.resize(fire_new, (0,0), fx=xscale, fy=yscale)
cv2.imshow('just the fire', new_restored)
cv2.waitKey(0)
cv2.destroyAllWindows
