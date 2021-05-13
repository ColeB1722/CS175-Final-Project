import PySimpleGUI as sg
import cv2
import numpy as np
import os
from smoothing import L0Smoothing

#resize the image to fit
def resizeImg(video):
	w = video.shape[1]
	h = video.shape[0]

	if w > h:
		usage = w
	else:
		usage = h

	factor = 900

	scale = factor / float(usage)

	dim = (int(w * scale), int(h * scale))

	resized = cv2.resize(video, dim, cv2.INTER_AREA)
	return resized

# function to create interface
def main():

	# select theme
	sg.theme('LightGray1')

	# ------ Get the filename ----- #
	filename = sg.popup_get_file('Filename to play')
	if filename == '':
		raise NameError('Please select a file')

	#retrieve useful data about the filename
	length = len(filename)
	count = 0
	for char in reversed(filename):
		if char == '/':
			break
		count += 1

	fname = filename[length - count: length - 4]
	ext = filename[length - 4:length]
	
	# depending on file type, save image in usable format
	
	# determine whether input is still or video 
	toggle = True
	if ext == '.mp4':
		rem = cv2.VideoCapture(filename)
		vidFile = rem
		num_frames = vidFile.get(cv2.CAP_PROP_FRAME_COUNT)

		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

		fps = rem.get(cv2.CAP_PROP_FPS)

	elif ext == '.jpg' or ext == '.png':
		vidFile = cv2.imread(filename)
		toggle = False
		num_frames = 1

		vidFile = resizeImg(vidFile)
	else:
		print('Only accepting .png, .jpg, or .mp4')

	# define the window layout

	# organize the first column, which displays the image
	image = [
		[
			sg.Button('Separate Foreground from Background', visible=not toggle),
			sg.Checkbox('Render', default=False, size=(10,1), key='-RENDER-')
		],
		[sg.Image(filename='', key='-IMAGE-', size=(100,100))],
		[
			sg.Button("Play/Pause", visible=toggle),
			sg.Slider(range=(0, num_frames),
					  size=(60, 10), enable_events=True, orientation='h', visible=toggle, key='-PLAY SLIDER-')
		],
		[sg.Button('Exit', size=(10, 1)),
		sg.Button('Save', size=(10, 1), visible=not toggle),
		sg.InputText(key='-TITLE-', visible=not toggle)]

	]

	# organize the second column, which contains all the effects

	# filtering functionality
	effects = [
		[sg.Text('Effects')],
		[
			sg.Checkbox('L0 Image Smoothing', default=False, size=(20, 1), key='-L0-'),
			sg.Text('Lambda'),
			sg.Slider((1e-3, 1e-1), 2e-2, .01, orientation='h', size=(20,15), key='-L0 SLIDER A-'),
			sg.Text('Kappa'),
			sg.Slider((2, 8), 2, 1, orientation='h', size=(20,15), key='-L0 SLIDER B-')
		],
		[
			sg.Checkbox('Bilateral Filter', default=False, size=(20, 1), key='-BIFILTER-'),
			sg.Slider((1, 10), 4, 1, orientation='h', size=(20, 15), key='-BIFILTER SLIDER A-'),
			sg.Slider((5, 200), 100, 5, orientation='h', size=(20, 15), key='-BIFILTER SLIDER B-')
		],
		[
			sg.Checkbox('Edge Preservation', default=False, size=(20, 1), key='-EPRES-'),
			sg.Slider((0, 200), 100, 1, orientation='h', size=(20, 15), key='-EPRES SLIDER A-'),
			sg.Slider((0, 1), .5, .1, orientation='h', size=(20, 15), key='-EPRES SLIDER B-')
		],
		[
			sg.Checkbox('Pencil Sketch', default=False, size=(20, 1), key='-PENCIL-'),
			sg.Slider((0, 200), 200, 1, orientation='h', size=(13, 15), key='-PENCIL SLIDER A-'),
			sg.Slider((0, 1), .5, .1, orientation='h', size=(14, 15), key='-PENCIL SLIDER B-'),
			sg.Slider((0, 0.1), .01, .01, orientation='h', size=(13, 15), key='-PENCIL SLIDER C-')
		],
		[
			sg.Checkbox('Stylization', default=False, size=(20, 1), key='-STYLE-'),
			sg.Slider((0, 200), 100, 1, orientation='h', size=(20, 15), key='-STYLE SLIDER A-'),
			sg.Slider((0, 1), .5, .1, orientation='h', size=(20, 15), key='-STYLE SLIDER B-')
		],
		[
			sg.Checkbox('Threshold', default=False, size=(20, 1), key='-THRESH-'),
			sg.Slider((0, 255), 128, 1, orientation='h', size=(40, 15), key='-THRESH SLIDER-')
		],
		[	
			sg.Checkbox('Canny', default=False, size=(20, 1), key='-CANNY-'),
			sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER A-'),
			sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER B-')
		],
		[
			sg.Checkbox('Guassian Blur', default=False, size=(20, 1), key='-BLUR-'),
			sg.Slider((1, 11), 1, 1, orientation='h', size=(40, 15), key='-BLUR SLIDER-')
		],
		[
			sg.Checkbox('Hue Shift', default=False, size=(20, 1), key='-HUE-'),
			sg.Slider((0, 225), 0, 1, orientation='h', size=(40, 15), key='-HUE SLIDER-')
		],
		[
			sg.Checkbox('Enhance', default=False, size=(20, 1), key='-ENHANCE-'),
			sg.Slider((0, 200), 100, 1, orientation='h', size=(20, 15), key='-ENHANCE SLIDER A-'),
			sg.Slider((0, 1), .5, .1, orientation='h', size=(20, 15), key='-ENHANCE SLIDER B-')
		]
	]

	# tone shifting functionality
	tone_maps = ['None', 'Autumn', 'Bone', 'Jet', 'Winter', 'Rainbow', 'Ocean', 'Summer', 
				 'Spring', 'Cool', 'HSV', 'Pink', 'Hot']
	tones = [[sg.Combo(tone_maps, size=(10,1), key='-TONE-', default_value='None')]]
	tone_adj = [
			[sg.Radio('Gamma Correction', 'tone', size=(20,1), key='-GAMMA-'),
			sg.Slider((.1,5), 1, .1, orientation='h', size=(40,15), key='-GAMMA PARAM-')],
			[sg.Radio('Histogram Equalization', 'tone', size=(20,1), key='-HISTO EQUAL-')],
			[sg.Radio('Constrast Stretching', 'tone', size=(20,1), key='-CONSTRAST-')]
	]

	# combine effects columns together
	layout_right = [
		[sg.Text('Color Tones')],
		[sg.Column(tones)],
		[sg.Column(tone_adj)],
		[sg.HSeparator()],
		[sg.Column(effects)]
	]

	# create final layout
	layout_f = [
		[
    		sg.Column(image),
    		sg.VSeparator(),
    		sg.Column(layout_right) 
    	]
    ]

	# create the window
	window = sg.Window('OpenCV Integration', layout_f, location=(1000, 500))

	# locate the elements we'll be updating
	image_elem = window['-IMAGE-']
	slider_elem = None
	if toggle:
		slider_elem = window['-PLAY SLIDER-']

    # ----- LOOP through video file by frame ----- #
	cur_frame = 0
	paused = True

	# relevant data about tones
	num = {'Autumn': cv2.COLORMAP_AUTUMN, 'Bone': cv2.COLORMAP_BONE, 
		   'Jet': cv2.COLORMAP_JET, 'Winter': cv2.COLORMAP_WINTER, 
		   'Rainbow': cv2.COLORMAP_RAINBOW, 'Ocean': cv2.COLORMAP_OCEAN, 
		   'Summer': cv2.COLORMAP_SUMMER, 'Spring': cv2.COLORMAP_SPRING, 
		   'Cool': cv2.COLORMAP_COOL, 'HSV': cv2.COLORMAP_HSV, 
		   'Pink': cv2.COLORMAP_PINK, 'Hot': cv2.COLORMAP_HOT}   

	vid = []

	while True:
		
		# get information about window
		event, values = window.read(timeout=10)

		render = values['-RENDER-']

		# separate background from foreground
		if event == 'Separate Foreground from Background':
			inname = "./Images/" + fname + ext
			outname = "./Images/" + fname + '_foreground.png'
			os.system('python3 seg.py {} {} 1'.format(inname, outname))
			vidFile = cv2.imread(outname)

		# get relevant image info, depending on type 
		if toggle:
			ret, frame = vidFile.read()

			if isinstance(frame, type(None)):
				if render: 
					print("Finished rendering video!")
				
				rem.set(cv2.CAP_PROP_POS_FRAMES, num_frames-1)
				ret, frame = rem.read()
				paused = True
		else: 
			frame = vidFile

		if event == 'Exit' or event == sg.WIN_CLOSED:
			break

		frame = resizeImg(frame)
		
		# if someone moved the slider manually, the jump to that frame
		if event == '-PLAY SLIDER-' or int(values['-PLAY SLIDER-']) != cur_frame:
			cur_frame = int(values['-PLAY SLIDER-'])

			if toggle:
				vidFile.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
			else:
				rem.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
				ret, frame = rem.read()
				frame = resizeImg(frame)

		if paused:
			vidFile = frame
			toggle = False

		if event == 'Play/Pause':
			if not paused:
				vidFile = frame
				paused = True
				toggle = False
			else:
				vidFile = rem
				paused = False
				toggle = True

		if values['-TONE-'] != 'None':
			frame = cv2.applyColorMap(frame, num[values['-TONE-']])

		if values['-GAMMA-']:
			gamma = values['-GAMMA PARAM-']

			invGamma = 1.0 / gamma
			table = np.array([((i / 255.0) ** invGamma) * 255
				for i in np.arange(0, 256)]).astype("uint8")

    		# apply gamma correction using the lookup table
			frame = cv2.LUT(frame, table)
		elif values['-HISTO EQUAL-']:
			img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

			# equalize the histogram of the Y channel
			img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

			# convert the YUV image back to RGB format
			frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
		elif values['-CONSTRAST-']:
			norm_img1 = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			norm_img2 = cv2.normalize(frame, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			# scale to uint8
			norm_img1 = (255*norm_img1).astype(np.uint8)
			norm_img2 = np.clip(norm_img2, 0, 1)
			frame = (255*norm_img2).astype(np.uint8)

		# apply filters
		if values['-L0-']:
			frame = L0Smoothing(frame, float(values['-L0 SLIDER A-']), int(values['-L0 SLIDER B-']))
		if values['-BIFILTER-']:
			frame = cv2.bilateralFilter(frame, int(values['-BIFILTER SLIDER A-']), values['-BIFILTER SLIDER B-'], values['-BIFILTER SLIDER B-'])
		if values['-EPRES-']:
			frame = cv2.edgePreservingFilter(frame, flags=1, sigma_s=values['-EPRES SLIDER A-'], sigma_r=values['-EPRES SLIDER B-'])
		if values['-PENCIL-']:
			_, frame = cv2.pencilSketch(frame, sigma_s=values['-PENCIL SLIDER A-'], sigma_r=values['-PENCIL SLIDER B-'], shade_factor=values['-PENCIL SLIDER C-'])
		if values['-STYLE-']:
			frame = cv2.stylization(frame, sigma_s=values['-STYLE SLIDER A-'], sigma_r=values['-STYLE SLIDER B-'])
		if values['-THRESH-']:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
			frame = cv2.threshold(frame, values['-THRESH SLIDER-'], 255, cv2.THRESH_BINARY)[1]
		if values['-CANNY-']:
			frame = cv2.Canny(frame, values['-CANNY SLIDER A-'], values['-CANNY SLIDER B-'])
		if values['-BLUR-']:
			frame = cv2.GaussianBlur(frame, (21, 21), values['-BLUR SLIDER-'])
		if values['-HUE-']:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			frame[:, :, 0] += int(values['-HUE SLIDER-'])
			frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
		if values['-ENHANCE-']:
			frame = cv2.detailEnhance(frame, sigma_s=values['-ENHANCE SLIDER A-'], sigma_r=values['-ENHANCE SLIDER B-'])
		
		# save image
		if event == 'Save':
			out = values['-TITLE-']
			if ext not in values['-TITLE-']:
				if out != '':
					out = values['-TITLE-'] + ext
				else:
					out = fname + "_edited" + ext
	
			cv2.imwrite('./Images/' + out, frame)

		if toggle:
			slider_elem.update(cur_frame)
			cur_frame += 1

			if render:
				if cur_frame >= len(vid):
					vid.append(frame)
				else:
					vid.insert(cur_frame, frame)

		# update
		imgbytes = cv2.imencode('.png', frame)[1].tobytes()
		window['-IMAGE-'].update(data=imgbytes)
		image_elem.update(data=imgbytes)

	window.close()

	return vid, fps


video, fps = main()
video = np.array(video)
width = len(video[0])
height = len(video[0][0])
out = cv2.VideoWriter('./Images/render.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (height, width))

for i in range(len(video)):
	out.write(video[i])

out.release