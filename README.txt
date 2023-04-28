CS 4375.503 Final Project
Stephen Wells

------ Libraries used/required: ---------
numpy, pandas, tensorflow, sklearn, matplotlib, keras, seaborn

--- Files included: -----
CNN.py, ProjectMain.py
CNN.py is just my own partial implementation of a convolutional neural net to get insight into how they work.
I stopped prior to implementing the whole CNN since it was time consuming and wouldn't have had the same accuracy
which would detract from the main subject, human assisted machine learning 

The relevant code is in ProjectMain.py which uses Keras to construct a CNN, and implements uncertainty and
random sampling. It does not depend on CNN.py

---- Dataset information ** important ** ----
Since the dataset is thousands of images, it is too big to host online and download on demand. I tried and it 
took too long. All the data used is on github and can be cloned (along with the python files)
and then they are grabbed locally which in my experience was faster, especially if running the program multiple times.
As a result the filepaths used in the program are relative to the ProjectMain.py file. 

The program assumes a file structure like: 
Project
[	ProjectMain.py
[	trainedmodel.h5
[	dataset
	[	mixed_set
	[	test
		[	cat
		[	dog
	[	small_training_set
		[	cats
		[	dogs
	[	training_set
		[	cats
		[	dogs
		
** Relevant Github repo:

--- Using the program ---
When you run ProjectMain it will search for trainedmodel.h5, a pretrained model meant to spare you the time training_set
If it doesnt find the file, it will train the model from scratch. The pretrained file is submitted on eLearning and can be
found on github: 

After training, choose to perform random sampling (automated), uncertainty sampling (automated), or manual
uncertainty sampling, where it will find uncertain images and ask you how many you want to label. You label the images
with 'c' or 'd' and once finished it will train the model using your data. 




