I downloaded the CT scanned images of rock sample from publicly available website Zenodo. 

The data which I collected is in TIF (Tagged Image File) format. So, I converted the files into 
JPG format which is more comfortable. Converted images are then resized to 640x640 pixels 
using OpenCV-python library.

CT scanned images are now annotated or labelled manually using opensource makesence.ai 
tool. This tool provides the co-ordinates of annotated image and store them in a VGG formatted 
Json file or COCO formatted Json file.

Now the Json file should be parsed to get the co-ordinated of the annotated fractures and stored 
them in the txt file which is compactable for Yolo v11 model.

I split the 140 CT scanned images into train, validation and test datasets. 100 images are for 
training and 40 images are for testing (~30%). In training dataset again 20 images are split into 
validation dataset.
