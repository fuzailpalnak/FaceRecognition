This dataset contains 1583 people with 202792 images.

The files for each person are put into the same folder under this person's name. 

The files in each folder are categorized into four types:
1. thumbnails: downsampled images for the web images with faces. These are for visualization purposes only. Please note that each image contains only one face as detected by a Haar detector.
2. info.txt: contains the "original web images" (OWI) for the thumbnail images. For each thumbnail, info.txt contains the following information:
   - information on the first line:
     -- number of duplicate URLs
     -- file name of corresponding thumbnail
     -- URL of the "original web image" where the thumbnail is downsampled from 
   - a list of duplicate URLs (one URL per line, total number of lines equal to "number of duplicate URLs" in the first line)
3. feature.bin: contains LBP features for the faces. The file starts with two int32 variables indicating the total number of faces and the dimension of LBP features, followed by a byte buffer storing all the features (one face after another).
4. filelist_LBP.txt: each line of this file contains a file name, corresponding to the order of features in feature.bin. The following four numbers on each line are the location of the faces in the OWIs, where the thumbnails are down-sampled (left top right bottom), and the last two numbers are the size of the OWIs. 
