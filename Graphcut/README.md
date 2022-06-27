<<<<<<< HEAD
# Computer-Vision---METHODS
=======
## In this repository, we tried  Graphcut, which is a simple but efficient type of segmentation of our data in three cases:
- Graphcut: OpenCV offers an out-of-the-box implementation of GrabCut see ``graphcut.ipynb``. We start similar to the original GrabCut version: a bounding rectangle containing the object of interest. Well, that is not exactly the expected result! Here, GrabCut did not have enough initial information to perform well and failed to retrieve an accurate mask for the image.
- We provide GrabCut with additional information (binarizing the original image and labeling as foreground all output black pixels) about the object of interest. This way, we help with the initial modeling of the foreground and the background of the image. 
- We apply Gaussian Mixture Model (GMM) to create a type of additional information and then we apply Graphcut (see ``images/seg.jpg``). Although the result is better than the other two methods, still it is not exactly what you want. Therefore, we apply connected components in OpenCV to remove the small unwanted component (see ``images/obj.jpg``).

### Our goal is to extract the object of the image and put it in the specified coordinate on the black background (see ``images/output.jpg``).
>>>>>>> f5f4cc1ef8aa3f73fb8238d13d62ac8e5e532d8d
