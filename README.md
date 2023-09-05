#Computer Vision


The goal is to create a new perspective view sequence of a scene that links a pair of rectified images. This can be achieved by altering the pose of the camera and synthesizing the image that was acquired from the new camera viewpoint.

In each example, there is a pair of rectified images captured by an identical pair of cameras (with the same intrinsics), where the distance between the cameras (baseline) is approximately 10 centimeters (0.1 meters)


Two Images left and riight :

![image](https://github.com/SlimanJammal/CV_Photometric-Stereo-based-View-Synthesis/assets/100062609/e802b899-ae15-4009-ac5c-7eef3e20e470)





Output - Sequence of synthesized images:

![image](https://github.com/SlimanJammal/CV_Photometric-Stereo-based-View-Synthesis/assets/100062609/ae3e5d8f-b49b-4d73-9fdd-ee23239725da)






Achieved by calculating: 


1. Left and Right Disparity Images -

   ![image](https://github.com/SlimanJammal/CV_Photometric-Stereo-based-View-Synthesis/assets/100062609/ff40ec60-c146-4ae4-8897-651cf96dce00)





2. Left and Right Depth Images -

 ![image](https://github.com/SlimanJammal/CV_Photometric-Stereo-based-View-Synthesis/assets/100062609/118b6bcf-88db-4da6-a5b5-c928933da94a)






