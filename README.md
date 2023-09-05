#Computer Vision


The goal is to create a new perspective view sequence of a scene that links a pair of rectified images. This can be achieved by altering the pose of the camera and synthesizing the image that was acquired from the new camera viewpoint.

In each example, there is a pair of rectified images captured by an identical pair of cameras (with the same intrinsics), where the distance between the cameras (baseline) is approximately 10 centimeters (0.1 meters)


Two Images left and riight :

![image](https://github.com/SlimanJammal/CV_Photometric-Stereo-based-View-Synthesis/assets/100062609/581c8744-1b90-41f9-a192-12be9cf5dc53)




Output - Sequence of synthesized images:


![image](https://github.com/SlimanJammal/CV_Photometric-Stereo-based-View-Synthesis/assets/100062609/c613cdc9-b2b2-4b98-80a9-37e383adeae7)





Achieved by calculating: 


1. Left and Right Disparity Images -

   ![image](https://github.com/SlimanJammal/CV_Photometric-Stereo-based-View-Synthesis/assets/100062609/7f00f372-fd01-4fa5-a3d4-d5ee51893eac)




2. Left and Right Depth Images -

   ![image](https://github.com/SlimanJammal/CV_Photometric-Stereo-based-View-Synthesis/assets/100062609/857651b1-2853-4eab-97cf-d647a1839dc2)





