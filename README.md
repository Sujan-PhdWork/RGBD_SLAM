## Modified RGBD SLAM
This repository contains a modified version of RGBD SLAM, focusing on implementing the methodology outlined in "An Evaluation of the RGB-D SLAM System" by Felix Endres et al. In the original paper, the system closed loops by considering the previous three random frames. However, this implementation enhances the algorithm by incorporating local mapping and loop closing based on keyframes for robustness and accuracy in loop closure. This modification results in faster and more accurate pose estimation.

It's worth noting that this implementation doesn't prioritize 3D mapping, howerver mapping benefits greatly from improved pose estimation.



# Run Program
```python main,py ```


# Trajectory of freiburg1_xyz


<p align="center">
<img src="https://github.com/Sujan-PhdWork/RGBD_SLAM/assets/104067836/c9638a16-b498-43fa-9bcc-75300dbe52bd">
</p>

##

<p align="center">
<img src="https://github.com/Sujan-PhdWork/RGBD_SLAM/assets/104067836/87427721-7fe6-47b4-ab0c-cb08a49960fc">
</p>

# Trajectory of freiburg1_floor
<p align="center">
<img src="https://github.com/Sujan-PhdWork/RGBD_SLAM/assets/104067836/f18b9308-49b9-4d9d-ba06-6907104fbbf9">
</p>


