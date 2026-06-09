# Day 12 - Series 1: Perception Capstone

> MS Robotics and Autonomous Systems, Arizona State University, Dec 2026

---

## What Series 1 Is

Eleven perception projects built in sequence. Each one answers a question the previous one raised. Each one uses real sensor data, produces real numbers, and finds something that was not obvious before I measured it.

This is not a tutorial collection. Every project has a novel finding. Every finding connects to the next.

---

## Series 1 Capstone Video

[![Series 1 Capstone](https://drive.google.com/thumbnail?id=1s65BKqvUQzPcdGB2qv0QXHKWCKKKOO5K&sz=w1280)](https://drive.google.com/file/d/1s65BKqvUQzPcdGB2qv0QXHKWCKKKOO5K/view)

*All 11 projects compiled into one video. Every visual output from every project. Click to watch.*

---

## The 11 Projects

### Day 01 - LiDAR Obstacle Detection

The starting point. Raw Velodyne HDL-64E point clouds, voxelization, RANSAC ground removal, DBSCAN clustering. 28 objects detected per frame in 86ms on real KITTI data. Established the baseline pipeline that every subsequent project builds on.

```
28 objects per frame
86ms end-to-end pipeline
KITTI HDL-64E sensor
```

---

### Day 02 - Stereo Camera Depth Safety

Ran stereo disparity estimation on KITTI stereo pairs. Measured depth accuracy at every range band. The result was clear: cameras achieve 1.04m MAE at 0-10m but become unreliable beyond 35m even in clear weather with no rain, no fog, no adverse conditions.

```
MAE 1.04m at 0-10m
MAE 8.81m beyond 50m
Reliable only to 35m
```

This finding set up the entire sensor fusion argument for Day 8.

---

### Day 03 - PointPillars 3D Detector

Built PointPillars from scratch. Pillar feature extraction, pseudo-image projection, 2D detection head. Trained on KITTI. 98.9% loss reduction over 8 epochs. 2.5 million parameters. No pretrained weights. Understanding the architecture at this level made every subsequent deep learning project faster to build.

```
2.5M parameters built from scratch
98.9% training loss reduction
KITTI 3D object detection
```

---

### Day 04 - Multi-Camera BEV Perception

Six cameras fused into a single bird's eye view representation using IPM homography projection. Ran detection on nuScenes Singapore. 178 objects visible simultaneously from a viewpoint no single camera can see. This is the same representation Tesla uses in their vision-only stack.

```
178 objects from 6 cameras
Bird's eye view fusion
nuScenes Singapore dataset
```

---

### Day 05 - Multi-Object Tracking SORT

Implemented SORT tracking on top of KITTI detections. Kalman filter state prediction, Hungarian algorithm assignment, IoU-based matching. Achieved 95.1% MOTP which beats the original paper's reported 77.5%. The key insight: the tracker is not the bottleneck. The detector is. A fast imperfect tracker on good detections outperforms a sophisticated tracker on weak detections.

```
95.1% MOTP, beats paper 77.5%
1158 FPS tracker throughput
Detector is the bottleneck
```

---

### Day 06 - Semantic Segmentation ROS2

Deployed DeepLabV3 inside a ROS2 Humble node. Measured real inference time including ROS2 message overhead, not just raw PyTorch inference. 52.6 FPS on RTX 4050. Measured warmup cost separately and found the first frame takes 3.2x longer than steady state, something most benchmarks ignore.

```
52.6 FPS on RTX 4050
ROS2 Humble deployment
Warmup cost measured: 3.2x first frame
```

---

### Day 07 - Adverse Weather Perception

Applied Marshall-Palmer rain attenuation and Koschmieder fog extinction models to real KITTI LiDAR scans. Tested 42 rain and fog combinations. The finding surprised me. Rain at 100mm/hr, which is Phoenix monsoon level, is survivable. Fog below 75m visibility is not. They are not symmetric failures and they should not be treated the same way in safety documentation.

```
Rain safe at 100mm/hr
Fog unsafe below 75m visibility
42 conditions tested, ODD boundary measured
```

---

### Day 08 - LiDAR-Camera Depth Completion

Took the Day 2 finding (cameras unreliable beyond 35m) and the Day 7 finding (LiDAR survives all rain) and built the fusion that covers both. Sparse LiDAR projected onto camera image, dense completion network fills gaps. 44x MAE improvement at 0-10m over camera-only. 100% spatial coverage from 1.3% sparse input.

```
44x MAE improvement 0-10m
100% coverage from 1.3% sparse input
108-frame demo video
```

---

### Day 09 - Domain Shift Analysis

Took the same detector trained on KITTI Germany and deployed it on nuScenes Singapore. 58.4% detection drop. I then spent time finding the root cause. It was not the city. It was not the scene complexity. It was the sensor. HDL-64E produces 121k points per scan. HDL-32E produces 34k points. The detector learned point density patterns specific to one sensor and fails when those patterns change.

```
58.4% detection drop across datasets
Root cause: sensor, not scene
HDL-64E 121k pts vs HDL-32E 34k pts
```

---

### Day 10 - Neural Occupancy Network

Built a 4-state voxel uncertainty system. FREE-CONFIRMED means a laser beam passed through the voxel and confirmed it empty. FREE-ASSUMED means nothing confirmed it but nothing contradicted it either. These are fundamentally different states and treating them the same way in path planning is how autonomous vehicles end up driving into occluded objects.

Trained a 3D CNN on 400k KITTI samples. 100% accuracy. Found the unsafe planning boundary: at 40m from the sensor, FREE-CONFIRMED and FREE-ASSUMED voxels equalize. The system cannot tell safe space from uncertain space beyond that range.

```
4-state uncertainty classification
Unsafe planning boundary at 40m
3D CNN 72k parameters, 100% accuracy
282ms pipeline on RTX 4050
```

---

### Day 11 - ASU Campus Perception

Took a KITTI-trained YOLOv8 detector and deployed it on real footage I filmed at ASU Tempe campus. Measured five types of failure systematically.

The most important finding: sun glare causes a 56.5% detection rate collapse. I then ran YOLOv8x, the 68 million parameter version, 20 times larger than nano, on the same footage. It performed worse in glare than the small model. Model size does not fix a training distribution mismatch. Arizona sun is simply outside the distribution of overcast German highway footage. No amount of scaling resolves that.

```
56.5% detection drop in Arizona sun
YOLOv8x worse than nano in glare
Golf cart: 0 detections, vocabulary gap
2565 frames across 8 campus scenarios
```

---

## The Connected Story

```
Day 2: cameras fail beyond 35m
Day 7: LiDAR survives rain, fails in dense fog
Day 8: fuse them together, cover both failures

Day 9: same detector, different city, 58.4% drop
Day 11: same detector, different continent, same problem
        root cause confirmed: distribution mismatch

Day 10: not just where objects are
        but where space is safe to plan through
        the representation that ties perception to planning
```

---

## Series 1 Summary

| Project | Key Finding |
|---------|-------------|
| Day 01 LiDAR Detection | 86ms, 28 objects, KITTI baseline |
| Day 02 Stereo Depth | Camera unsafe beyond 35m |
| Day 03 PointPillars | 98.9% loss reduction from scratch |
| Day 04 BEV Perception | 178 objects, 6 cameras fused |
| Day 05 MOT SORT | 95.1% MOTP, beats published paper |
| Day 06 Segmentation | 52.6 FPS in live ROS2 pipeline |
| Day 07 Adverse Weather | Fog unsafe below 75m, rain survivable |
| Day 08 Depth Completion | 44x MAE improvement, 100% coverage |
| Day 09 Domain Shift | 58.4% drop, sensor not scene |
| Day 10 Occupancy Network | Unsafe planning boundary at 40m |
| Day 11 ASU Perception | Scaling cannot fix domain shift |

---

## All Project Repositories

| Project | Repository |
|---------|------------|
| Day 01 | [day-001-lidar-obstacle-detection](https://github.com/GVK-Engine/day-001-lidar-obstacle-detection) |
| Day 02 | [day-002-stereo-depth-analysis](https://github.com/GVK-Engine/day-002-stereo-depth-analysis) |
| Day 03 | [day-003-pointpillars-3d-detector](https://github.com/GVK-Engine/day-003-pointpillars-3d-detector) |
| Day 04 | [day-004-bev-perception](https://github.com/GVK-Engine/day-004-bev-perception) |
| Day 05 | [day-005-multi-object-tracking](https://github.com/GVK-Engine/day-005-multi-object-tracking) |
| Day 06 | [day-006-semantic-segmentation](https://github.com/GVK-Engine/day-006-semantic-segmentation) |
| Day 07 | [day-007-adverse-weather-perception](https://github.com/GVK-Engine/day-007-adverse-weather-perception) |
| Day 08 | [day-008-depth-completion](https://github.com/GVK-Engine/day-008-depth-completion) |
| Day 09 | [day-009-domain-shift](https://github.com/GVK-Engine/day-009-domain-shift) |
| Day 10 | [day-010-occupancy-network](https://github.com/GVK-Engine/day-010-occupancy-network) |
| Day 11 | [day-011-asu-perception](https://github.com/GVK-Engine/day-011-asu-perception) |

---

## Run the Capstone Video Builder

```bash
git clone https://github.com/GVK-Engine/day-012-series1-capstone
cd day-012-series1-capstone
pip install -r requirements.txt
```

Update all project paths in `capstone_video.py` to match your local directory structure.

```bash
py -3.11 capstone_video.py
```

Builds `series1_capstone.mp4` pulling every visual output from all 11 project folders.

---

## Stack

`Python 3.11` `PyTorch 2.6` `OpenCV` `Matplotlib` `imageio` `NumPy` `YOLOv8` `ROS2 Humble` `KITTI` `nuScenes`

---

## What Comes Next

Series 1 covers the full perception stack from raw sensor data to world representation. Series 2 builds planning and control on top of it. Series 3 integrates everything into a running autonomous system in simulation.

The perception problems measured in Series 1, domain shift, adverse weather, vocabulary gaps, uncertainty quantification, are exactly the problems that make deploying AV systems in new cities hard. Series 2 starts knowing where the perception boundaries are.
