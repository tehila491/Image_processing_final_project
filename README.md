# Evaluating Robustness of Computer Vision Models 👁️

## 1. Project Overview

This project evaluates the robustness of classical image processing algorithms and deep learning models under different image distortions.  
The goal is to analyze how visual degradation affects computer vision performance across low-level, mid-level, and high-level tasks.

We evaluate three representative tasks:

1. **Low-level vision:** Edge Detection using the Canny Edge Detector  
2. **Mid-level vision:** Foreground Segmentation using GrabCut  
3. **High-level vision:** Multi-label Image Classification using ResNet50  

The selected distortions are:

- **Salt & Pepper Noise**
- **Overexposure**
- **Motion Blur**

For each task, we first measure performance on clean images, then evaluate performance degradation under distorted images, and finally test whether image restoration/enhancement methods can recover part of the lost performance.  
For the deep learning task, we also evaluate model behavior under distortions and compare it with the clean-image baseline.

**Team Members:** Tehila Segal, Hadar Semama

## 2. Project Decisions & Setup

Based on the project requirements, we selected a public dataset, three computer vision tasks, three distortion types, and suitable evaluation metrics for each task.

| Category | Selection | Description |
| :--- | :--- | :--- |
| **Dataset** | [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) | Public computer vision dataset containing RGB images with object labels, bounding boxes, and segmentation masks. |
| **Task 1 - Low-Level Vision** | Edge Detection | **Algorithm:** Canny Edge Detector |
| **Task 2 - Mid-Level Vision** | Foreground Segmentation | **Algorithm:** GrabCut initialized using ground-truth bounding boxes |
| **Task 3 - High-Level Vision** | Multi-Label Image Classification | **Model:** ResNet50 initialized with ImageNet pretrained weights |
| **Distortions** | Salt & Pepper Noise, Overexposure, Motion Blur | Applied at multiple severity levels to evaluate robustness. |

## 3. Exploratory Data Analysis (EDA)
### Clean Images with Ground Truth Bounding Boxes
The figure below shows clean PASCAL VOC images with their ground-truth bounding box annotations and object class labels.
<img width="1189" height="946" alt="image" src="https://github.com/user-attachments/assets/ca2f3a3a-531c-44ee-b529-6309531b4400" />

### Class Distribution
The following plot shows the distribution of object classes in the selected PASCAL VOC split.  
This helps verify that the evaluation includes a variety of object categories.
<img width="1190" height="590" alt="image" src="https://github.com/user-attachments/assets/058e2453-c825-4cc1-b12c-efa295522d67" />

### Ground Truth Segmentation Masks

The PASCAL VOC segmentation masks are used as ground truth for evaluating the GrabCut foreground segmentation task.  
For each selected image, we extract the binary mask of the target object class and compare the GrabCut output against this mask using Segmentation IoU.

<img width="1389" height="1166" alt="image" src="https://github.com/user-attachments/assets/5bb17913-b5b0-4bb8-a0f4-6ddb0d4d5599" />

## 4. Baseline Performance (Clean Images)

Before applying distortions, we evaluated all three selected tasks on clean PASCAL VOC 2012 images.  
This stage establishes the clean baseline performance, which will later be used as the reference point for measuring degradation under distortions and recovery after enhancement or fine-tuning.

| Task / Model | Evaluation Metric | Baseline Score |
| :--- | :--- | :--- |
| Canny Edge Detection | Mean Edge Density |9.68% |
| GrabCut Segmentation | Mean IoU | 65.96% |
| ResNet50 Multi-Label Classification | Mean Multi-label IoU / Jaccard | 57.44% |

### Baseline Methodology

For the clean-image baseline, we evaluated three different computer vision tasks:

1. **Canny Edge Detection** was applied to grayscale versions of the clean images.  
   Since edge detection does not directly produce object-level predictions, we report the mean edge density as a low-level structural measurement.

2. **GrabCut Foreground Segmentation** was initialized using the bounding box of the largest annotated object in each image.  
   The predicted foreground mask was compared against the PASCAL VOC segmentation ground truth using Intersection over Union (IoU).

3. **ResNet50 Multi-Label Classification** was initialized with ImageNet pretrained weights and adapted to the 20 PASCAL VOC object classes.  
   Since PASCAL VOC images may contain more than one object class, we evaluate the model using mean multi-label IoU / Jaccard score rather than Top-1 accuracy.

---

### Overall Baseline Summary

The following plot summarizes the clean-image baseline scores for the three selected tasks.

<img width="790" height="470" alt="overall_baseline_clean" src="https://github.com/user-attachments/assets/7ded7ede-e048-4a81-837a-5330f270e44a" />

---

### ResNet50 Training Curve

The ResNet50 model was fine-tuned on clean PASCAL VOC images before being used as the clean baseline classifier.  
The training and validation loss curves are shown below.

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/ea4f44f1-0f31-432c-bae8-7d3c968bcfae" />


---

### Qualitative Baseline Results

The figure below shows examples of the clean-image baseline outputs for the three tasks.  
Each row presents the original image, Canny edge map, GrabCut foreground extraction, ResNet50 multi-label prediction, and the ground-truth segmentation mask.

<img width="2500" height="2000" alt="image" src="https://github.com/user-attachments/assets/32c957e9-a842-4fca-a8e8-139d8a5cc279" />


---

### Per-Class Baseline Performance on Clean Images

In addition to the overall baseline scores, we also report per-class performance.  
This helps identify which object categories are easier or harder for each method.  
In the IoU plots, the dashed horizontal line represents the mean IoU (mIoU) across the evaluated classes.

#### GrabCut - IoU per Class

GrabCut was evaluated by comparing the predicted foreground mask with the PASCAL VOC segmentation ground truth.  
The method performs better on objects with clear foreground-background separation and usually performs worse on complex scenes, thin structures, or objects with ambiguous boundaries.

<img width="1484" height="730" alt="image" src="https://github.com/user-attachments/assets/a3ab4917-8424-45da-ae2b-b9cc6e162029" />


#### ResNet50 - Multi-Label IoU / Jaccard per Class

The ResNet50 classifier was evaluated using multi-label IoU / Jaccard score for each of the 20 PASCAL VOC classes.  
This per-class analysis shows which categories are recognized more reliably on clean images and which classes are more challenging.

<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/c1f23fe8-a1bc-41b8-843f-3af5a16a9191" />


#### Canny - Edge Density per Class

Canny edge detection does not output semantic labels or object masks.  
Therefore, for this low-level task we report the average edge density per class, which measures the proportion of detected edge pixels in the clean images.

<img width="1484" height="730" alt="image" src="https://github.com/user-attachments/assets/7ad12be1-c86d-48fa-9096-cf0991d89219" />


---

### Clean Baseline Summary

Overall, the clean-image baseline shows that GrabCut achieves reasonable foreground segmentation performance when initialized with ground-truth bounding boxes, while ResNet50 provides a meaningful multi-label classification baseline on clean images.  
These clean-image results are used as the reference point for the next stage, where we evaluate performance degradation under Salt & Pepper noise, overexposure, and motion blur.


## 5. Evaluating Robustness on Distorted Images

After establishing the clean-image baseline, we evaluated the robustness of the three selected methods under image distortions.  
Each clean image was distorted using three different distortion types, and each distortion was applied at six increasing severity levels.

The evaluated distortions were:

1. **Salt & Pepper Noise** - random black and white pixels are inserted into the image.
2. **Overexposure** - brightness and intensity are increased, causing saturated and washed-out regions.
3. **Motion Blur** - directional blur is applied to simulate camera or object motion.

For each distorted image, we measured the performance of the three selected tasks and compared the results to the clean-image reference.

---

### Distortion Levels and SNR

Each distortion was applied at six severity levels.  
For every distorted image, we computed the Signal-to-Noise Ratio (SNR) relative to the original clean image.  
Lower SNR indicates a stronger distortion.

<img width="3426" height="1431" alt="image" src="https://github.com/user-attachments/assets/6b76ab9b-ffad-4ee4-9628-7446c41882d7" />

The figure above shows visual examples of the three distortions across severity levels 1 to 6.  
As the severity level increases, the visual quality decreases and the SNR becomes lower.

---

### Evaluation Metrics

We used task-specific metrics because each computer vision task produces a different type of output.

| Task / Model | Metric on Distorted Images | Meaning |
| :--- | :--- | :--- |
| Canny Edge Detection | Edge-Map IoU vs Clean Reference | Measures how similar the distorted edge map is to the clean edge map |
| GrabCut Segmentation | Segmentation IoU vs Ground Truth | Measures overlap between the predicted foreground mask and the PASCAL VOC segmentation mask |
| ResNet50 Multi-Label Classification | Multi-label F1-score | Measures classification quality when multiple object labels may exist in the same image |
| Distortion Strength | SNR | Lower SNR means stronger image degradation |

For **Canny**, PASCAL VOC does not provide ground-truth edge maps.  
Therefore, we use the clean Canny edge map as the reference and measure how much the edge structure changes under distortions.

For **GrabCut**, the predicted foreground mask is compared directly against the PASCAL VOC segmentation ground truth using IoU.

For **ResNet50**, we use multi-label F1-score because PASCAL VOC images may contain multiple object classes.

---

### Performance Degradation vs. SNR

The following plots show how each method behaves as the distortion severity increases and the SNR decreases.  
The red dashed line represents the clean reference or clean baseline.

#### Canny - Edge-Map IoU vs Clean Reference

<img width="1334" height="882" alt="image" src="https://github.com/user-attachments/assets/08bcafb0-6a3c-454a-9c36-146e5bd1a5ca" />


Canny is highly sensitive to distortions.  
Salt & Pepper noise introduces many false edges, while Motion Blur removes sharp edges, causing a strong decrease in edge-map similarity.

#### GrabCut - Segmentation IoU vs Ground Truth

<img width="1334" height="882" alt="image" src="https://github.com/user-attachments/assets/d1427778-24f7-4208-b9db-824483c24192" />

GrabCut performance decreases when image distortions damage foreground/background color separation or object boundaries.  
The degradation is strongest under severe Salt & Pepper noise and severe Overexposure.

#### ResNet50 - Multi-Label F1-Score

<img width="1334" height="882" alt="image" src="https://github.com/user-attachments/assets/15a6e144-20f3-4049-b59b-dfe496372380" />


ResNet50 is relatively stable under mild distortions, but its performance drops under severe Motion Blur and strong Salt & Pepper noise.  
This shows that the high-level classification model is more robust than Canny at mild distortion levels, but still sensitive to strong image degradation.

---

### Quantitative Results Across Distortion Levels

The table below summarizes the average performance for each distortion type and severity level.

| Condition | Mean SNR | Canny Edge-Map IoU | GrabCut Segmentation IoU | ResNet50 Multi-Label F1 |
| :--- | ---: | ---: | ---: | ---: |
| Clean | inf | 1.000 | 0.653 | 0.842 |
| Motion Blur L1 | 24.05 | 0.556 | 0.666 | 0.855 |
| Motion Blur L2 | 19.79 | 0.247 | 0.630 | 0.796 |
| Motion Blur L3 | 18.24 | 0.174 | 0.599 | 0.713 |
| Motion Blur L4 | 17.22 | 0.136 | 0.575 | 0.677 |
| Motion Blur L5 | 15.68 | 0.092 | 0.540 | 0.607 |
| Motion Blur L6 | 14.26 | 0.060 | 0.517 | 0.380 |
| Overexposure L1 | 16.08 | 0.879 | 0.658 | 0.845 |
| Overexposure L2 | 8.81 | 0.715 | 0.620 | 0.836 |
| Overexposure L3 | 4.81 | 0.516 | 0.561 | 0.813 |
| Overexposure L4 | 2.56 | 0.368 | 0.505 | 0.823 |
| Overexposure L5 | 0.97 | 0.252 | 0.419 | 0.713 |
| Overexposure L6 | -0.02 | 0.162 | 0.304 | 0.643 |
| Salt & Pepper L1 | 21.97 | 0.730 | 0.631 | 0.848 |
| Salt & Pepper L2 | 18.98 | 0.586 | 0.566 | 0.880 |
| Salt & Pepper L3 | 14.25 | 0.341 | 0.515 | 0.773 |
| Salt & Pepper L4 | 10.66 | 0.201 | 0.475 | 0.750 |
| Salt & Pepper L5 | 7.53 | 0.130 | 0.451 | 0.590 |
| Salt & Pepper L6 | 4.86 | 0.101 | 0.297 | 0.447 |

Some mild distortions slightly improve a score compared to the clean case.  
This can happen because of threshold effects, small sample size, or because a mild blur/noise can occasionally suppress irrelevant details.  
However, the overall trend is clear: stronger distortions generally reduce performance.

---

### Main Task Comparison Across Distortion Levels

The following bar plots compare the clean reference with all distortion levels for each task.

#### Canny Robustness

<img width="1484" height="883" alt="image" src="https://github.com/user-attachments/assets/b26cda11-3e14-4d63-916e-87b9f6d0dbc7" />

#### GrabCut Robustness

<img width="1484" height="883" alt="image" src="https://github.com/user-attachments/assets/f2ce1a63-848c-4186-89c9-2ef96fd5bb98" />


#### ResNet50 Robustness

<img width="1484" height="883" alt="image" src="https://github.com/user-attachments/assets/93dd41b5-8d25-4b59-8059-f78cf1b6a254" />


These plots provide a direct comparison between the clean reference and the distorted-image performance.  
The red dashed line marks the clean baseline or clean reference for each task.

---

### Qualitative Results on Distorted Images

The figure below shows qualitative examples of the algorithms applied to distorted images.  
Each row presents a distortion condition and the corresponding outputs: distorted input image, Canny edge map, GrabCut mask, ResNet50 prediction, and ground-truth segmentation mask.

<img width="2893" height="2065" alt="image" src="https://github.com/user-attachments/assets/b312b593-eb2f-4838-bd52-8ef8e41995c6" />


The qualitative examples show that different distortions affect different tasks in different ways.  
Salt & Pepper noise creates many false local structures, Motion Blur removes fine details and object boundaries, and Overexposure reduces contrast and saturates bright regions.

---

### Per-Class Performance Under Distortions

We also analyzed average performance per target object class across the distorted images.

<img width="2384" height="880" alt="image" src="https://github.com/user-attachments/assets/e34a6440-66bb-4d0e-883b-8150efcbc080" />


The per-class analysis shows that some object categories are more robust to distortions than others.  
Classes with clearer shapes and stronger foreground/background separation tend to preserve better performance, while visually complex or smaller objects are more sensitive to degradation.

---

### Robustness Summary

The distortion experiments show that all three methods degrade as image quality decreases, but each method is affected differently:

- **Canny Edge Detection** is the most sensitive to distortions because it depends directly on local intensity gradients.
- **GrabCut Segmentation** is strongly affected when distortions damage object boundaries or foreground/background color distributions.
- **ResNet50 Classification** is more stable under mild distortions, but severe Motion Blur and strong Salt & Pepper noise significantly reduce its multi-label classification performance.

These results motivate the next stage of the project: applying image enhancement and restoration methods before re-evaluating the models.

## 6. Image Enhancement and Restoration

After evaluating the degradation caused by image distortions, we applied restoration and enhancement methods in order to recover part of the lost performance.

For this stage, each distorted image was enhanced using a restoration method matched to the specific distortion type.  
Then, the same three computer vision tasks were evaluated again:

1. **Canny Edge Detection**
2. **GrabCut Foreground Segmentation**
3. **ResNet50 Multi-Label Classification**

The goal was to compare performance before and after restoration using the same metrics as in the distorted-image evaluation.

---

### Restoration Methods

Each distortion type was restored using a different method, because each distortion damages the image in a different way.

| Distortion | Restoration / Enhancement Method | Motivation |
| :--- | :--- | :--- |
| Salt & Pepper Noise | Median Filtering | Removes isolated black and white impulse noise while preserving object boundaries |
| Overexposure | Inverse brightness correction + mild CLAHE | Reduces excessive brightness and restores local contrast |
| Motion Blur | Richardson-Lucy deconvolution + bilateral filtering + mild luminance sharpening | Attempts to recover blurred edges while controlling artifacts |

---

### Visual Restoration Examples

The following figures show examples of clean, distorted, and restored images for each distortion type.

#### Salt & Pepper Noise Restoration

<img width="2235" height="1420" alt="image" src="https://github.com/user-attachments/assets/58fa6192-7d5a-4f19-968e-26724a80bb11" />

#### Overexposure Restoration

<img width="2235" height="1420" alt="image" src="https://github.com/user-attachments/assets/8e9f55b9-23f6-40bf-b789-f85ae6bf6813" />

#### Motion Blur Restoration

<img width="2235" height="1420" alt="image" src="https://github.com/user-attachments/assets/e630a5be-9767-4cfc-8de4-d43a2173c11c" />

The visual examples show that Salt & Pepper noise is strongly reduced by median filtering.  
Overexposure correction restores part of the original brightness and contrast.  
Motion Blur restoration is more challenging, but the Richardson-Lucy based method recovers some edge sharpness while avoiding the severe artifacts caused by more aggressive deconvolution.

---

### Evaluation Metrics on Restored Images

We used task-specific metrics because each task produces a different type of output.

| Task / Model | Metric | Meaning |
| :--- | :--- | :--- |
| Canny Edge Detection | Edge-Map IoU vs Clean Reference | Measures how similar the restored edge map is to the clean edge map |
| GrabCut Segmentation | Segmentation IoU vs Ground Truth | Measures overlap between the predicted foreground mask and the PASCAL VOC segmentation mask |
| ResNet50 Multi-Label Classification | Multi-label F1-score | Measures classification quality when multiple object labels may exist in the same image |
| Image Quality | SNR | Measures how close the distorted/restored image is to the original clean image |

For Canny, the clean edge map is used as the reference because PASCAL VOC does not provide ground-truth edge maps.  
For GrabCut, the predicted foreground mask is compared against the PASCAL VOC segmentation ground truth.  
For ResNet50, we use multi-label F1-score because a PASCAL VOC image may contain more than one object class.

---

### Quantitative Restoration Results

The table below summarizes the average performance before and after restoration.

| Distortion | Distorted SNR | Restored SNR | Distorted Canny Edge-Map IoU | Restored Canny Edge-Map IoU | Distorted GrabCut Segmentation IoU | Restored GrabCut Segmentation IoU | Distorted ResNet50 F1 | Restored ResNet50 F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Salt & Pepper | 10.660 | 19.772 | 0.201 | 0.253 | 0.467 | 0.674 | 0.700 | 0.839 |
| Overexposure | 5.164 | 12.498 | 0.527 | 0.576 | 0.561 | 0.613 | 0.819 | 0.833 |
| Motion Blur | 17.219 | 18.248 | 0.136 | 0.204 | 0.578 | 0.622 | 0.677 | 0.680 |

The restoration methods improved performance for all three distortion types, but the improvement was not uniform across tasks.

Salt & Pepper noise showed the strongest recovery, mainly because median filtering is highly effective for impulse noise.  
Overexposure also improved after inverse brightness correction and local contrast enhancement.  
Motion Blur showed a clear improvement in Canny Edge-Map IoU and GrabCut Segmentation IoU, but only a very small improvement in ResNet50 Multi-Label F1-score. This indicates that motion blur is harder to recover using classical preprocessing methods.

---

### Relative Performance: Distorted vs Enhanced

The following plots compare the relative performance of distorted and enhanced images.  
The dashed red line represents the clean-image reference level. A value closer to 1.0 means that the method performs closer to its clean-image baseline.

#### Canny Edge Stability

<img width="1334" height="882" alt="image" src="https://github.com/user-attachments/assets/397d892e-2659-4ab5-bf00-007bd2c135d6" />

#### GrabCut Segmentation

<img width="1334" height="882" alt="image" src="https://github.com/user-attachments/assets/119fb527-5dcf-4ed9-be34-05a1c1cf30e5" />

#### ResNet50 Multi-Label Classification

<img width="1334" height="882" alt="image" src="https://github.com/user-attachments/assets/55356b22-deef-43e9-9cf2-f6fc62d48532" />

These plots show that restoration improves the low-level and segmentation tasks more clearly than the high-level classification task.  
The strongest improvement appears under Salt & Pepper noise, while Motion Blur remains the most difficult distortion to restore.

---

### SNR Before and After Restoration

The following plot compares image quality before and after restoration using Signal-to-Noise Ratio.

<img width="1484" height="886" alt="image" src="https://github.com/user-attachments/assets/9d66009b-5a63-402a-a59d-0d27e22bcea5" />

SNR improves for all three distortions after restoration.  
The largest improvement occurs for Salt & Pepper noise and Overexposure.  
Motion Blur has a smaller SNR improvement, which is expected because motion blur removes high-frequency details that are difficult to reconstruct without a stronger learned restoration model.

---

### Running the Algorithms on Enhanced Images

The following figures show the outputs of the three vision tasks - Canny Edge Detection, GrabCut Foreground Segmentation, and ResNet50 Multi-Label Classification - on clean, distorted, and enhanced images for all three distortion types.
**Canny Edge Detection: Clean vs Distorted vs Enhanced**
<img width="1914" height="1858" alt="image" src="https://github.com/user-attachments/assets/cd50dbbe-9908-4dfc-8198-e981c3194525" />
**GrabCut Foreground Segmentation: Clean vs Distorted vs Enhanced**
<img width="1914" height="1858" alt="image" src="https://github.com/user-attachments/assets/9adfb78f-0274-4be7-a2fe-d54294201068" />
**ResNet50 Multi-Label Classification: Clean vs Distorted vs Enhanced**
<img width="2063" height="2004" alt="image" src="https://github.com/user-attachments/assets/e0803d7a-2fd2-49d3-9aca-b8bc50a2d2fa" />


This comparison demonstrates how image restoration affects both visual quality and downstream task outputs.

For **Canny Edge Detection**, restoration reduces false edges caused by Salt & Pepper noise and partially restores edge structure after Motion Blur.  
For **GrabCut Segmentation**, enhanced images improve foreground/background separation, especially when object boundaries become clearer after restoration.  
For **ResNet50 Classification**, restoration improves predictions mainly when the object appearance is preserved, while Motion Blur produces only limited recovery.

Overall, Salt & Pepper and Overexposure benefit most from enhancement, while Motion Blur remains the hardest distortion to recover.

---

### Per-Class Performance on Enhanced Images

We also measured enhanced-image performance per target object class.  
This analysis helps identify which classes benefit more from restoration and which classes remain sensitive to distortions.

#### Enhanced Canny Edge-Map IoU per Class

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/0b120580-8af5-4716-86a4-ad6176b6bd35" />

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/79a86a72-cad7-432f-90c2-594864744ea7" />

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/80992c88-dc3f-4c02-89df-6d46fbbf87ae" />

#### Enhanced GrabCut Segmentation IoU per Class

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/ab817473-4c72-4aed-9f29-812b6e31e855" />

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/a562ed5c-867d-4e6c-a386-0af9897d0f3a" />

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/787baef7-954e-4ab8-b37d-b572d773e869" />

#### Enhanced ResNet50 Multi-Label F1-Score per Class

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/838d6281-0078-4a10-8245-5474e12d75d0" />

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/6d3ca5e9-21db-45e6-ae29-016caa66b735" />

<img width="1784" height="882" alt="image" src="https://github.com/user-attachments/assets/192be864-778e-447c-a4b6-66ba332e558d" />

The per-class results show that restoration does not affect all object categories equally.  
Classes with clearer shapes and stronger foreground-background separation tend to benefit more from restoration, especially for GrabCut.  
Smaller objects or visually complex categories remain more sensitive to distortion.

---

### Per-Class SNR on Restored Images

In addition to task performance, we also analyzed the restored-image SNR per class.

<details>
<summary>Restored SNR per class</summary>

#### Salt & Pepper

<img width="2084" height="884" alt="image" src="https://github.com/user-attachments/assets/8ed5be1b-1eff-45bc-a3f1-0fc0875c10fd" />

#### Overexposure

<img width="2084" height="884" alt="image" src="https://github.com/user-attachments/assets/618ea237-22bb-4057-9c96-b944f72d0aca" />

#### Motion Blur

<img width="2084" height="884" alt="image" src="https://github.com/user-attachments/assets/28267f74-4ef7-4e4d-b156-2c14b179b4b9" />

</details>

---

### Restoration Summary

The restoration stage shows that preprocessing can recover part of the lost performance, but the amount of recovery depends strongly on the distortion type and the downstream task.

- **Salt & Pepper Noise:** Median filtering produced the strongest improvement across SNR, Canny, GrabCut, and ResNet50.
- **Overexposure:** Inverse brightness correction and mild CLAHE improved both image quality and task performance.
- **Motion Blur:** Richardson-Lucy based restoration improved edge-map similarity and segmentation IoU, but classification improvement was limited.

Overall, the results show that restoration is useful, but it must be matched carefully to the distortion type.  
The improvement is strongest when the distortion can be reduced without introducing new artifacts.

---

---

### Final Summary: Traditional Computer Vision Tasks

The following figures provide a direct visual summary of the complete evaluation pipeline for the two traditional computer vision tasks: performance on clean images, degradation under distortions, and recovery after restoration.

#### Task 1: Canny Edge-Map Stability

For Canny Edge Detection, the clean edge map is used as the reference. Therefore, the clean Edge-Map IoU equals **1.0**.

<img width="989" height="602" alt="Canny clean distorted and restored comparison" src="https://github.com/user-attachments/assets/a470a402-7f78-4905-a25e-1f7ef554045f" />

#### Task 2: GrabCut Segmentation

GrabCut performance is measured using Segmentation IoU against the PASCAL VOC ground-truth mask.

<img width="989" height="590" alt="GrabCut clean distorted and restored comparison" src="https://github.com/user-attachments/assets/240752bd-5568-4f91-bb5d-186683114877" />

Across all three distortion types, restoration improved both traditional computer vision tasks. GrabCut showed the strongest recovery under Salt & Pepper Noise and slightly exceeded its clean baseline after restoration. Canny also improved in every condition, although its restored edge maps remained below the clean reference, indicating that lost edge structure is more difficult to reconstruct completely.

## 7. Distortion-Aware Fine-Tuning

After evaluating image restoration as a preprocessing approach, we investigated a second method for improving deep-learning robustness: fine-tuning the ResNet50 classifier directly on distorted images.

The objective of this experiment was to determine whether exposure to distorted images during training improves multi-label classification performance under image degradation.

---

### Controlled Experimental Design

To ensure a fair comparison, we trained two ResNet50 models under identical experimental conditions.

Both models were initialized from the same ResNet50 checkpoint previously trained on clean PASCAL VOC images.

The two evaluated models were:

1. **Clean-Control ResNet50** - received additional training using clean images.
2. **Distortion-Aware ResNet50** - received additional training using distorted versions of the same original images.

Both models used:

- The same original training-image indices
- The same number of training and validation images
- The same batch order
- The same number of epochs
- The same trainable layers
- The same optimizer and learning rates
- The same validation images
- The same checkpoint-selection criterion
- The same evaluation threshold and metrics

Therefore, the only intentional difference between the two models was the training input: clean images versus distorted images.

---

### Fine-Tuning Configuration

| Parameter | Setting |
| :--- | :--- |
| Base Architecture | ResNet50 |
| Initial Checkpoint | Clean-trained PASCAL VOC model |
| Training Images per Model | 4,500 |
| Validation Images | 800 |
| Additional Training Epochs | 8 |
| Trainable Layers | Layer 3, Layer 4, and the final classification layer |
| Loss Function | Binary Cross-Entropy with Logits |
| Optimizer | AdamW |
| Batch Size | 32 |
| Evaluation Threshold | 0.30 |
| Evaluation Images per Condition | 400 |

Because PASCAL VOC is a multi-label dataset, each image retained its original 20-dimensional label vector. The distortions modify the visual appearance of the image but do not change which object classes are present.

During distortion-aware training, the input images were randomly modified using Salt & Pepper Noise, Overexposure, or Motion Blur. All three distortions were represented at six severity levels.

Higher severity levels were sampled more frequently so that the model would receive sufficient exposure to challenging degradation.

| Severity Level | Relative Sampling Frequency |
| :--- | ---: |
| L1 | 1 |
| L2 | 1 |
| L3 | 2 |
| L4 | 3 |
| L5 | 4 |
| L6 | 5 |

---

### Training and Validation Behavior

Both models were evaluated during training on clean validation images and fixed distorted versions of the same validation images.

The selected checkpoint was the epoch with the lowest combined validation loss across the two validation sets.


<img width="2234" height="740" alt="part4_controlled_finetuning_loss_curves" src="https://github.com/user-attachments/assets/364b732a-4e17-4f28-9748-0e704bdc49de" />

The Clean-Control model achieved its lowest combined validation loss at epoch 2. Although its training loss continued to decrease, the validation loss began to increase, indicating overfitting after the second epoch.

The Distortion-Aware model continued improving until epoch 5 and remained stable during the following epochs.

| Model | Best Epoch | Best Combined Validation Loss |
| :--- | ---: | ---: |
| Clean-Control ResNet50 | 2 | 0.1022 |
| Distortion-Aware ResNet50 | 5 | **0.0811** |

---

### Overall Fine-Tuning Results

The models were evaluated on the same 400 PASCAL VOC validation images under 19 conditions:

- Clean images
- Six Salt & Pepper severity levels
- Six Overexposure severity levels
- Six Motion Blur severity levels

The primary metric was the sample-averaged multi-label F1-score.

| Model | Mean F1 on Distorted Conditions |
| :--- | ---: |
| Clean-Control ResNet50 | 0.7195 |
| Distortion-Aware ResNet50 | **0.8031** |
| Absolute Improvement | **+0.0837** |

The distortion-aware model achieved an absolute F1 improvement of **0.0837**, corresponding to a relative improvement of approximately **11.6%**.

Because both models received the same amount of additional training, this controlled comparison shows that the improvement resulted from exposure to distorted training inputs rather than from additional training alone.

---

### Average Performance per Distortion Type


<img width="1484" height="877" alt="part4_controlled_finetuning_per_distortion_type" src="https://github.com/user-attachments/assets/a64505ee-1e77-48b0-95a8-c9ad4901db9e" />

| Distortion | Clean-Control F1 | Distortion-Aware F1 | Improvement |
| :--- | ---: | ---: | ---: |
| Salt & Pepper Noise | 0.7083 | **0.8008** | **+0.0925** |
| Overexposure | 0.7825 | **0.8213** | **+0.0388** |
| Motion Blur | 0.6675 | **0.7873** | **+0.1197** |

The largest average improvement was observed for **Motion Blur**, followed by **Salt & Pepper Noise**.

The improvement for Overexposure was smaller because the Clean-Control model was already relatively stable under mild and medium overexposure.

---

### Performance Across Distortion Severity Levels

The following graph compares the models separately at every distortion severity level.


<img width="2534" height="1028" alt="part4_controlled_finetuning_all_conditions" src="https://github.com/user-attachments/assets/7f6aaa02-e19d-4d27-beab-89bba281725a" />


At mild distortion levels, the performance difference between the models is relatively small. As the severity increases, the advantage of distortion-aware fine-tuning becomes substantially larger.

At the strongest evaluated levels:

| Condition | Clean-Control F1 | Distortion-Aware F1 | Improvement |
| :--- | ---: | ---: | ---: |
| Salt & Pepper L6 | 0.4698 | **0.7395** | **+0.2697** |
| Overexposure L6 | 0.6074 | **0.7502** | **+0.1428** |
| Motion Blur L6 | 0.3442 | **0.6441** | **+0.2999** |

The largest individual improvement was obtained for Motion Blur L6, where the F1-score increased by approximately 0.30.

---

### Performance per SNR

The following plots show classification performance as a function of mean SNR for each distortion type.

<img width="2684" height="812" alt="part4_controlled_finetuning_per_snr" src="https://github.com/user-attachments/assets/828db423-dced-47e9-a4b0-ca61e678767f" />


Lower SNR generally represents stronger image degradation.

For all three distortion types, the Distortion-Aware model preserves higher F1-scores as SNR decreases. The difference is especially pronounced for severe Salt & Pepper Noise and Motion Blur.

These results demonstrate that distortion-aware fine-tuning reduces the rate at which classification performance degrades as image quality becomes worse.

---

### Per-Class Performance

We also compared the average F1-score of the two models for each of the 20 PASCAL VOC classes across all distorted conditions.

<img width="2384" height="1027" alt="part4_controlled_per_class_comparison" src="https://github.com/user-attachments/assets/f2fd6c20-4508-49b4-a265-f6d6dadd61c4" />


The mean per-class F1-score increased from **0.695** for the Clean-Control model to **0.786** for the Distortion-Aware model.

All 20 PASCAL VOC classes showed a positive average improvement.

The largest improvements were observed for:

| Class | F1 Improvement |
| :--- | ---: |
| Sofa | +0.233 |
| Dining Table | +0.188 |
| Bicycle | +0.129 |
| Chair | +0.128 |
| Sheep | +0.120 |

The results show that the overall improvement is distributed across multiple object categories rather than being caused by only a small number of classes.

---

### Clean-Image Performance

Distortion-aware fine-tuning did not reduce performance on clean images.

| Model | F1 on Clean Images |
| :--- | ---: |
| Clean-Control ResNet50 | 0.8511 |
| Distortion-Aware ResNet50 | **0.8565** |

The Distortion-Aware model slightly improved clean-image performance while achieving a much larger improvement under severe distortions.

---

### Fine-Tuning Summary

The controlled fine-tuning experiment demonstrates that direct training on distorted images significantly improves ResNet50 robustness.

The main findings are:

- Mean F1 across all distorted conditions increased from **0.7195** to **0.8031**.
- The absolute improvement was **0.0837**, or approximately **11.6% relative improvement**.
- Motion Blur produced the largest average improvement.
- The advantage of fine-tuning increased as distortion severity increased and SNR decreased.
- Mean per-class F1 increased from **0.695** to **0.786**.
- All 20 PASCAL VOC classes showed a positive improvement.
- Performance on clean images was preserved.

Overall, distortion-aware fine-tuning was substantially more effective than additional clean-image training, particularly under severe image degradation.

### Qualitative Prediction Comparison

The following example compares the predictions of the Clean-Control and
Distortion-Aware models on the same image under clean and distorted
conditions.

<img width="2859" height="1226" alt="part4_controlled_qualitative_comparison" src="https://github.com/user-attachments/assets/aee0eeb9-eaa4-483f-8873-969433398d11" />

The ground-truth class for this image is **car**.

The Distortion-Aware model achieved a higher image-level F1-score on the
clean image, Salt & Pepper Noise, and Motion Blur conditions. Under Salt &
Pepper Noise and Motion Blur, it correctly preserved the car prediction
while producing fewer incorrect additional labels.

For Overexposure, both models achieved the same F1-score. This example
illustrates that the benefit of distortion-aware fine-tuning depends on the
distortion type, while showing clear improvement under noise and blur.

---

### Task 3 Summary: ResNet50 Multi-Label Classification

The following figure summarizes the complete evaluation pipeline for the deep-learning task. It compares the clean-image baseline with performance on distorted images, restored images, and distorted images evaluated using the distortion-aware fine-tuned model.

<img width="630" height="337" alt="task3" src="https://github.com/user-attachments/assets/20b35b50-a2fc-4ce1-a064-dccbabecc3a9" />

The results show that the most effective improvement method depends on the distortion type.

- For **Salt & Pepper Noise**, image restoration produced the best result and nearly recovered the clean-image baseline.
- For **Overexposure**, distortion-aware fine-tuning achieved the highest F1-score and slightly exceeded the clean baseline.
- For **Motion Blur**, fine-tuning produced a substantial improvement and fully recovered the lost classification performance.

Overall, restoration was particularly effective for impulse noise, while distortion-aware fine-tuning was more effective for distortions that were difficult to correct through preprocessing, especially Motion Blur.

## 8. Overall Project Summary and Conclusions

This project evaluated the robustness of three computer vision methods under Salt & Pepper Noise, Overexposure, and Motion Blur.

The complete evaluation included:

- Performance on clean images
- Performance degradation under multiple distortion severity levels
- Image restoration and enhancement
- Distortion-aware fine-tuning for the deep-learning model

### Summary by Task

| Task | Main Effect of Distortions | Improvement Method | Main Finding |
| :--- | :--- | :--- | :--- |
| Canny Edge Detection | Noise introduced false edges, while Motion Blur removed and shifted edge structure | Distortion-specific image restoration | Restoration improved Edge-Map IoU under all distortions, but did not fully recover the clean reference |
| GrabCut Segmentation | Distortions damaged object boundaries and foreground-background color separation | Distortion-specific image restoration | Restoration improved Segmentation IoU under all distortions, with the strongest recovery under Salt & Pepper Noise |
| ResNet50 Multi-Label Classification | Severe Motion Blur and Salt & Pepper Noise caused the largest classification degradation | Distortion-aware fine-tuning | Fine-tuning substantially improved robustness, especially at severe distortion levels, while preserving clean-image performance |

### Main Conclusions

The experiments demonstrate that image distortions affect computer vision tasks differently.

Canny was the most sensitive method because it depends directly on local intensity gradients. Although restoration improved its results, lost or shifted edge information could not be fully reconstructed.

GrabCut benefited strongly from restoration, especially after median filtering of Salt & Pepper Noise. In this condition, the restored segmentation score slightly exceeded the clean baseline, likely because smoothing produced more stable foreground and background regions.

For ResNet50, distortion-aware fine-tuning was more effective than classical image restoration under severe degradation. The largest gains were observed for Motion Blur and strong Salt & Pepper Noise.

Overall, the results show that no single improvement method is optimal for every task. Classical restoration is effective when the distortion can be reduced directly, while distortion-aware fine-tuning provides stronger robustness for high-level deep-learning models.
