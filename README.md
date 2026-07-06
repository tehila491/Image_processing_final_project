# Evaluating Robustness of Computer Vision Models 👁️

## 1. Project Overview
This project evaluates the robustness of various computer vision algorithms and deep learning models under different image distortions. We analyze how classical and modern methods degrade when facing noise, bad lighting, and weather conditions, and attempt to recover performance through enhancements and fine-tuning.

**Team Members:** Tehila Segal, Hadar Semama

## 2. Project Decisions & Setup
Based on the project requirements, we selected the following dataset, tasks, and distortions:

| Category | Selection | Description |
| :--- | :--- | :--- |
| **Dataset** | [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) | High-quality dataset for object detection and segmentation. |
| **Task 1 (Low-Level)** | Edge Detection | **Model:** Canny Edge Detector |
| **Task 2 (Mid-Level)** | Image Segmentation | **Model:** GrabCut Algorithm |
| **Task 3 (High-Level)**| Image Classification | **Model:** ResNet50 (Pre-trained) |
| **Distortions** | Salt & Pepper Noise, High Exposure, Motion Blur | Applied at varying severities to test robustness. |

## 3. Exploratory Data Analysis (EDA)
Here is a sample of our clean data with ground truth annotations:
<img width="1189" height="946" alt="image" src="https://github.com/user-attachments/assets/ca2f3a3a-531c-44ee-b529-6309531b4400" />

 
## 4. Baseline Performance (Clean Images)

Before applying distortions, we evaluated all three selected tasks on clean PASCAL VOC 2012 images.  
This stage establishes the clean baseline performance, which will later be used as the reference point for measuring degradation under distortions and recovery after enhancement or fine-tuning.

| Task / Model | Evaluation Metric | Baseline Score |
| :--- | :--- | :--- |
| Canny Edge Detection | Mean Edge Density | 0.0968 |
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

<img width="1184" height="731" alt="image" src="https://github.com/user-attachments/assets/a669a7a9-7808-4be1-a248-6ec6ab4d4637" />


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

#### GrabCut — IoU per Class

GrabCut was evaluated by comparing the predicted foreground mask with the PASCAL VOC segmentation ground truth.  
The method performs better on objects with clear foreground-background separation and usually performs worse on complex scenes, thin structures, or objects with ambiguous boundaries.

<img width="1484" height="730" alt="image" src="https://github.com/user-attachments/assets/a3ab4917-8424-45da-ae2b-b9cc6e162029" />


#### ResNet50 — Multi-Label IoU / Jaccard per Class

The ResNet50 classifier was evaluated using multi-label IoU / Jaccard score for each of the 20 PASCAL VOC classes.  
This per-class analysis shows which categories are recognized more reliably on clean images and which classes are more challenging.

<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/c1f23fe8-a1bc-41b8-843f-3af5a16a9191" />


#### Canny — Edge Density per Class

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

1. **Salt & Pepper Noise** — random black and white pixels are inserted into the image.
2. **Overexposure** — brightness and intensity are increased, causing saturated and washed-out regions.
3. **Motion Blur** — directional blur is applied to simulate camera or object motion.

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

#### Canny — Edge-Map IoU vs Clean Reference

<img width="1334" height="882" alt="image" src="https://github.com/user-attachments/assets/08bcafb0-6a3c-454a-9c36-146e5bd1a5ca" />


Canny is highly sensitive to distortions.  
Salt & Pepper noise introduces many false edges, while Motion Blur removes sharp edges, causing a strong decrease in edge-map similarity.

#### GrabCut — Segmentation IoU vs Ground Truth

<img width="1334" height="882" alt="image" src="https://github.com/user-attachments/assets/d1427778-24f7-4208-b9db-824483c24192" />

GrabCut performance decreases when image distortions damage foreground/background color separation or object boundaries.  
The degradation is strongest under severe Salt & Pepper noise and severe Overexposure.

#### ResNet50 — Multi-Label F1-Score

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

## 6. Enhancements & Improvements
To recover the lost performance, we applied pre-processing enhancements (e.g., denoising, brightness adjustments) and fine-tuned our deep learning model.

### Pre-processing Enhancements (Canny & GrabCut)
> 🖼️ **[INSERT IMAGE HERE: Visual of Distorted Image -> Enhanced Image -> Output]**

### Fine-Tuning (ResNet50)
We fine-tuned ResNet50 on the distorted data to improve its robustness.
* **Weights:** `[Link to checkpoint/model weights if applicable]`

### Final Comparison (Improvement)
| Model | Distorted Score | Enhanced / Fine-Tuned Score | Recovery (%) |
| :--- | :--- | :--- | :--- |
| GrabCut (Rain) | [Score] | [Score] | [X%] |
| ResNet50 (Noise) | [Score] | [Score] | [X%] |

## 7. How to Run This Project
1. Clone the repository: `git clone [Your Repo URL]`
2. Install requirements: `pip install -r requirements.txt`
3. Download the data: Run `data_download_and_eda.py`
4. Run baseline evaluation: `python baseline_evaluation.py`
5. Apply distortions and evaluate: `python evaluate_distortions.py`
