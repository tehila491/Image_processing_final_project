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

<img width="1784" height="730" alt="image" src="https://github.com/user-attachments/assets/4e15a46c-e6ab-49fc-b073-60d4ec4bde82" />


#### Canny — Edge Density per Class

Canny edge detection does not output semantic labels or object masks.  
Therefore, for this low-level task we report the average edge density per class, which measures the proportion of detected edge pixels in the clean images.

<img width="1484" height="730" alt="image" src="https://github.com/user-attachments/assets/7ad12be1-c86d-48fa-9096-cf0991d89219" />


---

### Clean Baseline Summary

Overall, the clean-image baseline shows that GrabCut achieves reasonable foreground segmentation performance when initialized with ground-truth bounding boxes, while ResNet50 provides a meaningful multi-label classification baseline on clean images.  
These clean-image results are used as the reference point for the next stage, where we evaluate performance degradation under Salt & Pepper noise, overexposure, and motion blur.


## 5. Evaluating Robustness (Applied Distortions)
We applied our 3 distortions to the dataset and measured the degradation in performance. 

### Visualizing the Distortions
> 🖼️ **[INSERT IMAGE HERE: Side-by-side grid showing Original vs. Noise vs. Rain vs. Low-light]**

### Performance Degradation Results
| Model | Metric | Clean (Baseline) | Speckle Noise | Rain | Low-Light |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Canny | Edge Density | [Score] | [Score] | [Score] | [Score] |
| GrabCut | IoU | [Score] | [Score] | [Score] | [Score] |
| ResNet50 | Accuracy | [Score] | [Score] | [Score] | [Score] |

> 📉 **[INSERT IMAGE HERE: A line graph or bar chart showing the performance drop for ResNet/GrabCut across the distortions]**

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
