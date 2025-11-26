# 🛍️ Mannequin Detection Using YOLOv8 & Computer Vision

A high-accuracy **Mannequin Detection System** built using **YOLOv8**, **OpenCV**, and **Deep Learning**.  
This project focuses on detecting mannequins in retail-store images or video streams, helping automate  
store analytics, visual merchandising audits, and product-placement evaluation.

---

## 🚀 Overview

Retail stores often use mannequins to showcase products and improve customer engagement.  
Manual monitoring is inefficient and inconsistent.  
This project provides an **automated mannequin detection pipeline** capable of:

- Detecting mannequins in images and videos  
- Locating mannequins with bounding boxes  
- Operating in real time (depending on hardware)  
- Supporting integration with store analytics systems  

The system is trained and evaluated using a custom mannequin dataset and optimized for both performance and accuracy.

---

## 🎯 Project Goals

- Build a **robust mannequin-detection model** using YOLOv8  
- Improve accuracy using data augmentation and hyperparameter tuning  
- Support **real-time inference** with camera feeds  
- Provide clean inference functions for production usage  
- Enable easy testing on images, videos, and live streams  

---

## 🧠 Key Features

- ✔️ **YOLOv8-based mannequin detection**  
- ✔️ Works with **images, videos, CCTV footage**  
- ✔️ **High-speed inference**  
- ✔️ Support for **GPU and CPU**  
- ✔️ **Confidence-based filtering**  
- ✔️ Exportable model (`.pt`) for deployment  
- ✔️ Simple Python API for integration  

---

## 🛠️ Technologies & Tools Used

### **Core Frameworks**
- **Python**
- **Ultralytics YOLOv8**
- **OpenCV**
- **Numpy**

### **Additional Tools**
- Data augmentation  
- Custom dataset annotation (LabelImg / CVAT)  
- Training logs & evaluation metrics  

---

## 📊 Model Training

The mannequin detection model was trained using:

- **YOLOv8s** (small, fast, reliable)  
- Custom mannequin dataset with labeled bounding boxes  
- 100–300 epochs depending on hardware  
- Hyperparameters tuned for:  
  - learning rate  
  - confidence threshold  
  - IoU threshold  
  - augmentations  

The final model achieved:

- **High precision** on mannequin objects  
- **Strong generalization** across different lighting conditions  
- **Stable detection at multiple distances**  

(You may update exact metrics after training.)

---

## 🧪 Evaluation Metrics

The model was evaluated using:

- **mAP@50**
- **mAP@50–95**
- **Precision / Recall**
- **Confusion matrix**
- **Per-class performance**

Sample evaluation results are provided in "mannequin_test.ipynb" .

---

## 📷 Inference Capabilities

The project provides clean inference logic to run mannequin detection on:

- 📌 **Images**
- 🎥 **Videos**
- 📡 **Live camera streams**
- 🖼️ **Batch processing folders**

Each inference run outputs:

- Bounding box around detected mannequins  
- Class name and confidence score  
- Option to save results  

---

## 💡 Example Use Cases

- Retail visual merchandising
- Store analytics dashboards  
- Product placement monitoring  
- Automated inventory documentation  
- Crowd-free mannequin monitoring  
- Fashion photoshoot automation  

---

## 📥 Dataset Information

The model was trained on a **custom mannequin dataset** containing:

- Variety of store environments  
- Different mannequin sizes, poses, and clothing  
- Both indoor and outdoor setups  

Dataset includes annotations in **YOLO format**.

*(Upload link or dataset instructions can be added here once available.)*

---

## ⚙️ Training & Inference Summary

- The model was trained using YOLOv8 CLI/Python API  
- Hardware used: CPU or GPU (NVIDIA recommended)  
- Inference pipeline built with OpenCV  
- Outputs saved with bounding boxes + labels  

---

## 🏁 Results Summary

Key achievements of this project:

- High detection accuracy on custom mannequin dataset  
- Smooth inference on real-time video feeds  
- Reliable performance across different store environments  
- Modular code design for easy integration  

---

## 🧩 Future Improvements

- Add mannequin pose estimation  
- Improve detection in low-light environments  
- Add mannequin segmentation (instance segmentation)  
- Deploy model using Flask/FastAPI  
- Create a dashboard for store analytics  
- Convert model to TensorRT for faster inference  

---

## 📬 Contact

For queries, suggestions, or collaboration:

- **Email:** hemhalathavr@gmail.com  
- **LinkedIn:** https://www.linkedin.com/in/v-r-hemhalatha-804634326/

---

### ⭐ If you like this project, consider giving it a star!
