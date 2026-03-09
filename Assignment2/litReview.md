# **Technical Architectures and Optimization Protocols for Real-Time Deep Learning-Based Facial Detection and Recognition Systems**

The progression of facial recognition technology from controlled biometric environments to unconstrained real-time applications represents one of the most significant shifts in computer vision. Achieving high-accuracy detection and recognition in a live camera feed requires a synergistic integration of multi-task localization, dense facial alignment, and discriminative embedding generation. Current research has moved beyond simple classification toward the development of robust, pose-invariant, and quality-aware models that can be optimized for high-performance GPU hardware. The following analysis provides an exhaustive review of state-of-the-art architectures, loss functions, and data engineering techniques essential for implementing a modern face recognition system.

## **Evolution of Facial Detection Paradigms**

Facial detection serves as the fundamental preprocessing step, wherein the system must identify the presence, location, and scale of human faces within a high-resolution video stream. Traditional approaches have transitioned from rigid cascaded structures to single-stage dense localization frameworks and eventually to computation-aware architectures designed for real-time efficiency.

### **Multi-Task Cascaded Convolutional Neural Networks**

The Multi-Task Cascaded Convolutional Neural Network (MTCNN) represents a cornerstone in facial detection, utilizing a three-staged sequential architecture to detect faces and five facial landmarks simultaneously.1 The mechanism begins with the Proposal Network (P-Net), a fully convolutional network that generates candidate windows and their bounding box regression vectors. Following this, the Refinement Network (R-Net) filters the candidates, utilizing a more complex structure to reject a large number of false positives. Finally, the Output Network (O-Net) provides the final bounding box and landmark locations, ensuring high precision.1

While MTCNN is praised for its robustness against variations in scale and lighting, it encounters significant performance degradation when faces are subject to large-pose variations or partial occlusions.1 Benchmarking on the WIDER FACE dataset reveals that MTCNN achieves an Average Precision (AP) of approximately 85.10% on the "Easy" subset but drops precipitously to 60.84% on the "Hard" subset, which contains smaller and more occluded faces.4 This limitation stems from its multi-stage proposal mechanism, which can miss initial candidates under extreme conditions before they reach the refinement stages.

### **Single-Stage Dense Face Localization: RetinaFace**

The introduction of RetinaFace marked a milestone in single-stage detection, establishing a framework that performs face localization and five-point landmark prediction in a single forward pass.1 RetinaFace utilizes a ResNet-50 backbone combined with a Feature Pyramid Network (FPN) to extract multi-scale feature maps. These maps are further processed by a Context Module (CM), which captures long-range spatial information to improve the model's sensitivity to facial characteristics.1

RetinaFace's primary innovation lies in its multi-task loss function, which combines bounding box regression, classification, and landmark localization with extra supervisory signals. In some versions, dense 3D facial information is predicted through a self-supervised branch, providing the model with a richer geometric prior of the human face.5

| Model Variant | Easy AP | Medium AP | Hard AP | Latency (CPU) |
| :---- | :---- | :---- | :---- | :---- |
| RetinaFace (ResNet-50) | 94.76 | 93.22 | 84.92 | \~20-30ms 4 |
| RetinaFace (MobileNet-V1) | 89.77 | 86.98 | 74.75 | \<10ms 4 |
| R-RetinaFace (Refined) | 98.02 | 92.20 | 83.90 | N/A 1 |

Despite its accuracy, RetinaFace has demonstrated limitations in localizing landmarks on profile faces, leading to the development of Refined-RetinaFace (R-RetinaFace). This refined model incorporates a post-optimization module that ensures predicted landmarks remain within the output bounding boxes, thereby improving performance on the SDUMLA-HMT and CASIA-3D-FaceV1 databases.1

### **Optimized Real-Time Detection: SCRFD and YOLOv8-face**

As facial detection moves toward edge and high-speed deployments, models like SCRFD (Sample and Computation Redistribution for Face Detection) have emerged to optimize the efficiency-accuracy trade-off. SCRFD departs from anchor-based methods in favor of an anchor-free design, which reduces hyperparameter sensitivity and simplifies the training pipeline.5

The SCRFD framework utilizes two core strategies: Sample Redistribution (SR) and Computation Redistribution (CR). SR increases training samples for small faces—typically the most challenging cases—by expanding the scale range of training crops from ![][image1] to ![][image2].6 CR employs neural architecture search to reallocate computational resources across the backbone, neck, and head of the network, ensuring that FLOPS are focused where they contribute most to precision.6

| Model | AP (Hard) | GFLOPS | Parameters | GPU Speed (VGA) |
| :---- | :---- | :---- | :---- | :---- |
| SCRFD-34GF | 85.29 | 34.1 | 9.8M | 3x faster than TinaFace 7 |
| YOLOv8-face | 81.34 | 8.0 | 3.0M | Real-time 4 |
| GCS-YOLOv8 | 81.20 | 6.8 | 1.1M | Ultra-lightweight 7 |

YOLOv8-face represents another high-speed alternative, built upon the popular Ultralytics YOLO architecture. It directly predicts object classes and bounding boxes in a single pass, bypassing region proposals.9 While YOLO-based models were traditionally viewed as less accurate for small-object detection, recent optimizations—such as the HGStem module and shared convolution heads—have allowed variants like GCS-YOLOv8 to exceed 81% AP on the WiderFace Hard set while maintaining an ultra-compact parameter count.7

## **Face Alignment and 3D Modeling**

Once a face is detected, it must be aligned to a canonical pose to ensure the recognition model receives consistent input. Traditional 2D alignment relies on five fiducial landmarks (eyes, nose, and mouth corners), but this approach fails under large poses where parts of the face are self-occluded.10

### **The Move to 3D Dense Face Alignment**

3D Dense Face Alignment (3DDFA) addresses the limitations of 2D methods by fitting a 3D Morphable Model (3DMM) to the image via a Convolutional Neural Network (CNN).11 By estimating the 3D geometry of the face, the system can localize invisible landmarks and perform pose normalization, effectively "rotating" a profile face back to a frontal view.10

The 3DDFAv2 framework improves upon the original by utilizing a lightweight MobileNet backbone and a meta-joint optimization strategy. This allows the model to dynamically regress 3DMM parameters, enhancing both speed and stability.13 3DDFAv2 can regress over 30,000 vertices in approximately 1ms, enabling real-time dense alignment even in video streams where subjects move freely.13

### **PRNet and UV Map Regression**

The Position Map Regression Network (PRNet) provides an end-to-end method to jointly predict dense alignment and reconstruct 3D face shape. Unlike 3DDFA, which regresses model parameters, PRNet predicts a UV position map that records the 3D coordinates of all facial vertices.15 This representation is straightforward for the network to learn and allows for the recovery of 3D structure without intermediate warping or template matching.15

Evaluation on the AFLW2000-3D and Florence datasets shows that PRNet achieves a 25% relative improvement in dense face alignment over previous state-of-the-art methods.15 However, later comparisons indicate that 3DDFAv2 with a ResNet-22 or MobileNet backbone may offer smoother results, as PRNet can occasionally produce jagged mesh artifacts known as checkerboard artifacts.16

## **Face Recognition Backbones: CNNs vs. Transformers**

The choice of feature extractor (backbone) is critical for generating compact and discriminative embeddings. While Residual Networks (ResNet) have been the industry standard, the emergence of Vision Transformers (ViT) has introduced a new paradigm of global feature extraction.

### **Residual Networks and the Skip Connection Advantage**

ResNets maintain dominance in face recognition due to their ability to train very deep networks without degradation, thanks to shortcut connections that allow gradients to bypass layers.18 Backbones such as ResNet-50 and ResNet-100 are frequently paired with advanced margin losses to achieve state-of-the-art results on the LFW and IJB-C benchmarks.19

ResNet-100, in particular, has been utilized extensively in the cleaning and evaluation of massive datasets like WebFace42M, where it consistently provides a stable learning signal for millions of identities.20 Its hierarchical nature effectively captures local spatial features, making it highly data-efficient compared to Transformers.18

### **Vision Transformers and Global Context**

Vision Transformers (ViT) depart from convolutions by splitting images into flattened patches and processing them using a multi-head self-attention mechanism.22 This allows the model to capture relationships between disparate regions of the face (e.g., the distance between the eyes and the chin) more effectively than the local receptive fields of a CNN.23

Experimental results on Open-Set Recognition (OSR) show that ViTs excel at identifying unknown identities, achieving a mean F1 score of 90.50% compared to ResNet50's 85.96%.23 However, ViTs are notoriously data-hungry and typically require pre-training on enormous datasets like ImageNet-21k or JFT-300M to outperform CNNs on recognition tasks.18

### **Hybrid Models and Mobile Efficiency**

For real-time deployment, hybrid models like FaceLiVT combine the strengths of CNNs and Transformers. FaceLiVT utilizes reparameterized convolutional blocks for local feature extraction and Multi-Head Linear Attention (MHLA) to reduce computational complexity.24

| Model Architecture | Parameters | LFW Accuracy | IJB-C TAR @ 1e-4 | Latency (iPhone 15 Pro) |
| :---- | :---- | :---- | :---- | :---- |
| MobileFaceNet | \<1M | 99.70% | 88.31% | 0.52ms 24 |
| FaceLiVT-M | 4.5M | 99.80% | 95.00% | 0.71ms 25 |
| ViT-S | 22M | 99.80% | 96.70% | 15.1ms 25 |
| ResNet-100 | \~65M | 99.83% | 96.03% | High 20 |

FaceLiVT achieves a balance between the global receptive field of a ViT and the low latency required for mobile or edge applications, performing 8.6x faster than other hybrid models like EdgeFace while maintaining competitive accuracy.24

## **Angular Margin Loss Functions and Embedding Optimization**

The discriminative power of facial embeddings is determined by the loss function used during training. Margin-based softmax losses have revolutionized the field by enforcing a geometric separation between identity classes in the angular space.

### **ArcFace: Additive Angular Margin**

ArcFace (Additive Angular Margin Loss) introduced a constant angular margin ![][image3] into the standard softmax loss function to enhance inter-class separability and intra-class compactness.6 The optimization target is defined by:

![][image4]  
where ![][image5] is the angle between the embedding and the ground truth class weight vector.6 By adding the margin ![][image3] directly to the angle, ArcFace ensures a clear boundary between identities, which is essential for generalizing to unseen faces.6 ArcFace remains the most influential baseline for modern facial recognition, achieving 98.36% on the MegaFace benchmark.6

### **MagFace: Quality-Aware Feature Norms**

MagFace builds upon the angular margin concept by recognizing that not all face images contain the same amount of identifiable information. MagFace uses the feature norm ![][image6] as a proxy for image quality, adjusting the angular margin based on this norm.4 High-quality images (high norm) are pushed toward class centers with a larger margin, while low-quality images (low norm) are given smaller margins to prevent the network from trying to extract identity from noise.4

### **AdaFace: Quality Adaptive Margin**

AdaFace introduces further adaptiveness by arguing that the emphasis of the loss function should shift based on image recognizability.28 AdaFace approximates image quality using normalized feature norms calculated with an exponential moving average to remain batch-size independent.28

The intuition behind AdaFace is to prioritize "hard yet recognizable" samples. For high-quality images, the model emphasizes hard samples (those far from the boundary) to improve discriminability. For low-quality images, the model emphasizes easy samples to avoid over-optimizing on images where the identity is fundamentally missing.28 AdaFace has demonstrated superior performance on the IJB-B and IJB-C benchmarks, reducing error rates by over 20% compared to ArcFace and MagFace.28

## **Data Engineering and Large-Scale Benchmarks**

The quality and diversity of training data are the most significant factors in the real-world performance of face recognition systems. The "data gap" between research and industry has been narrowed by benchmarks like WebFace260M.

### **WebFace260M and WebFace42M**

The WebFace260M dataset is a million-scale benchmark containing 4 million identities and 260 million faces.21 Because the original web-scraped data is noisy, researchers developed the Cleaning Automatically utilizing Self-Training (CAST) pipeline.20 CAST uses an iterative teacher-student process: an initial teacher (trained on MS1MV2) cleans the data, and a student is trained on the resulting subset. This student then becomes the new teacher for the next round of purification.20

The final purified dataset, WebFace42M, contains 2 million identities and 42 million faces, representing the largest public training set of its kind.21 Training on WebFace42M reduces the failure rate on IJB-C by 40% compared to traditional datasets.20

### **Benchmarking and Evaluation Protocols**

Standard datasets like Labeled Faces in the Wild (LFW) have become saturated, with many models achieving over 99.8% accuracy.24 Consequently, the focus has shifted to more challenging unconstrained datasets like IJB-C (IARPA Janus Benchmark-C), which includes heavy occlusions, extreme poses, and varying image quality.29

| Dataset | Identities | Images/Videos | Key Challenge |
| :---- | :---- | :---- | :---- |
| LFW | 5,749 | 13,233 | Near-frontal, saturated 29 |
| IJB-C | 3,531 | 31,334 / 11,779 | Extreme pose/occlusion 29 |
| WebFace260M | 4M | 260M | Million-scale noisy data 21 |
| VGGFace2 | 9,131 | 3.31M | High intra-class variation 31 |

For real-time deployment, the Face Recognition Under Inference Time conStraint (FRUITS) protocol provides a framework for evaluating matchers based on strict latency limits (e.g., 100ms, 500ms, or 1000ms), which is critical for smooth user experiences in live camera applications.21

## **GPU Optimization and Deployment with TensorRT**

Designing a real-time system requires exploiting high-performance GPU hardware to handle the millions of parameters involved in modern deep learning models. NVIDIA's TensorRT SDK is the primary tool for optimizing inference on these platforms.

### **Mixed-Precision Inference: FP16 and INT8**

High-end GPUs like the RTX 3090 and RTX 4090 feature dedicated Tensor Cores designed for low-precision matrix multiplication.33 TensorRT optimizes models by converting them to half-precision (FP16) or 8-bit integer (INT8) formats.34

FP16 reduction typically offers a significant speedup with negligible accuracy loss, while INT8 quantization can provide a 4x reduction in model size and even greater throughput.34 However, INT8 requires a calibration step using representative data to ensure that the distribution of activations is correctly mapped to the lower-precision range.34

| Precision | Model Size | Throughput (ResNet50) | Latency |
| :---- | :---- | :---- | :---- |
| FP32 (Baseline) | 1x | 144 images/s | 6.94ms 36 |
| FP16 (Optimized) | 0.5x | 379 images/s | 2.63ms 36 |
| INT8 (Quantized) | 0.25x | 811 images/s | 1.23ms 38 |

### **Hardware Selection: RTX 3090 vs. RTX 4090**

For practitioners with extensive compute resources, the transition to the Ada Lovelace (RTX 4090\) architecture provides a massive performance boost over the Ampere (RTX 3090\) generation. The RTX 4090 features 16,384 CUDA cores and 512 Tensor Cores, offering over 1.6x the training throughput of its predecessor for vision tasks.36

| Feature | NVIDIA RTX 3090 | NVIDIA RTX 4090 | Difference |
| :---- | :---- | :---- | :---- |
| Architecture | Ampere | Ada Lovelace | Generation shift 39 |
| FP16 Performance | 35.58 TFLOPS | 82.58 TFLOPS | \+132% 39 |
| Memory Bandwidth | 936.2 GB/s | 1018 GB/s | \+9% 39 |
| TDP | 350 W | 450 W | \+29% 39 |

While the RTX 4090 consumes more power (450W vs 350W), its throughput-per-watt remains competitive, making it the superior choice for high-concurrency inference and large-scale model training.36 In multi-GPU setups, the lack of NVLink on the RTX 4090 is mitigated by its raw processing power, though scaling for certain models like BERT may drop to 1.7x rather than 2.0x.36

## **Implementation Strategy for Live Recognition**

A complete real-time pipeline involves the sequential execution of detection, tracking, alignment, and recognition. Integrating these components into a unified system requires careful management of the inference budget.

### **Managing the Live Demo Pipeline**

To meet the requirements of a live demo—highlighting faces with bounding boxes and identifying individuals—the following workflow is recommended:

1. **Face Detection**: Deploy a high-speed detector like SCRFD or RetinaFace-MobileNet via TensorRT to maintain a high frame rate (e.g., 30+ FPS).5  
2. **Tracking**: Implement a Multi-Object Tracking (MOT) algorithm to solve occlusion and motion blur issues. Tracking allows the system to maintain identity across frames without performing full recognition on every single frame, significantly saving compute.4  
3. **Pose Filtering**: Use 3DDFAv2 to filter out frames with extreme head poses (![][image7] yaw) where recognition accuracy is inherently lower, selecting only the highest-quality frames for the embedding step.14  
4. **Feature Extraction**: Pass the aligned face into an AdaFace-optimized ResNet-100 or ViT backbone. The 512-dimensional embedding generated here is the "identity signature".6  
5. **Similarity Matching**: Compare the live embedding against the pre-built database (containing 10+ images per team member) using Cosine similarity.  
6. **"Unknown" Signal Logic**: Establish a verification threshold. In a typical scenario, if the maximum cosine similarity to any database entry is below 0.35 (depending on the model's calibration), the person is labeled as "Unknown".3

### **The Challenge of Open-Set Recognition**

A critical aspect of the project objective is the "Unknown" signal, which falls under Open-Set Recognition (OSR). Unlike closed-set classification where the model chooses the best match among known labels, OSR requires the model to reject inputs that do not belong to the training distribution.23

Vision Transformers have shown better generalization for unknown identities because their global attention mechanism captures more robust identity-specific features rather than over-relying on local patterns.23 This leads to a lower False Positive Rate (FPR) in unconstrained environments, which is vital for preventing the system from misidentifying a stranger as a team member.23

## **Final Considerations for High-Performance Implementation**

The implementation of a state-of-the-art face recognition system on high-end NVIDIA GPUs provides the opportunity to explore "heavy" models that are often avoided in mobile or edge contexts. Utilizing a ResNet-100 or a ViT-Large backbone trained on WebFace42M with the AdaFace loss function represents the current upper limit of publicly available technology.

For the live demo, it is essential to perform the final model calibration and TensorRT engine build on the specific hardware used for deployment. Because TensorRT optimizes kernels for the specific GPU architecture (e.g., SM 8.6 for RTX 3090 vs SM 8.9 for RTX 4090), transferring pre-compiled engines between different GPU generations will result in errors or sub-optimal performance.33 By maintaining a strictly optimized pipeline and leveraging the massive parallel processing power of the 40-series GPUs, a system can achieve sub-10ms end-to-end latency, providing a seamless and accurate experience for real-time human face detection and recognition.

#### **Works cited**

1. (PDF) FACE DETECTION USING REFINED-RETINAFACE MODEL \- ResearchGate, accessed March 9, 2026, [https://www.researchgate.net/publication/399465227\_FACE\_DETECTION\_USING\_REFINED-RETINAFACE\_MODEL](https://www.researchgate.net/publication/399465227_FACE_DETECTION_USING_REFINED-RETINAFACE_MODEL)  
2. Comparative Evaluation of Face Detection Algorithms: Accuracy, Efficiency, and Robustness \- Zenodo, accessed March 9, 2026, [https://zenodo.org/records/15023420/files/1-IDJ2253.pdf?download=1](https://zenodo.org/records/15023420/files/1-IDJ2253.pdf?download=1)  
3. Face Recognition on Jetson Nano using TensorRT \- DEV Community, accessed March 9, 2026, [https://dev.to/jhparmar/face-recognition-on-jetson-nano-using-tensorrt-53gi](https://dev.to/jhparmar/face-recognition-on-jetson-nano-using-tensorrt-53gi)  
4. LittleFaceNet: A Small-Sized Face Recognition Method Based on ..., accessed March 9, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11766931/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11766931/)  
5. The Evolution of Face Detection: From Handcrafted Features to ..., accessed March 9, 2026, [https://www.insightface.ai/blog/the-evolution-of-face-detection-from-handcrafted-features-to-deep-learning-frameworks](https://www.insightface.ai/blog/the-evolution-of-face-detection-from-handcrafted-features-to-deep-learning-frameworks)  
6. The Evolution of Face Recognition with Neural Networks: From DeepFace to ArcFace and Beyond | InsightFace Blog, accessed March 9, 2026, [https://www.insightface.ai/blog/the-evolution-of-face-recognition-with-neural-networks-from-deepface-to-arcface-and-beyond](https://www.insightface.ai/blog/the-evolution-of-face-recognition-with-neural-networks-from-deepface-to-arcface-and-beyond)  
7. Sample and Computation Redistribution for Efficient Face Detection \- ResearchGate, accessed March 9, 2026, [https://www.researchgate.net/publication/351511411\_Sample\_and\_Computation\_Redistribution\_for\_Efficient\_Face\_Detection](https://www.researchgate.net/publication/351511411_Sample_and_Computation_Redistribution_for_Efficient_Face_Detection)  
8. Face Detection On Wider Face Hard | SOTA \- HyperAI, accessed March 9, 2026, [https://hyper.ai/en/sota/tasks/face-detection/benchmark/face-detection-on-wider-face-hard](https://hyper.ai/en/sota/tasks/face-detection/benchmark/face-detection-on-wider-face-hard)  
9. GCS-YOLOv8: A Lightweight Face Extractor to Assist Deepfake Detection \- PMC, accessed March 9, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11548442/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548442/)  
10. 68 landmarks are efficient for 3D face alignment: what about more? \- PMC, accessed March 9, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10066970/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10066970/)  
11. Face Alignment Across Large Poses: A 3D Solution \- CVF Open Access, accessed March 9, 2026, [https://openaccess.thecvf.com/content\_cvpr\_2016/papers/Zhu\_Face\_Alignment\_Across\_CVPR\_2016\_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhu_Face_Alignment_Across_CVPR_2016_paper.pdf)  
12. Face Alignment Across Large Poses: A 3D Solution, accessed March 9, 2026, [http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)  
13. arXiv:2009.09960v2 \[cs.CV\] 7 Feb 2021, accessed March 9, 2026, [https://arxiv.org/pdf/2009.09960](https://arxiv.org/pdf/2009.09960)  
14. GitHub \- cleardusk/3DDFA\_V2: The official PyTorch implementation of Towards Fast, Accurate and Stable 3D Dense Face Alignment, ECCV 2020., accessed March 9, 2026, [https://github.com/cleardusk/3DDFA\_V2](https://github.com/cleardusk/3DDFA_V2)  
15. Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network, accessed March 9, 2026, [https://ar5iv.labs.arxiv.org/html/1803.07835](https://ar5iv.labs.arxiv.org/html/1803.07835)  
16. PRNet Extension: 3D Face Reconstruction and Dense Alignment | by Mansi Anand, Yugant Rana, Parul Chopra, accessed March 9, 2026, [https://cmuvisual.medium.com/prnet-extension-3d-face-reconstruction-and-dense-alignment-9570c4559287](https://cmuvisual.medium.com/prnet-extension-3d-face-reconstruction-and-dense-alignment-9570c4559287)  
17. Towards Fast, Accurate and Stable 3D Dense Face Alignment \-Supplementary Material-, accessed March 9, 2026, [https://guojianzhu.com/assets/pdfs/3162-supp.pdf](https://guojianzhu.com/assets/pdfs/3162-supp.pdf)  
18. Comparing CNNs, ResNet, and Vision Transformers for Image Classification \- Medium, accessed March 9, 2026, [https://medium.com/@sameeraperera827/comparing-cnns-resnet-and-vision-transformers-for-image-classification-e7af0604cf44](https://medium.com/@sameeraperera827/comparing-cnns-resnet-and-vision-transformers-for-image-classification-e7af0604cf44)  
19. aylinaydincs/facebenchmark-ArcFacevsQMagFace: A ... \- GitHub, accessed March 9, 2026, [https://github.com/aylinaydincs/facebenchmark-ArcFacevsQMagFace](https://github.com/aylinaydincs/facebenchmark-ArcFacevsQMagFace)  
20. WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition, accessed March 9, 2026, [https://liner.com/review/webface260m-benchmark-unveiling-power-millionscale-deep-face-recognition](https://liner.com/review/webface260m-benchmark-unveiling-power-millionscale-deep-face-recognition)  
21. WebFace260M: A Benchmark Unveiling the ... \- CVF Open Access, accessed March 9, 2026, [https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu\_WebFace260M\_A\_Benchmark\_Unveiling\_the\_Power\_of\_Million-Scale\_Deep\_Face\_CVPR\_2021\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_WebFace260M_A_Benchmark_Unveiling_the_Power_of_Million-Scale_Deep_Face_CVPR_2021_paper.pdf)  
22. Can Vision Transformers with ResNet's Global Features Fairly Authenticate Demographic Faces? \- arXiv.org, accessed March 9, 2026, [https://arxiv.org/html/2506.05383v1](https://arxiv.org/html/2506.05383v1)  
23. Comparing CNN and ViT for Open-Set Face Recognition \- MDPI, accessed March 9, 2026, [https://www.mdpi.com/2079-9292/14/19/3840](https://www.mdpi.com/2079-9292/14/19/3840)  
24. Overleaf Example \- IEEE SigPort, accessed March 9, 2026, [https://sigport.org/sites/all/modules/pubdlcnt/pubdlcnt.php?fid=10093](https://sigport.org/sites/all/modules/pubdlcnt/pubdlcnt.php?fid=10093)  
25. FaceLiVT: Face Recognition using Linear Vision Transformer with Structural Reparameterization For Mobile Device \- arXiv.org, accessed March 9, 2026, [https://arxiv.org/pdf/2506.10361](https://arxiv.org/pdf/2506.10361)  
26. Face Recognition On Lfw | SOTA \- HyperAI, accessed March 9, 2026, [https://hyper.ai/en/sota/tasks/face-recognition/benchmark/face-recognition-on-lfw](https://hyper.ai/en/sota/tasks/face-recognition/benchmark/face-recognition-on-lfw)  
27. Development of a Deep Learning Model with a Novel Loss Function for Facial Recognition based on ArcFace, accessed March 9, 2026, [https://thesis.unipd.it/retrieve/196feecd-decc-4a12-a6d9-a7aeb2a73112/Peloso\_Mario\_Giovanni.pdf](https://thesis.unipd.it/retrieve/196feecd-decc-4a12-a6d9-a7aeb2a73112/Peloso_Mario_Giovanni.pdf)  
28. AdaFace: Quality Adaptive Margin for Face Recognition, accessed March 9, 2026, [http://cvlab.cse.msu.edu/project-adaface.html](http://cvlab.cse.msu.edu/project-adaface.html)  
29. IARPA Janus Benchmark – C: Face Dataset and Protocol ∗ \- Biometrics Research Group, accessed March 9, 2026, [http://biometrics.cse.msu.edu/Publications/Face/Mazeetal\_IARPAJanusBenchmarkCFaceDatasetAndProtocol\_ICB2018.pdf](http://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf)  
30. MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition \- Microsoft, accessed March 9, 2026, [https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/MSCeleb-1M-a.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/MSCeleb-1M-a.pdf)  
31. Leveraging Large-Scale Face Datasets for Deep Periocular Recognition via Ocular Cropping \- arXiv, accessed March 9, 2026, [https://arxiv.org/html/2510.26294v1](https://arxiv.org/html/2510.26294v1)  
32. WebFace260M: A Benchmark for Million-Scale Deep Face Recognition \- PubMed, accessed March 9, 2026, [https://pubmed.ncbi.nlm.nih.gov/35471873/](https://pubmed.ncbi.nlm.nih.gov/35471873/)  
33. Benchmarking NVIDIA TensorRT-LLM \- Jan, accessed March 9, 2026, [https://www.jan.ai/post/benchmarking-nvidia-tensorrt-llm](https://www.jan.ai/post/benchmarking-nvidia-tensorrt-llm)  
34. TensorRT Export for YOLO26 Models \- Ultralytics YOLO Docs, accessed March 9, 2026, [https://docs.ultralytics.com/integrations/tensorrt/](https://docs.ultralytics.com/integrations/tensorrt/)  
35. Worse performance after quantization on TensorRT \- NVIDIA Developer Forums, accessed March 9, 2026, [https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549](https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549)  
36. NVIDIA GeForce RTX 4090 Vs RTX 3090 Deep Learning Benchmark \- ProX PC, accessed March 9, 2026, [https://www.proxpc.com/blogs/nvidia-geforce-rtx-4090-vs-rtx-3090-deep-learning-benchmark](https://www.proxpc.com/blogs/nvidia-geforce-rtx-4090-vs-rtx-3090-deep-learning-benchmark)  
37. NVIDIA GeForce RTX 4090 vs RTX 3090 Deep Learning Benchmark \- Lambda, accessed March 9, 2026, [https://lambda.ai/blog/nvidia-rtx-4090-vs-rtx-3090-deep-learning-benchmark](https://lambda.ai/blog/nvidia-rtx-4090-vs-rtx-3090-deep-learning-benchmark)  
38. Best Practices — NVIDIA TensorRT, accessed March 9, 2026, [https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)  
39. GPU Benchmarks NVIDIA RTX 3090 vs. NVIDIA RTX 4090 \- Bizon-tech, accessed March 9, 2026, [https://bizon-tech.com/gpu-benchmarks/NVIDIA-RTX-3090-vs-NVIDIA-RTX-4090/579vs637](https://bizon-tech.com/gpu-benchmarks/NVIDIA-RTX-3090-vs-NVIDIA-RTX-4090/579vs637)  
40. Best Object Detection Models 2025: RF-DETR, YOLOv12 & Beyond \- Roboflow Blog, accessed March 9, 2026, [https://blog.roboflow.com/best-object-detection-models/](https://blog.roboflow.com/best-object-detection-models/)  
41. Recognition accuracy for IJB-C dataset, ResNet-50 (VGGFace2) \- ResearchGate, accessed March 9, 2026, [https://www.researchgate.net/figure/Recognition-accuracy-for-IJB-C-dataset-ResNet-50-VGGFace2\_tbl3\_349607007](https://www.researchgate.net/figure/Recognition-accuracy-for-IJB-C-dataset-ResNet-50-VGGFace2_tbl3_349607007)  
42. Benchmarking with TensorRT-LLM \- ProX PC, accessed March 9, 2026, [https://www.proxpc.com/blogs/benchmarking-with-tensorrt-llm](https://www.proxpc.com/blogs/benchmarking-with-tensorrt-llm)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAXCAYAAABdy4LVAAADFUlEQVR4Xu2YS6hOURTH10FIQkqJGCrkOVDKjOQ18BjIlAkDIxOZmhgYoIyYMRAlJBIzipTpHWDikZv3wOQSrvU/6+xz9l5nrePbX9/XvcmvVufu/1p777XX2efs812ivii00Cf54+T3+M/AOVDZRu3oJuPeVaEZPYbJXmrW7PKDbTnbIu0YGJOkGhXLSNY7rh0xY1oAvI6LJB1fsy1V7i7WsH0m6fuEbZ5bFE9PucSBWEQOB9lGefyffD2lndW82UX5zbY1amOAbVG7IV3YYbZzUfsySd/1tdJLIQp6RJID+sJWpAGdXGN7F7WRw5fQiKbvpSh1+Elqd9hlaBZhEX/TegXFzC0K4qcZmr6pnTnpnYLgEaUB6Eu0qO78W7ZxtRnyi9IMUBWlcIui5jpD9lzQXhqai1WUB0oD0ONHoweKEyT9dmhPC/uxyt0pODTUYsuBrRuj2wlWUW4pDUC/q8UOdpP0OasdGeQWxVo8sHTdToiLMoUk+HqkBaC/0KLDabarbL/YNgfR3gy2WhGKslI7HKzFA0uXtjO9tVNuKA1Af6jFv7CYpN9t7SipEorzUjmGoqxKZZdq8a2V+kVxsIpyT2kA+gUtBlpplLjPc6+EF+1q7XDAd4k1l5WDtK3EC7soI0Ys9P1aVOBx0YULCW1SuomzU9bGYiu3RrhP7cUDaPjuSbTWOBFjahYcx3rgDYaGnwV7ovY+8u8IbGrZkrkOsS2oIxpdE4qyLhaj0KNs05smLaT2/ADaMUNz0TtlPkmHmUHgJL7x5VkdIXgFmBG1se2h3Yk0/Ayw+lrgKEccPh412D3WONgRV6L2TmrHAEur0UUBW0g63WT7yvY4dZccp/YLGQVFUrBPJGOcTyKEp2wf5ZabW+QD23uSj8E31RXaqziIZByrYPheeU5yMCCH2am7JLsoA8JYcCPhpTiReO+UUh5iUXyK+pvHSW2YyJRD3Cn9rWmU+u05COSx9YpSer+zzWWblfqGSs7/ZzrprGzsbP6eQ7JeKYo1QIGjqiiPq+3a949yhOR41kf0xGDdlMnEHwQ2zEpNs54vAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAXCAYAAABdy4LVAAADQklEQVR4XtWYPawNURDHZxEEQYRECIlGokFIJBIdEV+Fj0JEIaGhEIVGtBqFAokKnUJIBBEfoVMQiVaBCqHwVQiej+ea/52zu3Nm5+zuve67+CWTvPOfOWdnZnfPnvuINFk06oM+F2g9LWsfOgj6v1j/M8eKQWa0K9hq6xAGeak/pDaVWqdmO5U1J/nOtoRtnnX8X7RuymKSejvWoRmxQuAcycQXfLmF1lnDMrb3JHMfss3MHd20G3JX7n1sn0nWuV3KrdjL9obtJ9vxUo4u3nNTfrGtV+MOr7dBjVPsZzutxhdILr5CaT5xsx6z7Ql/jydZAzapiEg39zLbazVGDh/ygZrWU1OOUXXCFkeLCBfLk9d4Wh3j2H5Q/HROJVkDr3oTiJvgaPamSk6J5hZNUYU9yTU1C/qCUnd5RdUGpJsSljavFZLn+Cyak6l1EnWAk+RfC9pzR0tinxQE3zMaQKL61WjDUZL1NllHA+fZFhktNEW3pNIePElesd6NseOIEbM2gq9HigD9lhWTZLSVZM4p6+qDKSRrvbQOg1c88HQ7jtBPCt5nBF9RWg70Z1as3qwuJ9gusY2ye6119k72hRqKCHjFA0+XsZd/VjQl2juu5gMF9PtWbGB+2AtuWIeP+2ocIlkDX6EmiuJNremmJPD2lDvlMGrW2VJvR8sN0oXjV5IcD9qCc4lXbLopiaS8pqivTzEP+k6tO4xS1LjuzDyhNaXeitlsX+XPIvOmPeUuVYsH0LrNLXsQf90stiloiJ2wytHws2CbGu8g546oJ0U//jitzlFje8cQ+zFSBJvDQbaJajyXqjEA2mFHS2KbMit0cbLSPpGcMjWVBoRxeeokWhq0m0rDzwBvrqYTcrCmX6XlStcg5qIab6ZqDPC0AtsUsI5k0jWSO/YgdoPsCBUbcnGbuaHdpGDvSNY4Y2LAI7a3WlDs5lDbjE544tRe1wXr4LRNZn2cV56SfBgwb5p2BnpuyjDApjh4EhunEDnRaJ+sengbCpl35hkTksX95Selmhd+1ldVDx3VbkZbapvyjW0GyVF6WNT/f6ZN8SHGD/XVwHSSemubgk8VbKN19ExtLv8MB6is2Wd4dQzvSv3yG6R63Ed2DXn1AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAABN0lEQVR4Xp2TMUoEQRBFa1iMxAuooZEmHkAwMTASNjMUjyBiIiKLRzASDPYqsrAXMNrUXFMTQf+vKadrqrvHcT/Uds/7f2q6C1ZkHTUReNXMAk+oYCb1zcFoTbWXPM8yGajJguV8gRbQSP33zdqE9cl+yj0Ld8L+EMsTatPQFuoOddXaXbMb7OdYdyzXaQO1QJ0h+I30A/Zz8+6FTGQftUJNUNvGdi2jerH1Qlpzliw9EdmbYxTZtQc8NsUv0TTpFS77TDXRE4scBE41NJYB8gSxyQzdEwuTp3HaR8p+r+vZu+5Cg3PJv0iRHYc02Um7bT698Wqmz3fzcC2ObB7ULWovWSJfqEcPoGfUR3q0Vg2vovObJm+UwuWj/rCDBtP9/0gtWuMj1c1jQIPmWhrTsZDJUA1kvNUPw1QuYeJ/xo0AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABgUlEQVR4Xp2UvS4GQRSGzwihJiEUNBQugehEFAqhUUpEFCpRibgCakoJlQsQEZW4AK2KOyBRiZ/P5z0zZ3bPzOzPfJ7k3Z3zzsyZszs7S5SDiQ2Nae5u6XXkjHHUjazzNWGlTTN2oL3YtGRlCDuGYHRxH4MWoM+gl6lIam+1CxBxwvkoHlRxSEMizzX0HVqmi3lLoZfPCHFVhtYinyvdj7xsbsgl0EyLtxz5lPXcsjkdqE+99jNKF8qmn9zkF+heib38pPpLAFvkJk9J7GHvKfKE9qN5Sr6iciRvGHsThdNGVOkJpY/5jt63yGupLmSFgkrNcBGXjJP7ji+U9+obNSfK7r6H26sqZu6gUcz7kXiA0oUTOAkP+oVmrJOsbB5w2ZbgCLqS9iS0Ke160nz2qiv7gGZVXEWSpor4FTHH5Kr+NwfQM8mRljL4D9YpS8oqrmADWpT2I7Su+s6hQx8EaZPPIlz0i9yPexf+beG6g3Vp7D+jB1TuubKZRVhWb28m5Q/1RkAZzp5efgAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAYCAYAAAB5j+RNAAAB/klEQVR4Xr2VOy8FQRTHZ1BIEAkfgaAVGoWaSis60alVPoBK4WOIeDSiEI1GJVEgkXgWJOJR0HmE65zdM3fmnMzMzu517y/5350zZ+a/Z2Zn7yrlollkCfU3A3avVt64NTRjRXHPQREPi7gK3FNX8syq/hGd5zyMryxAgWc6NRZpdcViQ7kauacKeCYgjW5EXIX/8My2Qxrdlt0mDx5PF+4fu5vXKDYhAa9nFWJG7VDmLly3SNieotw+aId0QH2GmGcCdmuk0Z3TbgOtqdwcx6HGaO4RxZhHucQ8E6DidLrRr8rHnoH6qT3KRlhSPQvhRjpohMsxu4da4mlGUXGdoDeVj1sROUZqcciIssUNiJxL3ZMekPQ8cdrfoA0nZshVxg7vK+hL2QLrJ1e83UWemO+m9jLFXmRCGhkOlf0s4ecI5+E5tIReMh30RPZAn7LTIItz/837QLOgVZWPm6F+eCG02T18ZHOgHsohMU8Jju2SnQZpdF1vaTUOP0/Qugc9Knt2JkAvoAfKPYOGKIfInbOenHdQr+x0qRvRU7k0sY/EL4dccObpzoX2KVw6KNx0UgxpdIE/1kinFuTi8WQu66AF0DxoEfThJl2kEa6IUa64bHRNTJKe5rwabZtEPs1OllUfizhIpOjKnpJpEU+KOB1brdczsphyNGLUyFzkD2OFhHXLMjWKAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAXCAYAAABNq8wJAAACJ0lEQVR4Xu2WzytFQRTHz5X8KFtSJGRBIisbRYlY2/kT5E9QSvEHKFbS27GxsPIrKxFKib2QUhLKr4Vfzzl3Zszcc+fNve++93gLn/r25nzP3Jkz8+bdeQAhPG7khjmcpZ1pNktXh5GUvA0Ul6gJo/KJKdjAf8FvLSZ/80ygPqWGWI5oQR2i0qhtliOqgJ71YBOLOs5TWbGHeUctyHY56gtVodPQB6JwRReLiVPd9Odd07Gmmxt54A51a8SzIIoz56J4zIiJN9SBEQ8bbeKGxT5lqAvULti2N+xE0QSiuAbmdxrtGhB96NNkS/qKPTb9nviwF1WCOkGdoypZLhv2QRdBmzNo5BSTED4uRAqCfjXqCTWHutS2fQEm66hHVC1PxIAKIC2DOOftMqZjpFiVHmce7H5iaEc+UB084UAtYEoZuF/N0quX1o6MOeq3Uhdwozbcng+4MyAG7jXNDKgFcMi7lu0lGXPwqHjkl4rQXlkSxkFMOMoTITznAtKypEy/gUWw+4mZBjFgP084UBeT5GcXzYX1yHbwLeSF3kKJoZ2gi6iNJ2LQCvYiyEsZh4LiER36PKPudZj9EdpAPYB4fblxj/0KNJbuQ38p+KLw74H/gvDxxIjUp1F5caEHj1BnELzq3bgXQFyBKIhuV/obQfcMh+6dF9QKiL4DwXQ86HqPLicDiR8MEGeUOH3+KRb8b8vylVms7Mh5ABsFGTTML03jpiiK0HwDCFFtoz4a2bcAAAAASUVORK5CYII=>