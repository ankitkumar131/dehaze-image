# Image Dehazing System Using Dark Channel Prior and Deep Learning Approaches

## Abstract

This thesis presents a comprehensive study and implementation of image dehazing techniques to address the challenges posed by haze, fog, and smoke in digital images. Haze significantly degrades image quality by reducing contrast and color fidelity, which adversely affects various computer vision applications including surveillance, autonomous driving, and remote sensing. This research explores two primary approaches to image dehazing: a physics-based method utilizing the Dark Channel Prior (DCP) with several enhancements, and a learning-based method employing a U-Net convolutional neural network architecture. The physics-based approach incorporates guided filtering for transmission map refinement and adaptive contrast enhancement techniques to overcome limitations of the original DCP algorithm. The deep learning approach leverages a dataset of paired hazy and clear images to train a model capable of directly mapping hazy inputs to their dehazed counterparts. Both methods are implemented in a user-friendly application with a modern graphical interface, allowing for real-time processing of both still images and video content. Extensive experimental results demonstrate that our enhanced DCP method achieves superior performance compared to the original algorithm, particularly in preserving edge details and natural colors. The deep learning model shows promising results with the potential for further improvement through additional training data and architectural refinements. This research contributes to the field by providing a comparative analysis of traditional and learning-based dehazing techniques, along with an accessible tool for practical applications in various domains including environmental monitoring, transportation safety, and computer vision preprocessing.

## Table of Contents

- [List of Figures](#list-of-figures)
- [List of Tables](#list-of-tables)
- [List of Abbreviations](#list-of-abbreviations)
- [Chapter 1: Introduction](#chapter-1-introduction)
  - [1.1 Background](#11-background)
  - [1.2 Problem Statement](#12-problem-statement)
  - [1.3 Research Significance](#13-research-significance)
  - [1.4 Thesis Structure](#14-thesis-structure)
- [Chapter 2: Literature Review, Motivation, and Objective](#chapter-2-literature-review-motivation-and-objective)
  - [2.1 Image Formation Model](#21-image-formation-model)
  - [2.2 Traditional Dehazing Methods](#22-traditional-dehazing-methods)
    - [2.2.1 Dark Channel Prior](#221-dark-channel-prior)
    - [2.2.2 Color Attenuation Prior](#222-color-attenuation-prior)
    - [2.2.3 Non-local Prior](#223-non-local-prior)
    - [2.2.4 Contrast Enhancement-Based Methods](#224-contrast-enhancement-based-methods)
    - [2.2.5 Fusion-Based Methods](#225-fusion-based-methods)
  - [2.3 Learning-Based Dehazing Methods](#23-learning-based-dehazing-methods)
    - [2.3.1 CNN-Based Methods](#231-cnn-based-methods)
    - [2.3.2 GAN-Based Methods](#232-gan-based-methods)
    - [2.3.3 Transformer-Based Methods](#233-transformer-based-methods)
    - [2.3.4 Physics-Guided Learning Methods](#234-physics-guided-learning-methods)
    - [2.3.5 Unsupervised and Self-Supervised Methods](#235-unsupervised-and-self-supervised-methods)
  - [2.4 Evaluation Metrics](#24-evaluation-metrics)
    - [2.4.1 Full-Reference Metrics](#241-full-reference-metrics)
    - [2.4.2 No-Reference Metrics](#242-no-reference-metrics)
    - [2.4.3 Human Visual Perception](#243-human-visual-perception)
  - [2.5 Challenges and Limitations](#25-challenges-and-limitations)
    - [2.5.1 Scene Diversity and Complexity](#251-scene-diversity-and-complexity)
    - [2.5.2 Color Distortion and Artifacts](#252-color-distortion-and-artifacts)
    - [2.5.3 Computational Efficiency](#253-computational-efficiency)
    - [2.5.4 Training Data Scarcity](#254-training-data-scarcity)
    - [2.5.5 Domain Gap and Generalization](#255-domain-gap-and-generalization)
  - [2.6 Research Motivation](#26-research-motivation)
  - [2.7 Research Objectives](#27-research-objectives)
- [Chapter 3: Proposed System](#chapter-3-proposed-system)
  - [3.1 System Overview](#31-system-overview)
    - [3.1.1 System Architecture](#311-system-architecture)
    - [3.1.2 Component Interaction](#312-component-interaction)
    - [3.1.3 Implementation Environment](#313-implementation-environment)
  - [3.2 Enhanced Dark Channel Prior Method](#32-enhanced-dark-channel-prior-method)
    - [3.2.1 Dark Channel Computation](#321-dark-channel-computation)
    - [3.2.2 Atmospheric Light Estimation](#322-atmospheric-light-estimation)
    - [3.2.3 Transmission Map Estimation](#323-transmission-map-estimation)
    - [3.2.4 Guided Filtering for Transmission Refinement](#324-guided-filtering-for-transmission-refinement)
    - [3.2.5 Scene Radiance Recovery](#325-scene-radiance-recovery)
    - [3.2.6 Adaptive Contrast Enhancement](#326-adaptive-contrast-enhancement)
    - [3.2.7 Sky Region Detection and Processing](#327-sky-region-detection-and-processing)
  - [3.3 Deep Learning-Based Dehazing Method](#33-deep-learning-based-dehazing-method)
    - [3.3.1 U-Net Architecture](#331-u-net-architecture)
    - [3.3.2 Dataset Preparation](#332-dataset-preparation)
    - [3.3.3 Data Augmentation](#333-data-augmentation)
    - [3.3.4 Training Process](#334-training-process)
    - [3.3.5 Model Optimization](#335-model-optimization)
    - [3.3.6 Loss Function Design](#336-loss-function-design)
    - [3.3.7 Inference Pipeline](#337-inference-pipeline)
  - [3.4 Video Dehazing Extension](#34-video-dehazing-extension)
    - [3.4.1 Frame Extraction and Processing](#341-frame-extraction-and-processing)
    - [3.4.2 Temporal Consistency Preservation](#342-temporal-consistency-preservation)
    - [3.4.3 Video Reconstruction](#343-video-reconstruction)
    - [3.4.4 Performance Optimization](#344-performance-optimization)
  - [3.5 Graphical User Interface](#35-graphical-user-interface)
    - [3.5.1 Design Principles](#351-design-principles)
    - [3.5.2 Implementation Details](#352-implementation-details)
    - [3.5.3 User Interaction Flow](#353-user-interaction-flow)
    - [3.5.4 Visualization Features](#354-visualization-features)
    - [3.5.5 Parameter Control Panel](#355-parameter-control-panel)
- [Chapter 4: Results and Discussion](#chapter-4-results-and-discussion)
  - [4.1 Experimental Setup](#41-experimental-setup)
    - [4.1.1 Hardware and Software Configuration](#411-hardware-and-software-configuration)
    - [4.1.2 Dataset Description](#412-dataset-description)
    - [4.1.3 Evaluation Methodology](#413-evaluation-methodology)
    - [4.1.4 Implementation Details](#414-implementation-details)
  - [4.2 Qualitative Analysis](#42-qualitative-analysis)
    - [4.2.1 Visual Comparison of Dehazing Methods](#421-visual-comparison-of-dehazing-methods)
    - [4.2.2 Case Studies](#422-case-studies)
    - [4.2.3 Visual Assessment by Expert Panel](#423-visual-assessment-by-expert-panel)
  - [4.3 Quantitative Analysis](#43-quantitative-analysis)
    - [4.3.1 Image Quality Metrics](#431-image-quality-metrics)
    - [4.3.2 Computational Efficiency](#432-computational-efficiency)
    - [4.3.3 Statistical Significance of Results](#433-statistical-significance-of-results)
  - [4.4 Ablation Studies](#44-ablation-studies)
    - [4.4.1 Impact of Parameter Selection](#441-impact-of-parameter-selection)
    - [4.4.2 Contribution of Individual Components](#442-contribution-of-individual-components)
    - [4.4.3 Network Architecture Variations](#443-network-architecture-variations)
  - [4.5 Comparative Analysis](#45-comparative-analysis)
    - [4.5.1 Comparison with State-of-the-Art Methods](#451-comparison-with-state-of-the-art-methods)
    - [4.5.2 Strengths and Limitations](#452-strengths-and-limitations)
    - [4.5.3 Performance Across Different Haze Densities](#453-performance-across-different-haze-densities)
    - [4.5.4 Performance on Various Scene Types](#454-performance-on-various-scene-types)
  - [4.6 User Experience Evaluation](#46-user-experience-evaluation)
    - [4.6.1 Survey Design and Methodology](#461-survey-design-and-methodology)
    - [4.6.2 Usability Assessment Results](#462-usability-assessment-results)
    - [4.6.3 User Feedback Analysis](#463-user-feedback-analysis)
- [Chapter 5: Conclusion and Future Scope](#chapter-5-conclusion-and-future-scope)
  - [5.1 Summary of Contributions](#51-summary-of-contributions)
  - [5.2 Limitations of the Current Work](#52-limitations-of-the-current-work)
  - [5.3 Future Research Directions](#53-future-research-directions)
    - [5.3.1 Algorithm Improvements](#531-algorithm-improvements)
    - [5.3.2 Application Extensions](#532-application-extensions)
    - [5.3.3 Integration with Other Computer Vision Tasks](#533-integration-with-other-computer-vision-tasks)
    - [5.3.4 Hardware Acceleration and Optimization](#534-hardware-acceleration-and-optimization)
    - [5.3.5 Multi-Modal Dehazing Approaches](#535-multi-modal-dehazing-approaches)
  - [5.4 Concluding Remarks](#54-concluding-remarks)
- [References](#references)
- [Appendices](#appendices)
  - [Appendix A: Code Implementation](#appendix-a-code-implementation)
  - [Appendix B: Additional Experimental Results](#appendix-b-additional-experimental-results)
  - [Appendix C: User Manual](#appendix-c-user-manual)
  - [Appendix D: Dataset Details](#appendix-d-dataset-details)
  - [Appendix E: Mathematical Derivations](#appendix-e-mathematical-derivations)

## List of Figures

1. Fig. 1.1: Examples of hazy images and their impact on visibility
2. Fig. 1.2: Applications affected by haze in images
3. Fig. 1.3: Comparison of clear and hazy images in autonomous driving scenarios
4. Fig. 1.4: Impact of haze on surveillance system effectiveness
5. Fig. 2.1: Atmospheric scattering model illustration
6. Fig. 2.2: Dark channel prior examples
7. Fig. 2.3: Comparison of traditional dehazing methods
8. Fig. 2.4: Evolution of learning-based dehazing approaches
9. Fig. 2.5: Contrast enhancement-based dehazing results
10. Fig. 2.6: Fusion-based dehazing framework
11. Fig. 2.7: CNN architecture evolution for image dehazing
12. Fig. 2.8: GAN-based dehazing model structure
13. Fig. 2.9: Transformer architecture for image dehazing
14. Fig. 2.10: Physics-guided neural network approach
15. Fig. 2.11: Unsupervised dehazing framework
16. Fig. 3.1: System architecture overview
17. Fig. 3.2: Component interaction diagram
18. Fig. 3.3: Dark channel computation process
19. Fig. 3.4: Atmospheric light estimation visualization
20. Fig. 3.5: Transmission map before and after refinement
21. Fig. 3.6: Guided filtering process illustration
22. Fig. 3.7: Contrast enhancement effect demonstration
23. Fig. 3.8: Sky region detection and processing
24. Fig. 3.9: U-Net architecture for dehazing
25. Fig. 3.10: Dataset preparation pipeline
26. Fig. 3.11: Data augmentation examples
27. Fig. 3.12: Training loss curves
28. Fig. 3.13: Model optimization process
29. Fig. 3.14: Loss function comparison
30. Fig. 3.15: Inference pipeline diagram
31. Fig. 3.16: Video dehazing framework
32. Fig. 3.17: Temporal consistency preservation
33. Fig. 3.18: GUI design and components
34. Fig. 3.19: User interaction workflow
35. Fig. 3.20: Visualization features demonstration
36. Fig. 4.1: Visual comparison on synthetic hazy images
37. Fig. 4.2: Visual comparison on real-world hazy images
38. Fig. 4.3: Case study: Extreme haze conditions
39. Fig. 4.4: Case study: Night-time hazy scenes
40. Fig. 4.5: Case study: Urban environments with complex structures
41. Fig. 4.6: Expert visual assessment results
42. Fig. 4.7: Performance metrics comparison chart
43. Fig. 4.8: Computation time comparison
44. Fig. 4.9: Statistical significance analysis
45. Fig. 4.10: Effect of patch size parameter
46. Fig. 4.11: Effect of omega parameter
47. Fig. 4.12: Component contribution analysis
48. Fig. 4.13: Network architecture variations comparison
49. Fig. 4.14: Comparison with state-of-the-art methods
50. Fig. 4.15: Performance across different haze densities
51. Fig. 4.16: Performance on various scene types
52. Fig. 4.17: User experience survey results
53. Fig. 5.1: Future research directions overview

## List of Tables

1. Table 2.1: Summary of traditional dehazing methods
2. Table 2.2: Summary of learning-based dehazing methods
3. Table 2.3: Comparison of evaluation metrics
4. Table 2.4: Challenges in image dehazing and proposed solutions
5. Table 3.1: System components and their functions
6. Table 3.2: Enhanced DCP algorithm parameters
7. Table 3.3: U-Net architecture details
8. Table 3.4: Dataset statistics
9. Table 3.5: Data augmentation techniques
10. Table 3.6: Training hyperparameters
11. Table 3.7: Loss function components
12. Table 3.8: Video processing parameters
13. Table 3.9: GUI components and their functionality
14. Table 4.1: Hardware specifications
15. Table 4.2: Software dependencies
16. Table 4.3: Synthetic dataset characteristics
17. Table 4.4: Real-world dataset characteristics
18. Table 4.5: Evaluation metrics and their significance
19. Table 4.6: Quantitative results on synthetic dataset
20. Table 4.7: Quantitative results on real-world dataset
21. Table 4.8: Computation time for different image resolutions
22. Table 4.9: Statistical analysis of performance metrics
23. Table 4.10: Ablation study results
24. Table 4.11: Network architecture variations performance
25. Table 4.12: Comparison with state-of-the-art methods
26. Table 4.13: Performance across different haze densities
27. Table 4.14: Performance on various scene types
28. Table 4.15: User experience survey results
29. Table 4.16: Expert feedback summary
30. Table 5.1: Summary of contributions
31. Table 5.2: Limitations and potential solutions
32. Table 5.3: Future research directions

## List of Abbreviations

- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **CNN**: Convolutional Neural Network
- **DCP**: Dark Channel Prior
- **FADE**: Fog Aware Density Evaluator
- **FCN**: Fully Convolutional Network
- **GAN**: Generative Adversarial Network
- **GPU**: Graphics Processing Unit
- **GUI**: Graphical User Interface
- **HDR**: High Dynamic Range
- **HVS**: Human Visual System
- **IoU**: Intersection over Union
- **LIDAR**: Light Detection and Ranging
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **NIQE**: Natural Image Quality Evaluator
- **NLP**: Non-Local Prior
- **PSNR**: Peak Signal-to-Noise Ratio
- **RGB**: Red, Green, Blue
- **ROI**: Region of Interest
- **SSIM**: Structural Similarity Index Measure
- **SVD**: Singular Value Decomposition
- **UI**: User Interface
- **UX**: User Experience
- **VGG**: Visual Geometry Group (neural network)

# Chapter 1: Introduction

## 1.1 Background

Digital images have become an integral part of modern society, serving critical roles in various applications including surveillance, autonomous driving, remote sensing, and medical imaging. However, the quality of these images can be significantly degraded by atmospheric conditions such as haze, fog, and smoke. These phenomena introduce a semi-transparent layer that reduces contrast, fades colors, and obscures details in the captured scenes.

Haze formation is primarily caused by atmospheric particles that absorb and scatter light rays traveling from the scene to the camera. This scattering process follows the atmospheric scattering model, which describes how light interacts with particles in the atmosphere. The visual degradation becomes more severe as the distance between the camera and the scene increases, creating a depth-dependent effect that makes distant objects appear more hazy than closer ones.

The presence of haze in images poses significant challenges for both human perception and computer vision systems. For humans, hazy images provide less visual information, making it difficult to identify objects and assess scenes accurately. For computer vision algorithms, haze reduces the effectiveness of feature extraction, object detection, and scene understanding processes, potentially leading to system failures in critical applications like autonomous driving or surveillance.

The impact of haze on computer vision systems is particularly concerning in safety-critical applications. For instance, in autonomous driving, haze can significantly reduce the visibility range of both cameras and LIDAR sensors, making it difficult to detect pedestrians, vehicles, and road signs at a safe distance. Similarly, in surveillance systems, haze can obscure important details such as facial features or license plate numbers, rendering the surveillance footage less useful for security and forensic purposes.

In remote sensing and satellite imaging, haze can obscure ground features and affect the accuracy of land cover classification, vegetation monitoring, and urban planning applications. In medical imaging, while not affected by atmospheric haze, similar degradation effects can occur due to tissue properties, reducing the diagnostic value of the images.

The economic impact of haze-related visibility issues is substantial. According to transportation safety studies, reduced visibility conditions contribute to thousands of accidents annually, resulting in significant human and economic costs. In the aviation industry, haze-related flight delays and cancellations cost millions of dollars each year. For outdoor surveillance systems, the effectiveness of security infrastructure worth billions of dollars is compromised during hazy conditions.

Consequently, image dehazing has emerged as an important research area in computer vision and image processing. The goal of image dehazing is to recover a clear, haze-free image from its hazy counterpart, effectively reversing the degradation caused by atmospheric conditions. This process involves estimating and removing the haze component while preserving the original scene details and natural appearance.

Over the past two decades, numerous approaches have been proposed for image dehazing, ranging from traditional physics-based methods to modern learning-based techniques. Physics-based methods typically rely on the atmospheric scattering model and various priors or assumptions about the statistical properties of haze-free images. Learning-based methods, on the other hand, leverage deep neural networks to learn the mapping from hazy to clear images directly from data, without explicitly modeling the physical process of haze formation.

Despite significant progress, image dehazing remains a challenging problem due to its ill-posed nature and the complexity of real-world haze distributions. The development of effective, efficient, and robust dehazing algorithms continues to be an active area of research, with important implications for various applications that rely on high-quality visual information.

## 1.2 Problem Statement

Despite significant advances in image dehazing techniques over the past decade, several challenges remain unresolved. The primary challenge lies in accurately estimating the atmospheric light and transmission map, which are essential components of the atmospheric scattering model used for haze removal. Traditional methods often rely on prior assumptions about the statistical properties of haze-free images, which may not hold true for all scenes and conditions.

The Dark Channel Prior (DCP), proposed by He et al., has been widely adopted due to its effectiveness and simplicity. However, it suffers from several limitations, including color distortion in sky regions, halo artifacts around depth discontinuities, and high computational complexity. While various improvements have been proposed to address these issues, a comprehensive solution that maintains a balance between effectiveness and efficiency is still needed.

More recently, learning-based approaches using deep neural networks have shown promising results by directly learning the mapping from hazy to clear images. However, these methods typically require large amounts of training data consisting of paired hazy and clear images, which are difficult to obtain in real-world scenarios. Additionally, models trained on synthetic data often struggle to generalize to real-world hazy conditions due to the domain gap between synthetic and real haze distributions.

Furthermore, most existing dehazing methods focus on processing still images, with limited attention given to video dehazing, which introduces additional challenges related to temporal consistency and real-time processing requirements. The development of a comprehensive dehazing system that can effectively process both images and videos while maintaining visual quality and computational efficiency remains an open research problem.

The specific problems addressed in this thesis can be summarized as follows:

1. **Limitations of Traditional Methods**: Existing physics-based methods like DCP often make simplifying assumptions that do not hold for all scenes, leading to artifacts and quality issues in the dehazed images. These methods also tend to be computationally expensive, limiting their practical applicability.

2. **Data Requirements for Learning-Based Methods**: Deep learning approaches require large amounts of paired training data (hazy and clear image pairs), which are difficult to collect in real-world settings. Creating synthetic haze often does not accurately represent the complexity of natural haze distributions.

3. **Generalization to Diverse Scenes**: Both traditional and learning-based methods often struggle with diverse scene types, such as images containing large sky regions, night-time scenes, or scenes with non-uniform haze distribution.

4. **Computational Efficiency vs. Quality Trade-off**: There is a persistent trade-off between dehazing quality and computational efficiency. High-quality dehazing often requires complex algorithms or deep networks that are computationally intensive, making real-time processing challenging.

5. **Temporal Consistency in Videos**: Applying image dehazing algorithms to video frames independently often results in temporal inconsistencies, such as flickering or jittering effects, which degrade the visual quality of the dehazed video.

6. **Usability and Accessibility**: Many existing dehazing algorithms are implemented as research prototypes without user-friendly interfaces, making them inaccessible to non-expert users who could benefit from dehazing technology.

7. **Objective Evaluation**: The lack of standardized evaluation protocols and the subjective nature of image quality assessment make it difficult to fairly compare different dehazing methods and measure progress in the field.

Addressing these challenges requires a multifaceted approach that combines the strengths of traditional physics-based methods and modern learning-based techniques, while also considering practical aspects such as computational efficiency, user interface design, and evaluation methodology.

## 1.3 Research Significance

This research addresses the aforementioned challenges by developing an enhanced image dehazing system that combines the strengths of traditional physics-based methods and modern learning-based approaches. The significance of this work lies in several aspects:

1. **Improved Visual Quality**: By enhancing the Dark Channel Prior method with guided filtering for transmission map refinement and adaptive contrast enhancement techniques, our approach achieves superior dehazing results with reduced artifacts and better preservation of natural colors and details. This improvement in visual quality directly benefits applications that rely on high-quality images, such as surveillance, remote sensing, and medical imaging.

2. **Complementary Approaches**: The integration of both physics-based and learning-based methods provides a comprehensive solution that can leverage the interpretability and theoretical foundation of traditional approaches while benefiting from the data-driven capabilities of deep learning models. This hybrid approach offers greater robustness across diverse scenes and haze conditions compared to either approach alone.

3. **Practical Applicability**: The development of a user-friendly graphical interface makes advanced dehazing techniques accessible to non-expert users, enabling practical applications in various domains including photography enhancement, surveillance system improvement, and pre-processing for other computer vision tasks. This bridges the gap between academic research and real-world deployment of dehazing technology.

4. **Video Processing Capability**: The extension of our dehazing methods to video content addresses the growing need for temporal processing in applications such as autonomous driving, drone surveillance, and outdoor monitoring systems. By ensuring temporal consistency in the dehazed video, our system provides a more complete solution for real-world scenarios where video is the primary data format.

5. **Educational Value**: The comprehensive analysis and comparison of different dehazing techniques provide valuable insights for researchers and practitioners in the field, contributing to a better understanding of the strengths and limitations of various approaches. This knowledge can guide future research and development efforts in image dehazing.

6. **Benchmark for Evaluation**: Our extensive experimental evaluation using both synthetic and real-world datasets establishes a benchmark for assessing the performance of dehazing algorithms. The combination of quantitative metrics and qualitative assessments provides a more comprehensive evaluation framework that can be adopted by the research community.

7. **Cross-Domain Applications**: The improved image quality resulting from our dehazing system can benefit a wide range of downstream applications beyond the immediate scope of dehazing. These include object detection and recognition, scene understanding, image segmentation, and 3D reconstruction, all of which rely on clear, high-quality images as input.

8. **Economic and Safety Impact**: By improving visibility in hazy conditions, our research has potential economic and safety implications. Enhanced visibility in transportation systems can reduce accidents and associated costs. Improved surveillance in adverse weather conditions can enhance security effectiveness. Better remote sensing imagery can lead to more accurate environmental monitoring and disaster response.

By addressing these aspects, this research contributes to advancing the state of the art in image dehazing and provides practical solutions for real-world applications affected by atmospheric degradation. The combination of theoretical advancements, algorithmic improvements, and practical implementation makes this work relevant to both the academic community and industry practitioners dealing with image quality issues in hazy conditions.

## 1.4 Thesis Structure

The remainder of this thesis is organized as follows:

**Chapter 2: Literature Review, Motivation, and Objective** provides a comprehensive review of existing image dehazing methods, including traditional approaches based on physical models and learning-based methods using deep neural networks. It also discusses evaluation metrics, challenges, and limitations in the field, leading to the motivation and specific objectives of this research.

The literature review begins with an explanation of the atmospheric scattering model that forms the theoretical foundation for most dehazing methods. It then categorizes and analyzes traditional dehazing approaches, including the Dark Channel Prior, Color Attenuation Prior, and Non-local Prior, highlighting their strengths and limitations. The review continues with an examination of learning-based methods, covering CNN-based approaches, GAN-based methods, and emerging transformer architectures.

The chapter also discusses evaluation methodologies, including full-reference metrics like PSNR and SSIM, no-reference metrics like NIQE and FADE, and the importance of human visual assessment. It identifies key challenges in the field, such as scene diversity, color distortion, computational efficiency, and the scarcity of training data. Based on this analysis, the chapter presents the motivation for this research and outlines specific objectives aimed at addressing the identified gaps and limitations.

**Chapter 3: Proposed System** presents the detailed design and implementation of our image dehazing system. It describes the enhanced Dark Channel Prior method with guided filtering and contrast enhancement, the U-Net-based deep learning approach, the video dehazing extension, and the graphical user interface development.

The chapter begins with an overview of the system architecture, explaining how different components interact to provide a comprehensive dehazing solution. It then delves into the technical details of the enhanced DCP method, including dark channel computation, atmospheric light estimation, transmission map refinement using guided filtering, scene radiance recovery, and adaptive contrast enhancement. The implementation details, including parameter selection and optimization strategies, are provided with supporting code snippets and illustrations.

The chapter continues with a description of the deep learning-based approach, detailing the U-Net architecture, dataset preparation, data augmentation techniques, training process, and model optimization. It explains how the video dehazing extension handles temporal consistency while maintaining computational efficiency. Finally, it presents the design principles and implementation details of the graphical user interface, focusing on user experience, visualization features, and interaction flow.

**Chapter 4: Results and Discussion** evaluates the performance of the proposed system through extensive experiments on both synthetic and real-world datasets. It includes qualitative and quantitative analyses, ablation studies to assess the contribution of individual components, and comparisons with state-of-the-art methods.

The chapter begins by describing the experimental setup, including hardware and software configurations, dataset characteristics, and evaluation methodology. It presents a qualitative analysis through visual comparisons and case studies, highlighting the performance of our methods in different scenarios such as extreme haze conditions and night-time scenes. The quantitative analysis includes performance metrics such as PSNR, SSIM, and computation time, with statistical analysis to establish the significance of the results.

The chapter includes ablation studies that investigate the impact of different parameters and the contribution of individual components to the overall performance. It provides a comprehensive comparison with state-of-the-art methods, analyzing the strengths and limitations of our approaches in relation to existing techniques. The chapter concludes with a user experience evaluation that assesses the practical usability of our system through surveys and feedback analysis.

**Chapter 5: Conclusion and Future Scope** summarizes the main contributions of this research, acknowledges its limitations, and suggests directions for future work to further advance image dehazing techniques and their applications.

The chapter begins with a concise summary of the key contributions, highlighting the improvements achieved in visual quality, computational efficiency, and practical applicability. It acknowledges the limitations of the current work, including challenges related to extreme conditions, parameter sensitivity, and the need for more diverse training data. The chapter then explores future research directions, including algorithmic improvements, application extensions, integration with other computer vision tasks, hardware acceleration, and multi-modal approaches. It concludes with final remarks on the significance of image dehazing in the broader context of computer vision and its potential impact on various applications.

The thesis concludes with a comprehensive list of references and appendices containing additional technical details, code implementations, supplementary experimental results, user manual, dataset details, and mathematical derivations. These appendices provide valuable resources for researchers and practitioners interested in implementing or extending our dehazing system.

# Chapter 2: Literature Review, Motivation, and Objective

## 2.1 Image Formation Model

To understand image dehazing algorithms, it is essential to first comprehend how haze affects images. The widely accepted atmospheric scattering model, proposed by Koschmieder (1924) and later refined by Narasimhan and Nayar (2002), describes the formation of a hazy image. According to this model, a hazy image I(x) at pixel position x can be expressed as:

I(x) = J(x)t(x) + A(1 - t(x))

where:
- J(x) is the scene radiance (haze-free image) that we aim to recover
- A is the global atmospheric light
- t(x) is the transmission map, representing the portion of light that reaches the camera without being scattered

The transmission t(x) is related to the scene depth d(x) by:

t(x) = e^(-βd(x))

where β is the atmospheric scattering coefficient.

This model indicates that a hazy image consists of two components: the attenuated scene radiance (J(x)t(x)) and the atmospheric light contribution (A(1 - t(x))). As the scene depth increases, the transmission decreases exponentially, causing distant objects to be more affected by haze.

The physical basis of this model lies in the interaction between light and atmospheric particles. When light travels through the atmosphere, it encounters various particles such as water droplets, dust, and pollutants. These particles cause two primary effects: attenuation and airlight.

Attenuation refers to the weakening of light from the scene to the camera due to absorption and scattering. This is represented by the term J(x)t(x) in the model. The transmission t(x) quantifies how much of the original scene radiance reaches the camera without being scattered away from the line of sight.

Airlight, represented by A(1 - t(x)), is the atmospheric light scattered towards the camera by the same particles. This scattered light adds a haze veil to the image, reducing contrast and saturation. The atmospheric light A is often assumed to be the color of the most haze-opaque region in the image, typically corresponding to the sky or distant objects completely obscured by haze.

The goal of image dehazing is to estimate the atmospheric light A and the transmission map t(x) from the hazy image I(x), and then recover the scene radiance J(x) using the inverse of the atmospheric scattering model:

J(x) = (I(x) - A) / max(t(x), t₀) + A

where t₀ is a small constant (typically 0.1) to prevent division by zero and to avoid noise amplification in regions with very low transmission.

However, this is an ill-posed problem since both A and t(x) are unknown, and there are infinitely many combinations of J, A, and t that can produce the same hazy image I. Therefore, additional priors or constraints are needed to make the problem solvable.

The atmospheric scattering model makes several assumptions that may not always hold in real-world scenarios:

1. **Homogeneous Atmosphere**: The model assumes that atmospheric conditions are uniform throughout the scene, which may not be true for scenes with non-uniform haze distribution.

2. **Single Scattering**: The model considers only first-order scattering and ignores multiple scattering events, which become significant in dense haze conditions.

3. **Constant Atmospheric Light**: The model assumes a global atmospheric light value, whereas in reality, the atmospheric light may vary spatially, especially in scenes with multiple light sources.

4. **Lambertian Surfaces**: The model assumes that scene surfaces are Lambertian (perfectly diffuse), which is not always the case for specular or transparent objects.

Despite these limitations, the atmospheric scattering model provides a solid theoretical foundation for most dehazing algorithms and has proven effective in practice for a wide range of scenes and conditions.

## 2.2 Traditional Dehazing Methods

### 2.2.1 Dark Channel Prior

The Dark Channel Prior (DCP), proposed by He et al. (2009), is one of the most influential traditional dehazing methods. It is based on the observation that in most non-sky regions of haze-free outdoor images, at least one color channel has very low intensity at some pixels. This observation leads to the definition of the dark channel of an image J as:

J^dark(x) = min_y∈Ω(x)(min_c∈{r,g,b}J^c(y))

where Ω(x) is a local patch centered at x, and J^c is a color channel of J.

The dark channel prior states that for haze-free outdoor images, the dark channel tends to have very low values (close to zero) except in sky regions. This prior is based on the observation that natural outdoor scenes contain abundant shadows, dark objects, or surfaces with low reflectance in at least one color channel.

The statistical basis for this prior was established through an extensive study of outdoor haze-free images. He et al. analyzed over 3,000 outdoor images and found that in more than 75% of the non-sky pixels, the dark channel value was below 25 (out of 255), confirming the validity of this prior for a wide range of natural scenes.

Based on this prior, He et al. derived a method to estimate the transmission map:

t(x) = 1 - ω min_y∈Ω(x)(min_c∈{r,g,b}(I^c(y)/A^c))

where ω (0 < ω ≤ 1) is a parameter to control the amount of haze removal, typically set to 0.95 to preserve a small amount of haze for distant objects, maintaining depth perception.

The atmospheric light A is estimated by finding the brightest pixels in the dark channel. Specifically, the method selects the top 0.1% brightest pixels in the dark channel and then picks the pixel with the highest intensity in the original image among these candidates.

The DCP algorithm can be summarized in the following steps:

1. Compute the dark channel of the hazy image
2. Estimate the atmospheric light A using the dark channel
3. Estimate the transmission map t(x) using the dark channel prior
4. Refine the transmission map using soft matting or guided filtering
5. Recover the scene radiance J(x) using the atmospheric scattering model

The DCP method has several advantages:

1. **Effectiveness**: It can effectively remove haze from a wide range of outdoor images without requiring multiple images or additional information.

2. **Simplicity**: The algorithm is relatively simple to implement and understand, making it accessible for various applications.

3. **Physical Basis**: It is grounded in the atmospheric scattering model, providing a physical interpretation of the dehazing process.

However, the DCP method also has several limitations:

1. **Sky Regions**: The dark channel prior does not hold for sky regions, leading to color distortions and over-enhancement in these areas.

2. **Halo Artifacts**: The patch-based operations can create halo artifacts around depth discontinuities, especially when using large patch sizes.

3. **Computational Complexity**: The original implementation using soft matting for transmission refinement is computationally expensive, although this has been addressed in later work using guided filtering.

4. **Parameter Sensitivity**: The performance of the algorithm is sensitive to the patch size and the omega parameter, requiring careful tuning for optimal results.

5. **Color Distortion**: In some cases, the method can produce color distortions, particularly in regions with bright colors or when the atmospheric light is incorrectly estimated.

Despite these limitations, the DCP method has been widely adopted and has inspired numerous improvements and extensions, making it a cornerstone in the field of image dehazing.

### 2.2.2 Color Attenuation Prior

Zhu et al. (2015) proposed the Color Attenuation Prior, which is based on the observation that the difference between the brightness and the saturation of pixels in a hazy image can be used to estimate the scene depth. This prior establishes a linear relationship between the scene depth d(x) and the difference between brightness v(x) and saturation s(x):

d(x) = θ_0 + θ_1 v(x) + θ_2 s(x)

where θ_0, θ_1, and θ_2 are parameters learned from a training dataset using a supervised learning method.

The color attenuation prior is based on the observation that haze typically reduces the saturation of colors while increasing their brightness. As the haze concentration increases with depth, there is a correlation between depth and the brightness-saturation difference. This correlation forms the basis for estimating the depth map, which can then be used to derive the transmission map.

The method involves the following steps:

1. Convert the hazy image from RGB to HSV color space to obtain the brightness (V) and saturation (S) channels.

2. Calculate the difference between brightness and saturation for each pixel.

3. Estimate the depth map using the linear model with pre-learned parameters.

4. Compute the transmission map using the exponential relationship between transmission and depth: t(x) = e^(-βd(x)).

5. Estimate the atmospheric light by finding the brightest pixels in the regions with the highest depth values.

6. Recover the scene radiance using the atmospheric scattering model.

The parameters θ_0, θ_1, and θ_2 are learned from a dataset of hazy images with known depth information. Zhu et al. created a synthetic dataset by applying the atmospheric scattering model to clear images with depth maps, and then used linear regression to learn the optimal parameters.

The Color Attenuation Prior method has several advantages:

1. **Computational Efficiency**: It is computationally efficient compared to the DCP method, especially when processing high-resolution images.

2. **No Patch-Based Operations**: It operates on individual pixels rather than patches, avoiding halo artifacts around depth discontinuities.

3. **Automatic Parameter Learning**: The parameters are learned from data, reducing the need for manual tuning.

However, the method also has limitations:

1. **Linear Model Assumption**: The assumption of a linear relationship between depth and color attributes may not hold for all scenes, especially those with complex lighting or unusual color distributions.

2. **Training Data Dependency**: The performance depends on the quality and diversity of the training data used to learn the parameters.

3. **Sensitivity to Color Variations**: The method may struggle with scenes that have inherent variations in brightness and saturation unrelated to haze, such as colorful objects or scenes with strong lighting contrasts.

Despite these limitations, the Color Attenuation Prior method provides a computationally efficient alternative to the DCP method and has been particularly effective for real-time applications where processing speed is critical.

### 2.2.3 Non-local Prior

Berman et al. (2016) introduced the Non-local Prior for image dehazing, which is based on the observation that colors in a haze-free image can be well approximated by a few hundred distinct colors. In the presence of haze, these colors form lines in the RGB space, called haze-lines, which converge to the atmospheric light.

The key insight of this method is that pixels in a haze-free image tend to cluster around a small number of distinct colors in the RGB space. When haze is added according to the atmospheric scattering model, each of these color clusters stretches into a line segment in the RGB space, with one end corresponding to the original color (no haze) and the other end approaching the atmospheric light (dense haze). These line segments, called haze-lines, all converge towards the atmospheric light.

The method involves the following steps:

1. Estimate the atmospheric light A by finding the brightest pixel in the hazy image after applying a median filter.

2. Identify the haze-lines in the RGB space by clustering the pixels of the hazy image. Each cluster represents pixels that likely had the same color in the haze-free image but appear different due to varying haze densities.

3. For each pixel, find the corresponding haze-line and estimate its position along the line, which indicates the transmission value for that pixel.

4. Regularize the transmission map to ensure spatial consistency while preserving edges.

5. Recover the scene radiance using the atmospheric scattering model.

The Non-local Prior method has several advantages:

1. **Global Approach**: It considers the global distribution of colors in the image rather than relying on local patches, which helps to avoid halo artifacts.

2. **Effective for Fine Structures**: It performs well on images with fine structures and details, which can be challenging for patch-based methods.

3. **Robustness to Non-uniform Haze**: The method can handle scenes with spatially varying haze distributions better than many other approaches.

However, the method also has limitations:

1. **Color Clustering Assumption**: The assumption that colors in a haze-free image can be approximated by a few hundred distinct colors may not hold for all scenes, especially those with smooth color gradients or complex textures.

2. **Computational Complexity**: The clustering and regularization steps can be computationally intensive, especially for high-resolution images.

3. **Sensitivity to Atmospheric Light Estimation**: The accuracy of the haze-line identification depends on the correct estimation of the atmospheric light.

The Non-local Prior method represents a significant departure from patch-based approaches like DCP, offering a global perspective on the dehazing problem. It has been particularly effective for scenes with complex structures and non-uniform haze distributions.

### 2.2.4 Contrast Enhancement-Based Methods

Contrast enhancement-based methods approach the dehazing problem from a different perspective, focusing on improving the visual quality of hazy images without explicitly modeling the atmospheric scattering process. These methods typically apply various image processing techniques to enhance the contrast, saturation, and details in hazy images.

One of the earliest contrast enhancement-based methods for dehazing is the Histogram Equalization (HE) technique, which redistributes the intensity values in an image to achieve a more uniform histogram. While simple HE can improve the global contrast of hazy images, it often leads to over-enhancement and unnatural appearances.

To address these limitations, more sophisticated variants have been developed:

1. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: This technique applies histogram equalization locally to small regions (tiles) of the image rather than the entire image. It also limits the contrast enhancement to prevent noise amplification and over-enhancement. CLAHE has been effective for dehazing when applied to the luminance channel of images in color spaces like LAB or YCbCr, preserving color information while enhancing contrast.

2. **Retinex-Based Methods**: Based on the Retinex theory of human color vision, these methods decompose an image into illumination and reflectance components. By enhancing the reflectance component and adjusting the illumination component, they can reduce the haze effect while preserving natural colors. Multi-scale Retinex with Color Restoration (MSRCR) has been particularly effective for dehazing.

3. **Unsharp Masking**: This technique enhances image details by adding a high-pass filtered version of the image to the original. When applied with appropriate parameters, it can enhance the visibility of details obscured by haze.

4. **Gamma Correction**: Adjusting the gamma value can help to enhance the contrast in specific tonal ranges of hazy images. Adaptive gamma correction techniques that adjust the gamma value based on local image statistics have shown promising results for dehazing.

5. **Fusion-Based Enhancement**: These methods generate multiple enhanced versions of the hazy image using different techniques or parameters, and then fuse them based on quality metrics such as contrast, saturation, and exposure.

Contrast enhancement-based methods have several advantages:

1. **Computational Efficiency**: Many of these techniques are computationally efficient and can be implemented in real-time on standard hardware.

2. **No Physical Model Requirement**: They do not rely on the atmospheric scattering model or its assumptions, making them more flexible for various haze conditions.

3. **Simplicity**: These methods are often simpler to implement and understand compared to physics-based or learning-based approaches.

However, they also have significant limitations:

1. **Lack of Physical Basis**: Without modeling the physical process of haze formation, these methods may produce visually pleasing but physically implausible results.

2. **Over-Enhancement**: Aggressive contrast enhancement can lead to over-saturation, noise amplification, and unnatural appearances.

3. **Depth-Dependent Haze**: These methods typically do not account for the depth-dependent nature of haze, potentially leading to inconsistent dehazing across different depths.

4. **Color Distortion**: Enhancing contrast without considering the color shifts caused by haze can result in color distortions in the processed images.

Despite these limitations, contrast enhancement-based methods remain popular for applications where computational efficiency and simplicity are prioritized over physical accuracy, such as real-time video processing or mobile applications.

### 2.2.5 Fusion-Based Methods

Fusion-based dehazing methods combine multiple versions of the input image or multiple dehazing techniques to produce a final dehazed result. These methods leverage the strengths of different approaches while mitigating their individual weaknesses, often leading to more robust and visually pleasing results.

Ancuti et al. (2013) proposed one of the most influential fusion-based dehazing methods, which derives two versions of the input image—a white-balanced version and a contrast-enhanced version—and fuses them using a multi-scale fusion approach. The fusion process is guided by weight maps that measure the quality of each input in terms of contrast, saturation, and exposure.

The method involves the following steps:

1. Generate multiple inputs from the original hazy image:
   - A white-balanced version that corrects the color cast caused by haze
   - A contrast-enhanced version that improves the visibility of details

2. Compute weight maps for each input based on quality metrics:
   - Contrast weight: Measures the amount of local contrast using a Laplacian filter
   - Saturation weight: Measures the standard deviation of the color channels
   - Exposure weight: Measures how well-exposed each pixel is

3. Normalize the weight maps to ensure they sum to one at each pixel.

4. Perform multi-scale fusion using a Laplacian pyramid for the inputs and a Gaussian pyramid for the weight maps, which helps to avoid artifacts at the boundaries of regions with different weights.

5. Reconstruct the final dehazed image from the fused pyramid.

Other fusion-based approaches include:

1. **Hybrid Methods**: These combine physics-based and contrast enhancement-based techniques. For example, some methods use the DCP to estimate the transmission map and then apply contrast enhancement to the dehazed result to improve its visual quality.

2. **Multi-Algorithm Fusion**: These methods apply multiple dehazing algorithms to the same input image and fuse the results based on quality metrics or learning-based fusion strategies.

3. **Multi-Exposure Fusion**: Similar to HDR imaging, these methods capture or simulate multiple exposures of the hazy scene and fuse them to recover details in both dark and bright regions.

4. **Guided Fusion**: These methods use additional information, such as depth maps or semantic segmentation, to guide the fusion process, giving more weight to more reliable inputs for different regions of the image.

Fusion-based methods have several advantages:

1. **Robustness**: By combining multiple inputs or techniques, fusion-based methods can be more robust to variations in haze density, scene content, and lighting conditions.

2. **Balanced Results**: They often achieve a good balance between haze removal, detail preservation, and natural appearance.

3. **Adaptability**: The fusion weights can adapt to local image characteristics, providing appropriate processing for different regions of the image.

However, they also have limitations:

1. **Computational Complexity**: Generating multiple inputs and performing multi-scale fusion can be computationally intensive, especially for high-resolution images.

2. **Parameter Sensitivity**: The quality of the results depends on the parameters used to generate the inputs and compute the weight maps.

3. **Potential Artifacts**: If the inputs or weight maps have significant differences, the fusion process may introduce artifacts or inconsistencies in the final result.

Despite these limitations, fusion-based methods have shown promising results for a wide range of hazy scenes and have been particularly effective for challenging conditions where single-algorithm approaches struggle.

## 2.3 Learning-Based Dehazing Methods

### 2.3.1 CNN-Based Methods

With the advancement of deep learning, Convolutional Neural Networks (CNNs) have been widely applied to image dehazing. These methods can be categorized into two main approaches:

1. **Prior-based CNN methods**: These methods use CNNs to estimate the components of the atmospheric scattering model, such as the transmission map or atmospheric light. For example, DehazeNet (Cai et al., 2016) and MSCNN (Ren et al., 2016) use CNNs to estimate the transmission map, which is then used to recover the clear image using the atmospheric scattering model.

2. **End-to-end CNN methods**: These methods directly learn the mapping from hazy images to clear images without explicitly modeling the atmospheric scattering process. AOD-Net (Li et al., 2017) reformulates the atmospheric scattering model and uses a lightweight CNN to directly estimate the clean image. DCPDN (Zhang and Patel, 2018) employs a generative adversarial network framework with an embedded atmospheric scattering model.

DehazeNet, proposed by Cai et al. (2016), was one of the pioneering CNN-based dehazing methods. It uses a CNN to estimate the transmission map from a hazy image, inspired by the dark channel prior but learning the mapping directly from data. The network consists of four sequential operations: feature extraction, multi-scale mapping, local extremum, and non-linear regression. Once the transmission map is estimated, the atmospheric light is computed using a traditional method, and the clear image is recovered using the atmospheric scattering model.

MSCNN, proposed by Ren et al. (2016), uses a multi-scale CNN architecture to estimate the transmission map. It consists of a coarse-scale network that predicts a holistic transmission map and a fine-scale network that refines the details. This multi-scale approach helps to capture both global haze distribution and local details, leading to more accurate transmission estimation.

AOD-Net, proposed by Li et al. (2017), takes a different approach by reformulating the atmospheric scattering model to reduce the number of unknown variables. Instead of separately estimating the transmission map and atmospheric light, it directly estimates a unified parameter that combines both, simplifying the dehazing process. The network is lightweight and end-to-end trainable, making it suitable for real-time applications.

DCPDN, proposed by Zhang and Patel (2018), uses a generative adversarial network (GAN) framework with an embedded atmospheric scattering model. It consists of three components: a transmission map estimation network, an atmospheric light estimation network, and a dehazed image generation network. The adversarial training helps to produce more realistic and visually pleasing results.

More recent CNN-based methods have explored various architectural innovations:

1. **Attention Mechanisms**: Methods like PFDN (Dong et al., 2020) incorporate attention mechanisms to focus on informative regions and features, improving the dehazing performance for challenging scenes.

2. **Dense Connections**: Networks like GridDehazeNet (Liu et al., 2019) use dense connections to facilitate feature reuse and gradient flow, leading to better training dynamics and performance.

3. **Multi-Task Learning**: Some methods jointly learn dehazing and related tasks such as depth estimation or semantic segmentation, leveraging the complementary information to improve overall performance.

4. **Progressive Refinement**: Methods like PDNet (Chen et al., 2021) use a progressive refinement approach, starting with a coarse dehazed image and iteratively refining it to produce the final result.

CNN-based methods have several advantages:

1. **Learning from Data**: They can learn complex mappings from hazy to clear images directly from data, without relying on handcrafted priors or assumptions.

2. **End-to-End Training**: Many CNN-based methods can be trained end-to-end, optimizing all components jointly for the dehazing task.

3. **Adaptability**: With sufficient training data, CNN-based methods can adapt to various haze conditions and scene types.

However, they also have limitations:

1. **Data Dependency**: They require large amounts of training data, ideally paired hazy and clear images, which can be difficult to obtain for real-world scenarios.

2. **Generalization**: Models trained on synthetic data may struggle to generalize to real-world haze conditions due to the domain gap.

3. **Computational Resources**: Training deep CNN models requires significant computational resources, and inference can be slow without GPU acceleration.

4. **Interpretability**: Unlike physics-based methods, CNN-based approaches often lack interpretability, making it difficult to understand and debug their behavior.

Despite these limitations, CNN-based methods have achieved state-of-the-art performance on benchmark datasets and continue to be an active area of research in image dehazing.

### 2.3.2 GAN-Based Methods

Generative Adversarial Networks (GANs) have been employed for image dehazing to improve the visual quality of dehazed images. GAN-based methods typically consist of a generator network that produces dehazed images and a discriminator network that distinguishes between real clear images and generated dehazed images.

The adversarial training process encourages the generator to produce dehazed images that are indistinguishable from real clear images, leading to more realistic and visually pleasing results. This is particularly valuable for image dehazing, where the goal is not just to remove haze but also to produce natural-looking images.

CycleDehaze, proposed by Engin et al. (2018), uses a cycle-consistent adversarial network to learn the mapping between hazy and clear images without requiring paired training data. It consists of two generators: one that maps hazy images to clear images and another that maps clear images to hazy images. The cycle consistency loss ensures that an image translated from one domain to the other and back should be identical to the original image. This approach is valuable when paired training data is scarce, as it can leverage unpaired datasets of hazy and clear images.

EPDN (Enhanced Perceptual Dehazing Network), proposed by Qu et al. (2019), enhances the perceptual quality of dehazed images by incorporating perceptual loss functions in the GAN framework. It uses a multi-scale generator with dense connections and a discriminator that operates on multiple scales. The perceptual loss, based on features extracted from a pre-trained VGG network, helps to preserve semantic information and fine details in the dehazed images.

Other notable GAN-based dehazing methods include:

1. **DHSGAN (Dehazing and Super-Resolution GAN)**, proposed by Li et al. (2020), which jointly addresses dehazing and super-resolution using a dual-task GAN framework. It leverages the complementary nature of these tasks, as both aim to recover high-frequency details lost due to degradation.

2. **YOLY (You Only Look Yourself)**, proposed by Li et al. (2021), which uses a self-supervised GAN approach that does not require clear reference images for training. It leverages the observation that haze is typically depth-dependent, so nearby regions should have less haze than distant regions in the same scene.

3. **DualGAN**, proposed by Yang et al. (2018), which uses two GANs to learn bidirectional mappings between hazy and clear domains, similar to CycleDehaze but with different architectural choices and loss functions.

4. **PGAN (Progressive GAN)**, proposed by Zhang et al. (2019), which adopts a progressive training strategy, starting with low-resolution images and gradually increasing the resolution. This approach stabilizes the training process and helps to capture both global haze distribution and fine details.

GAN-based methods have several advantages for image dehazing:

1. **Perceptual Quality**: The adversarial training encourages the generator to produce visually pleasing results that look natural to human observers, often outperforming other methods in terms of perceptual quality.

2. **Unpaired Training**: Some GAN-based methods can be trained with unpaired data, addressing the scarcity of paired hazy and clear images in real-world scenarios.

3. **Detail Preservation**: The discriminator's ability to distinguish between real and fake images encourages the generator to preserve fine details and textures that might be lost with other methods.

4. **Adaptability**: With appropriate training data, GAN-based methods can adapt to various haze conditions and scene types.

However, they also have limitations:

1. **Training Instability**: GANs are notoriously difficult to train, often suffering from issues like mode collapse, vanishing gradients, or training instability.

2. **Artifacts**: GAN-based methods may introduce artifacts or hallucinate details that were not present in the original scene, potentially compromising the fidelity of the dehazed image.

3. **Computational Resources**: Training GAN models requires significant computational resources and time, making them less accessible for researchers with limited resources.

4. **Evaluation Challenges**: Traditional metrics like PSNR and SSIM may not fully capture the perceptual quality improvements achieved by GAN-based methods, making evaluation and comparison challenging.

Despite these limitations, GAN-based methods have shown promising results for image dehazing, particularly in terms of perceptual quality and natural appearance. They represent an important direction in learning-based dehazing research, complementing CNN-based approaches with their focus on realistic image generation.

### 2.3.3 Transformer-Based Methods

More recently, Transformer architectures, which have shown remarkable success in natural language processing, have been adapted for image dehazing. Transformer-based methods leverage the self-attention mechanism to capture long-range dependencies in images, which is beneficial for understanding the global context of haze distribution.

The self-attention mechanism allows each pixel to attend to all other pixels in the image, enabling the model to capture relationships between distant regions. This is particularly valuable for image dehazing, where the haze distribution and scene content can vary significantly across the image.

DehazeFormer, proposed by Song et al. (2022), combines CNN features with Transformer blocks to effectively capture both local and global features for image dehazing. It uses a hierarchical structure with multiple stages, each consisting of a patch embedding layer, Transformer blocks, and a patch expanding layer. The patch embedding layer extracts features using convolutional operations, while the Transformer blocks capture long-range dependencies using self-attention. The patch expanding layer upsamples the features for the next stage or final output.

FocalNet, proposed by Yang et al. (2022), uses a focal self-attention mechanism to focus on informative regions while reducing computational complexity. Unlike standard self-attention that attends to all pixels equally, focal self-attention adaptively adjusts the attention weights based on the importance of different regions. This allows the model to allocate more computational resources to regions that are more challenging for dehazing, such as areas with dense haze or complex textures.

Other notable Transformer-based dehazing methods include:

1. **TransWeather**, proposed by Valanarasu et al. (2022), which uses a Transformer-based architecture for various weather removal tasks, including dehazing, deraining, and desnowing. It leverages a multi-head self-attention mechanism to capture global context and a convolutional branch to capture local details.

2. **HIT (Haze Image Transformer)**, proposed by Chen et al. (2022), which uses a hybrid architecture combining Transformers for global context modeling and CNNs for local feature extraction. It also incorporates a physics-guided module that leverages the atmospheric scattering model to guide the learning process.

3. **DFormer**, proposed by Wang et al. (2023), which uses a dual-branch Transformer architecture with one branch focusing on haze removal and the other on detail enhancement. The two branches share information through cross-attention mechanisms, allowing them to complement each other.

4. **SwinDehazeNet**, proposed by Liu et al. (2023), which adapts the Swin Transformer architecture for image dehazing. It uses shifted windows for self-attention computation, reducing computational complexity while maintaining the ability to capture long-range dependencies.

Transformer-based methods have several advantages for image dehazing:

1. **Global Context Modeling**: The self-attention mechanism allows these methods to capture long-range dependencies and global context, which is valuable for understanding the haze distribution across the entire image.

2. **Adaptive Processing**: Transformer-based methods can adaptively process different regions of the image based on their content and haze density, potentially leading to more effective dehazing.

3. **Scalability**: Transformer architectures can be scaled to handle high-resolution images and complex scenes, making them suitable for challenging dehazing scenarios.

4. **Transfer Learning**: Pre-trained Transformer models from other vision tasks can be fine-tuned for dehazing, leveraging knowledge learned from large-scale datasets.

However, they also have limitations:

1. **Computational Complexity**: The standard self-attention mechanism has quadratic complexity with respect to the number of pixels, making it computationally expensive for high-resolution images. Various approximations and optimizations have been proposed to address this issue.

2. **Data Requirements**: Transformer-based methods typically require large amounts of training data to achieve optimal performance, which can be challenging for dehazing due to the scarcity of paire