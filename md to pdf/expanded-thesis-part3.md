d hazy and clear images.

3. **Model Size**: Transformer models typically have more parameters than CNN-based models, requiring more memory for training and inference.

4. **Lack of Inductive Bias**: Unlike CNNs, which have built-in inductive biases for processing images (e.g., translation equivariance), Transformers rely more heavily on learning from data, which can be a challenge with limited training samples.

Despite these limitations, Transformer-based methods have demonstrated state-of-the-art performance on benchmark datasets and represent a promising direction for future research in image dehazing. Their ability to capture global context and long-range dependencies makes them particularly suitable for complex scenes with varying haze distributions.

### 2.3.4 Physics-Guided Learning Methods

Physics-guided learning methods combine the strengths of traditional physics-based approaches and modern learning-based techniques. These methods incorporate the atmospheric scattering model or other physical principles into the neural network architecture or loss function, providing a stronger inductive bias and improving the interpretability of the model.

The key insight behind physics-guided learning is that while deep learning models excel at learning complex mappings from data, they can benefit from incorporating domain knowledge about the physical process of haze formation. This can lead to more efficient learning, better generalization to unseen data, and more physically plausible results.

PGAN (Physics-Guided Adversarial Network), proposed by Li et al. (2020), incorporates the atmospheric scattering model into a GAN framework. The generator is designed to estimate the transmission map and atmospheric light, which are then used to recover the clear image using the atmospheric scattering model. The discriminator evaluates both the dehazed image and the estimated physical parameters, encouraging physically plausible results.

PHYD-Net (Physics-based Hybrid Dehazing Network), proposed by Chen et al. (2021), uses a hybrid architecture that combines a physics-based branch and a learning-based branch. The physics-based branch estimates the transmission map and atmospheric light using traditional methods, while the learning-based branch directly predicts a residual image to refine the physics-based result. The two branches complement each other, with the physics-based branch providing a physically plausible initial estimate and the learning-based branch addressing its limitations.

Other notable physics-guided learning methods include:

1. **PMHLD (Physical Model-Guided Hybrid Learning Dehazing)**, proposed by Zhao et al. (2021), which uses a multi-stage approach that first estimates the transmission map and atmospheric light using a CNN, then refines the transmission map using a guided filter, and finally recovers the clear image using the atmospheric scattering model.

2. **KDPN (Knowledge-Driven Perception Network)**, proposed by Yang et al. (2020), which incorporates physical priors such as the dark channel prior and color attenuation prior into the network architecture through specialized modules that mimic the behavior of these priors.

3. **PAD-Net (Physics-Aware Dehazing Network)**, proposed by Li et al. (2022), which uses a physics-aware loss function that enforces consistency with the atmospheric scattering model, in addition to the standard reconstruction loss.

4. **PSDL (Physics-Supervised Deep Learning)**, proposed by Wang et al. (2021), which uses the atmospheric scattering model to generate synthetic training data with known physical parameters, allowing for supervised learning of these parameters even when ground truth clear images are not available.

Physics-guided learning methods have several advantages:

1. **Interpretability**: By incorporating physical models, these methods provide more interpretable results, allowing users to understand the estimated physical parameters such as transmission and atmospheric light.

2. **Data Efficiency**: The physical constraints can serve as a strong regularization, reducing the amount of training data needed to achieve good performance.

3. **Generalization**: The physical priors can help the model generalize better to unseen data, especially when the test conditions differ significantly from the training data.

4. **Physical Plausibility**: The results are more likely to be physically plausible, avoiding artifacts or unrealistic appearances that might occur with purely data-driven approaches.

However, they also have limitations:

1. **Model Complexity**: Incorporating physical models can increase the complexity of the network architecture and training process.

2. **Simplified Physics**: The atmospheric scattering model used in these methods is often simplified and may not capture all the complexities of real-world haze formation.

3. **Balance of Physics and Learning**: Finding the right balance between physical constraints and learning from data can be challenging, as overly strict physical constraints might limit the model's ability to handle cases where the physical model is inaccurate.

Physics-guided learning represents a promising direction for image dehazing research, combining the interpretability and theoretical foundation of physics-based methods with the data-driven capabilities of learning-based approaches.

### 2.3.5 Unsupervised and Self-Supervised Methods

Unsupervised and self-supervised learning methods for image dehazing address the challenge of limited paired training data by learning from unpaired or single images. These methods leverage various forms of supervision signals derived from the data itself, without requiring explicit ground truth clear images.

Unsupervised methods typically use cycle consistency, domain adaptation, or other techniques to learn the mapping between hazy and clear domains without paired examples. Self-supervised methods, on the other hand, create supervisory signals from the input data itself, such as by synthesizing hazy images from clear ones using the atmospheric scattering model or by exploiting the structure of natural images.

CycleGAN-based methods, such as CycleDehaze mentioned earlier, use cycle consistency to learn bidirectional mappings between hazy and clear domains. The key idea is that if an image is translated from the hazy domain to the clear domain and back, it should be identical to the original hazy image, and vice versa. This constraint allows the model to learn meaningful mappings without paired data.

Domain adaptation methods, such as DADA (Domain Adaptation for Dehazing Algorithm) proposed by Shao et al. (2020), address the domain gap between synthetic and real haze by learning domain-invariant features. These methods typically use adversarial training to align the feature distributions of synthetic and real data, allowing models trained on synthetic data to generalize better to real-world haze.

Self-supervised methods include:

1. **Zero-DCE (Zero-Reference Deep Curve Estimation)**, proposed by Guo et al. (2020), which formulates dehazing as an image-specific curve estimation problem. It learns to estimate pixel-wise curves that map the hazy image to a dehazed version, using non-reference loss functions based on color constancy, exposure control, and illumination smoothness.

2. **SSID (Self-Supervised Image Dehazing)**, proposed by Yang et al. (2021), which uses the atmospheric scattering model to synthesize hazy images from the input image itself, creating pseudo pairs for training. It leverages the observation that haze is typically depth-dependent, so nearby regions should have less haze than distant regions.

3. **YOLY (You Only Look Yourself)**, mentioned earlier, which uses a self-supervised GAN approach that exploits the depth-dependent nature of haze to create supervisory signals from a single hazy image.

4. **CDD (Contrastive Disentangled Dehazing)**, proposed by Chen et al. (2022), which uses contrastive learning to disentangle content and haze features, allowing for more effective dehazing without paired supervision.

Unsupervised and self-supervised methods have several advantages:

1. **Data Efficiency**: They can learn from unpaired or single images, addressing the scarcity of paired hazy and clear images in real-world scenarios.

2. **Domain Adaptation**: They can better bridge the domain gap between synthetic and real haze, improving generalization to real-world conditions.

3. **Scalability**: They can potentially leverage large amounts of unpaired data, which is more readily available than paired data.

However, they also have limitations:

1. **Training Complexity**: Unsupervised and self-supervised training can be more complex and less stable than supervised training, requiring careful design of loss functions and training strategies.

2. **Performance Gap**: Without direct supervision from ground truth clear images, these methods may not achieve the same level of performance as supervised methods when paired data is available.

3. **Evaluation Challenges**: Evaluating these methods can be challenging, as traditional metrics like PSNR and SSIM require ground truth references, which may not be available for real-world hazy images.

Despite these limitations, unsupervised and self-supervised methods represent an important direction for image dehazing research, particularly for real-world applications where paired training data is scarce or unavailable.

## 2.4 Evaluation Metrics

### 2.4.1 Full-Reference Metrics

Full-reference metrics evaluate the quality of dehazed images by comparing them with ground truth clear images. These metrics are widely used in the literature to quantitatively assess the performance of dehazing algorithms on synthetic datasets where paired hazy and clear images are available.

**Peak Signal-to-Noise Ratio (PSNR)** is one of the most commonly used full-reference metrics. It measures the pixel-wise difference between the dehazed image and the ground truth, expressed in decibels (dB). Higher PSNR values indicate better quality, with typical values for good dehazing results ranging from 20 to 30 dB. PSNR is defined as:

PSNR = 10 * log₁₀(MAX²/MSE)

where MAX is the maximum possible pixel value (typically 255 for 8-bit images) and MSE is the mean squared error between the dehazed image and the ground truth.

While PSNR is widely used due to its simplicity and ease of computation, it has limitations. It assumes that image quality is primarily determined by pixel-wise differences, which may not align with human perception. For example, a slight shift in the image can result in a significant drop in PSNR, even if the visual quality is largely unchanged.

**Structural Similarity Index Measure (SSIM)** addresses some of the limitations of PSNR by considering structural information. It evaluates the similarity between two images based on luminance, contrast, and structure, with values ranging from -1 to 1 (higher is better). SSIM is more aligned with human perception than PSNR and is defined as:

SSIM(x, y) = [l(x, y)]^α · [c(x, y)]^β · [s(x, y)]^γ

where l, c, and s represent the luminance, contrast, and structure comparison functions, respectively, and α, β, and γ are positive constants that adjust the relative importance of the three components.

SSIM is more sensitive to structural changes and less sensitive to uniform changes in brightness or contrast, which aligns better with human perception. However, it still has limitations, particularly for images with complex textures or when the dehazed image has different color characteristics than the ground truth.

**Color Difference (CIEDE2000)** is a metric specifically designed to measure perceptual color differences. It operates in the CIELAB color space, which is designed to be perceptually uniform, meaning that a given change in color value should produce a similar perceived change regardless of where in the color space the change occurs. CIEDE2000 takes into account variations in human color perception across the color spectrum and is particularly useful for evaluating color fidelity in dehazed images.

Other full-reference metrics used in dehazing evaluation include:

1. **Feature Similarity Index (FSIM)**, which considers phase congruency and gradient magnitude features to better capture the perceptual quality of images.

2. **Visual Information Fidelity (VIF)**, which quantifies the mutual information between the reference and distorted images in various wavelet subbands.

3. **Mean Absolute Error (MAE)**, which measures the average absolute difference between corresponding pixels in the dehazed and ground truth images.

4. **Learned Perceptual Image Patch Similarity (LPIPS)**, which uses features from pre-trained deep neural networks to measure perceptual similarity, often aligning better with human judgments than traditional metrics.

Full-reference metrics provide a quantitative basis for comparing different dehazing algorithms, but they have several limitations:

1. **Requirement for Ground Truth**: They require ground truth clear images, which are often unavailable for real-world hazy images.

2. **Alignment Sensitivity**: Many metrics are sensitive to slight misalignments between the dehazed and ground truth images, which can occur due to geometric transformations during the dehazing process.

3. **Perceptual Correlation**: The correlation between these metrics and human perception is not perfect, especially for complex scenes or when different types of distortions are present.

4. **Context Insensitivity**: Most metrics do not consider the semantic importance of different regions in the image, treating all pixels equally regardless of their content.

Despite these limitations, full-reference metrics remain valuable tools for quantitative evaluation when ground truth images are available, particularly when used in combination to capture different aspects of image quality.

### 2.4.2 No-Reference Metrics

No-reference metrics evaluate the quality of dehazed images without requiring ground truth clear images. These metrics are particularly valuable for assessing dehazing performance on real-world hazy images where ground truth is unavailable.

**Natural Image Quality Evaluator (NIQE)** is a no-reference metric that measures the naturalness of an image based on statistical features. It uses a quality-aware collection of statistical features based on a simple and successful space domain natural scene statistic model. NIQE is trained on a corpus of natural images and assigns lower scores to images that deviate from the statistical properties of natural images. Lower NIQE scores indicate better perceptual quality.

NIQE has the advantage of not requiring any training on human-rated distorted images, making it more general and less biased towards specific types of distortions. However, it may not always align with human perception, especially for images with artistic or stylistic elements that deviate from natural statistics.

**Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)** is another no-reference metric that operates in the spatial domain. It uses scene statistics of locally normalized luminance coefficients to quantify possible losses of naturalness in the image due to the presence of distortions. BRISQUE is trained on a database of images with known distortions and human quality ratings, allowing it to predict the quality of new images. Like NIQE, lower BRISQUE scores indicate better quality.

BRISQUE has shown good correlation with human perception for various distortions, including those introduced by dehazing algorithms. However, it is trained on specific types of distortions and may not generalize well to novel distortions or artistic effects.

**Fog Aware Density Evaluator (FADE)** is a specialized no-reference metric designed specifically for assessing the presence of fog or haze in images. It uses a regression model trained on a database of foggy images with human-rated fog density scores. FADE provides a fog density score, with lower values indicating less fog or haze. This metric is particularly useful for evaluating the effectiveness of dehazing algorithms in terms of haze removal, rather than overall image quality.

Other no-reference metrics used in dehazing evaluation include:

1. **Perception-based Image Quality Evaluator (PIQE)**, which estimates quality based on local spatial statistics of an image without requiring training on human-rated images.

2. **Blind Image Quality Index (BIQI)**, which uses a two-step framework that first identifies the distortion type and then applies distortion-specific quality assessment.

3. **Contrast-to-Noise Ratio (CNR)**, which measures the contrast between regions of interest and background noise, providing an indication of visibility enhancement.

4. **Entropy**, which measures the amount of information or randomness in an image, with higher entropy often indicating more detailed and less hazy images.

5. **Edge Preservation Index (EPI)**, which evaluates how well edges are preserved in the dehazed image compared to the hazy input.

No-reference metrics have several advantages for dehazing evaluation:

1. **Applicability to Real-World Images**: They can be applied to real-world hazy images where ground truth clear images are unavailable.

2. **Direct Assessment**: They directly assess the quality of the dehazed image without requiring comparison with a reference.

3. **Specialized Evaluation**: Some metrics, like FADE, are specifically designed for haze-related quality assessment.

However, they also have limitations:

1. **Limited Correlation with Human Perception**: The correlation between no-reference metrics and human perception can be lower than that of full-reference metrics, especially for complex scenes or novel distortions.

2. **Training Bias**: Metrics that rely on training data may be biased towards the types of images and distortions represented in the training set.

3. **Lack of Consensus**: There is less consensus on which no-reference metrics best capture dehazing quality, leading to inconsistent evaluation across different studies.

4. **Context Insensitivity**: Like full-reference metrics, most no-reference metrics do not consider the semantic importance of different regions in the image.

Despite these limitations, no-reference metrics provide valuable tools for evaluating dehazing performance on real-world images and complement full-reference metrics when ground truth is available.

### 2.4.3 Human Visual Perception

Human visual perception plays a crucial role in evaluating the quality of dehazed images, as the ultimate goal of image dehazing is to improve the visual experience for human observers. Subjective evaluation through human assessment provides insights that may not be captured by objective metrics, particularly regarding the naturalness, pleasantness, and overall visual appeal of dehazed images.

Subjective evaluation typically involves presenting dehazed images to human observers and collecting their ratings or preferences. Several methodologies are commonly used:

1. **Mean Opinion Score (MOS)**: Observers rate the quality of dehazed images on a scale (e.g., 1-5 or 1-10), and the average rating is computed as the MOS. This approach provides a direct measure of perceived quality but can be influenced by individual biases and the specific instructions given to observers.

2. **Paired Comparison**: Observers are presented with pairs of images (e.g., results from different dehazing algorithms) and asked to select the one they prefer. This approach is more intuitive for observers and can provide more reliable results, especially when the quality differences are subtle.

3. **Rank Ordering**: Observers rank multiple dehazed images from best to worst according to specific criteria such as haze removal effectiveness, color fidelity, or overall quality. This approach provides a relative assessment of different algorithms but may become challenging with a large number of images.

4. **Expert Assessment**: Domain experts, such as professional photographers or image processing specialists, evaluate the dehazed images based on their expertise. This approach can provide more informed judgments but may be less representative of the general population.

The criteria used for human assessment of dehazed images typically include:

1. **Haze Removal Effectiveness**: How well the algorithm removes haze and improves visibility, especially in distant regions of the scene.

2. **Color Fidelity**: Whether the colors in the dehazed image appear natural and consistent with expectations for the scene.

3. **Detail Preservation**: How well fine details and textures are preserved or recovered in the dehazed image.

4. **Artifact Presence**: Whether the dehazing process introduces artifacts such as color distortions, halos, or noise.

5. **Overall Visual Quality**: The general aesthetic appeal and pleasantness of the dehazed image.

Human visual perception is influenced by various factors that should be considered in subjective evaluation:

1. **Context and Expectations**: The perceived quality of a dehazed image can be influenced by the context in which it is viewed and the observer's expectations. For example, observers may have different expectations for artistic photographs versus surveillance footage.

2. **Viewing Conditions**: Factors such as display calibration, ambient lighting, viewing distance, and viewing angle can affect the perceived quality of dehazed images.

3. **Individual Differences**: Observers may have different preferences, sensitivities, and visual acuity, leading to variations in their assessments.

4. **Familiarity with the Scene**: If observers are familiar with the scene depicted in the image, they may have prior knowledge about how it should appear without haze, influencing their assessment.

5. **Adaptation**: The human visual system adapts to viewing conditions, including haze. Observers may mentally compensate for haze in the original image, affecting their perception of the dehazed result.

To address these factors and ensure reliable subjective evaluation, several best practices are recommended:

1. **Standardized Viewing Conditions**: Conduct evaluations under controlled and consistent viewing conditions, including display calibration, ambient lighting, and viewing distance.

2. **Sufficient Sample Size**: Include a sufficient number of observers to account for individual variations and obtain statistically significant results.

3. **Diverse Image Set**: Use a diverse set of images representing different scenes, haze conditions, and challenging cases to comprehensively evaluate dehazing performance.

4. **Blind Evaluation**: Present images without revealing which algorithm produced them to avoid bias based on expectations or prior knowledge.

5. **Clear Instructions**: Provide clear and consistent instructions to observers about the evaluation criteria and rating scale.

6. **Statistical Analysis**: Apply appropriate statistical methods to analyze the results, including measures of central tendency, variability, and statistical significance.

While human visual assessment provides valuable insights, it also has limitations:

1. **Subjectivity**: Human judgments are inherently subjective and can vary based on individual preferences and biases.

2. **Resource Intensity**: Conducting large-scale human studies is time-consuming, expensive, and logistically challenging.

3. **Reproducibility**: Subjective evaluations may be difficult to reproduce exactly due to variations in observers, instructions, and viewing conditions.

4. **Limited Scope**: Practical constraints often limit the number of images and algorithms that can be evaluated through human assessment.

Despite these limitations, human visual perception remains the ultimate benchmark for image quality assessment, and subjective evaluation provides essential insights that complement objective metrics. A comprehensive evaluation of dehazing algorithms should ideally combine both objective metrics and human assessment to capture different aspects of performance.

## 2.5 Challenges and Limitations

### 2.5.1 Scene Diversity and Complexity

One of the major challenges in image dehazing is handling the diversity and complexity of real-world scenes. Different scenes can have vastly different characteristics in terms of content, lighting, colors, and textures, making it difficult to develop dehazing algorithms that perform well across all scenarios.

Sky regions pose a particular challenge for many dehazing algorithms, especially those based on the dark channel prior. The dark channel prior assumes that at least one color channel has very low intensity in most non-sky regions of haze-free images. However, this assumption does not hold for sky regions, which are naturally bright and have high intensity in all color channels. As a result, algorithms based on this prior often overestimate the haze in sky regions, leading to color distortions and over-enhancement.

Night-time scenes present another challenging scenario for dehazing algorithms. Most dehazing methods are designed for daylight conditions and may not perform well in low-light environments. Night-time haze has different optical properties compared to daylight haze, with artificial light sources creating complex illumination patterns and glare effects. Additionally, low-light conditions often introduce noise, which can be amplified by dehazing algorithms, further degrading the image quality.

Scenes with complex depth structures, such as urban environments with buildings at various distances, present challenges for algorithms that assume a smooth depth distribution. Depth discontinuities can lead to halo artifacts around object boundaries, especially for patch-based methods that use a fixed patch size. These artifacts appear as unnatural bright or dark regions around edges where there is a significant depth change.

Water bodies, reflective surfaces, and transparent objects also pose challenges for dehazing algorithms. These elements have optical properties that differ from the assumptions made by most dehazing methods. For example, water surfaces can reflect light and create specular highlights, while transparent objects like glass can transmit light from behind them. These phenomena are not well captured by the standard atmospheric scattering model used in most dehazing algorithms.

Scenes with non-uniform haze distribution, such as those with fog patches or varying haze density across the image, challenge algorithms that assume a uniform atmospheric scattering coefficient. In reality, haze density can vary spatially due to factors such as wind, temperature gradients, and terrain features. Algorithms that estimate a global transmission map or atmospheric light may struggle with these non-uniform conditions.

To address the challenge of scene diversity and complexity, several approaches have been proposed:

1. **Adaptive Parameter Selection**: Some methods dynamically adjust parameters based on local image characteristics, allowing for more effective processing of different regions within the same image.

2. **Scene Segmentation**: By segmenting the image into regions with different characteristics (e.g., sky, buildings, vegetation) and applying specialized processing to each region, algorithms can better handle diverse scene elements.

3. **Learning-Based Approaches**: Deep learning methods can potentially learn to handle diverse scenes if trained on a sufficiently varied dataset. However, this requires large and diverse training data, which can be challenging to obtain.

4. **Fusion-Based Methods**: Combining multiple dehazing techniques or processing paths can help address different aspects of scene complexity, with each component handling specific challenges.

5. **Physics-Guided Learning**: Incorporating physical models and constraints into learning-based approaches can help them generalize better to diverse scenes by providing a stronger inductive bias.

Despite these approaches, scene diversity and complexity remain significant challenges for image dehazing, and developing algorithms that perform well across all scenarios is an ongoing research goal.

### 2.5.2 Color Distortion and Artifacts

Color distortion and artifacts are common issues in image dehazing that can significantly affect the visual quality of the results. These problems can arise from various sources, including limitations of the atmospheric scattering model, inaccurate parameter estimation, and algorithmic constraints.

Color distortion in dehazed images often manifests as unnatural color saturation, color shifts, or color inconsistencies. This can occur when the dehazing algorithm incorrectly estimates the atmospheric light or transmission map, leading to improper color correction. For example, if the atmospheric light is estimated to be too blue, the dehazed image may have a yellowish tint as the algorithm tries to compensate for the perceived blue haze.

The dark channel prior-based methods are particularly prone to color distortion in regions where the prior does not hold, such as sky regions or objects with colors similar to the atmospheric light. In these cases, the transmission may be underestimated, leading to excessive color correction and unnatural appearances.

Halo artifacts are another common issue, especially for patch-based methods. These artifacts appear as bright or dark bands around edges where there are significant depth discontinuities. They occur because the patch-based operations used to estimate the transmission map do not respect object boundaries, leading to incorrect transmission values near edges. The size of the patch used in these methods directly affects the severity of halo artifacts, with larger patches generally producing more noticeable halos.

Over-enhancement is a problem where the dehazing algorithm removes too much haze, resulting in an unnatural, over-processed appearance. This can lead to excessive contrast, saturated colors, and loss of the atmospheric perspective that provides depth cues in natural scenes. Over-enhancement often occurs when the algorithm parameters are not properly tuned or when the algorithm does not adapt to the specific haze density in the scene.

Noise amplification is another artifact that can degrade the quality of dehazed images. Since dehazing involves dividing the input image by the transmission map (which can have small values in hazy regions), any noise in the original image can be significantly amplified in the dehazed result. This is particularly problematic for images captured in low-light conditions or with high ISO settings, which already contain noise before dehazing.

Texture loss can occur when dehazing algorithms over-smooth the image in an attempt to reduce noise or artifacts. This can result in a loss of fine details and textures, making the dehazed image appear artificial or cartoon-like. Some methods use strong regularization or smoothing operations on the transmission map, which can inadvertently remove texture information from the final result.

To address these issues, various techniques have been proposed:

1. **Guided Filtering**: Using edge-preserving filters like the guided filter to refine the transmission map can help reduce halo artifacts while preserving edge details.

2. **Adaptive Parameter Selection**: Adjusting parameters based on local image characteristics can help prevent over-enhancement and color distortion in different regions of the image.

3. **Color Correction**: Post-processing steps that specifically address color fidelity, such as white balancing or color transfer techniques, can help mitigate color distortion.

4. **Noise-Aware Processing**: Incorporating noise models or denoising steps into the dehazing pipeline can help prevent noise amplification in the dehazed result.

5. **Detail Preservation Techniques**: Using methods that explicitly preserve or enhance texture details can help prevent texture loss during dehazing.

6. **Perceptual Loss Functions**: For learning-based methods, using perceptual loss functions that consider human visual perception can help reduce artifacts and improve the naturalness of dehazed images.

Despite these techniques, color distortion and artifacts remain significant challenges in image dehazing, particularly for challenging scenes with complex lighting, colors, or depth structures. Balancing effective haze removal with natural appearance and minimal artifacts is a delicate trade-off that continues to drive research in this field.

### 2.5.3 Computational Efficiency

Computational efficiency is a critical consideration for image dehazing algorithms, particularly for applications that require real-time processing or have limited computational resources. The computational complexity of dehazing algorithms varies widely, from simple contrast enhancement techniques to sophisticated deep learning models.

Traditional physics-based methods like the Dark Channel Prior (DCP) involve several computationally intensive steps. The original DCP method uses soft matting for transmission map refinement, which has a high computational complexity of O(N²) for an image with N pixels. This makes it impractical for real-time applications or high-resolution images. Later improvements, such as using guided filtering instead of soft matting, reduced the complexity to O(N), making the algorithm more efficient but still challenging for real-time processing of high-resolution images.

Patch-based operations, which are common in many traditional dehazing methods, can be computationally expensive, especially for large patch sizes. These operations typically involve sliding a window over the entire image and computing statistics or applying filters within each window. The computational complexity increases with both the image size and the patch size, making these methods slow for high-resolution images or when large patches are needed for effective dehazing.

Deep learning-based methods, while often more effective than traditional approaches, can be computationally intensive during both training and inference. Training deep neural networks for image dehazing requires significant computational resources, including high-performance GPUs and large amounts of memory. The training process can take hours or even days, depending on the network architecture, dataset size, and training strategy.

During inference, the computational requirements of deep learning models depend on their architecture and size. Large models with millions of parameters may achieve better dehazing performance but require more memory and computational power for inference. This can be a limitation for deployment on resource-constrained devices such as mobile phones, embedded systems, or edge devices.

Real-time processing is a particular challenge for image dehazing, especially for high-resolution images or video streams. Many applications, such as autonomous driving, surveillance, or augmented reality, require dehazing to be performed in real-time with minimal latency. Achieving this level of performance often requires compromises in terms of dehazing quality or specialized hardware acceleration.

Memory consumption is another important aspect of computational efficiency. Some dehazing algorithms, particularly those based on global operations or large neural networks, require significant memory resources. This can be a limitation for devices with limited memory capacity or when processing very high-resolution images.

To address these computational efficiency challenges, various approaches have been proposed:

1. **Algorithm Optimization**: Optimizing the implementation of dehazing algorithms using techniques such as parallel processing, vectorization, or algorithm-specific optimizations can significantly improve performance.

2. **Hardware Acceleration**: Leveraging specialized hardware such as GPUs, FPGAs, or dedicated image processing units can accelerate dehazing algorithms, particularly those based on deep learning or parallel operations.

3. **Model Compression**: For deep learning-based methods, techniques such as pruning, quantization, or knowledge distillation can reduce the model size and computational requirements while maintaining reasonable performance.

4. **Multi-Resolution Processing**: Processing lower-resolution versions of the image for initial estimates and then refining at higher resolutions can reduce computational load while preserving quality.

5. **Approximate Computing**: Using approximate algorithms or computations that trade some accuracy for speed can be acceptable for applications where perfect dehazing is not critical.

6. **Incremental Processing**: For video dehazing, processing only the changed parts of consecutive frames or reusing computations from previous frames can improve efficiency.

7. **Lightweight Architectures**: Designing neural network architectures specifically for efficiency, such as MobileNet-inspired designs or networks with depthwise separable convolutions, can reduce computational requirements.

Despite these approaches, achieving a good balance between dehazing quality and computational efficiency remains a challenge, particularly for real-time applications or resource-constrained devices. The development of more efficient algorithms and hardware acceleration techniques continues to be an important research direction in image dehazing.

### 2.5.4 Training Data Scarcity