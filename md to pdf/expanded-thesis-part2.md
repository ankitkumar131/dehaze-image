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