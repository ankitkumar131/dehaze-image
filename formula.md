# Understanding Image Dehazing Formulas

## 1. Dark Channel Computation

Formula: `D(x) = min(min(Ic(y)))`
                y∈Ω(x) c∈{r,g,b}

### Simple Explanation
1. **What it does**:
   - Finds the darkest pixel in a small neighborhood
   - Looks at all three color channels (red, green, blue)
   - Like finding the darkest shadow in a small area

2. **Components**:
   - D(x): The dark channel value at position x
   - Ic(y): Color intensity at pixel y
   - Ω(x): A small window around position x

3. **Real-world example**:
   - Imagine looking at a small patch of a black car
   - Even if it's shiny, some part will be very dark
   - The formula finds this darkest spot

## 2. Transmission Map Estimation

Formula: `t(x) = 1 - ω · D(I(x)/A)`

### Simple Explanation
1. **What it does**:
   - Estimates how much light reaches the camera
   - Like measuring how thick the haze is
   - Values range from 0 (complete haze) to 1 (no haze)

2. **Components**:
   - t(x): Transmission value at position x
   - ω: A fine-tuning constant (usually 0.95)
   - A: Atmospheric light (brightness of the haze)
   - D: Dark channel of normalized image

3. **Real-world example**:
   - Looking through fog at different distances
   - Nearby objects: high transmission (clear)
   - Far objects: low transmission (hazy)

## 3. Guided Filter (Refinement)

Formula: `q = a · I + b`
Where: `a = (CovI,p)/(VarI + ε)` and `b = μp - a · μI`

### Simple Explanation
1. **What it does**:
   - Smooths out the transmission map
   - Preserves important edges
   - Like using a smart blur tool

2. **Components**:
   - q: Refined output
   - I: Guidance image (original grayscale)
   - p: Initial transmission map
   - μ: Local mean
   - Cov: Covariance
   - Var: Variance
   - ε: Small number to avoid division by zero

3. **Real-world example**:
   - Like using tracing paper to smooth a rough sketch
   - But keeping sharp lines where they matter
   - Blending only in areas that should be smooth

## 4. Image Recovery (Dehazing)

Formula: `J(x) = (I(x) - A)/max(t(x), t0) + A`

### Simple Explanation
1. **What it does**:
   - Removes haze from the image
   - Recovers original colors
   - Prevents over-enhancement

2. **Components**:
   - J(x): Recovered clear image
   - I(x): Original hazy image
   - A: Atmospheric light
   - t(x): Transmission map
   - t0: Lower bound (typically 0.1)

3. **Real-world example**:
   - Like cleaning a dusty window:
     * First, figure out how dirty each part is (t(x))
     * Remove the appropriate amount of 'dirt' (I(x) - A)
     * Make sure not to over-clean (max(t(x), t0))
     * Add back the natural ambient light (+ A)

## Putting It All Together

1. **Step-by-Step Process**:
   - Calculate dark channel (find darkest areas)
   - Estimate transmission (measure haze thickness)
   - Refine with guided filter (smooth out)
   - Recover the final image (remove haze)

2. **Why This Works**:
   - Dark channel finds areas affected by haze
   - Transmission map measures haze intensity
   - Guided filter maintains image quality
   - Recovery formula balances haze removal and image quality

3. **Practical Tips**:
   - Works best on outdoor images
   - May need adjustment for different weather conditions
   - Balance between haze removal and noise
   - Consider processing time vs quality tradeoffs