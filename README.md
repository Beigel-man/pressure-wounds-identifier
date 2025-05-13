# TAU DataMed Home Assignment project

I will start by stating the challenges (and how I overcame):

---

## First challenge – Segmentation

- Wounds are amorphic and come in many shapes.
- Every image has a different size.
- Very little labeled data for masks (smart augmentation is needed).

### Laying the ground for a U-Net

I started with building a small U-Net model and trained it myself. Reasons to use U-Net:
- Can leverage data augmentations (which is much needed here)
- Originally developed for medical purposes
- The skip connections and encoder-decoder structure help a lot with segmentation tasks

This proved frustrating because although it worked okay, the segmentation was not precise enough.

Next, I remembered a trained model for public use called SAM (Segment Anything Model), which can segment an area by clicking on any part of the wound. To automate this, I created a small model to capture a change in coloring as a closed shape around the center of the image. I then chose the centroid, which was used as the prompt point for SAM. When the point landed in the wound — it worked great, but it wasn't always the case.

**Finally** — I decided to go with a **pretrained U-Net model with a ResNet encoder**, which has some domain "knowledge" of wounds. Combined with labeled masks, augmentation, and a lot of hyperparameter tuning — really good results were obtained.

---

## Second challenge – Color classification

A naive approach to color classification.

The biggest issues:
- Labeled data did not contain all possible labels.
- Color is highly subjective and depends heavily on lighting.

### Calibration approach

To calibrate, I created an interactive tool (with ChatGPT's help) to match HSV values to colors in the image. The code opens an image and allows the user to click on points and label them as red, yellow, black, or pink. This allowed me to define color ranges for the classification function.

> **Note:** This is a **rule-based solution** due to insufficient labeled data. A trained model would require more robust data.

### Why HSV?

- Better than RGB under different lighting conditions
- More intuitive when working with specific color ranges

### Rule for unlabeled images:

If a dominant color appears **more than twice** as often as the second most dominant — it's classified as that color.  
If not — it’s labeled as **"mixed"**.

### Notes on black classification:

- "Black" is tricky: a hole with no light can appear black, but it might just be an unknown area.
- Seeing more examples would help define it better.

### Future improvements:

- Get more labeled examples
- Train a model that includes texture analysis, not just color

### Color certainty score:

- For **single-color**: Combines how dominant the color is and how much of the wound is classified.
- For **mixed**: Measures how close the top colors are — the more equal, the more confident it's truly mixed.

> But the black classification issue still overshadows (wink 😉) this metric.

---

##  Installation

-You can install the package directly from GitHub: (read-only)

pip install git+https://github.com/Beigel-man/pressure-wounds-identifier.git

-or for developers:

git clone https://github.com/Beigel-man/pressure-wounds-identifier.git

cd pressure-wounds-identifier

python -m venv .venv && source .venv/bin/activate  

- On Windows: 
.venv\Scripts\activate
pip install -e .

##  After Installation
You can use these commands:
Each command runs a different part of the pipeline (this is the correct order):

'train-model' — Train the segmentation model

'run-inference' — Apply the model to unlabeled images

'classify-colors' — Classify wounds colors and tissue type
