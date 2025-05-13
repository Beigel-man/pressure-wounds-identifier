# TAU DataMed Home Assignment project

I will start by stating the challenges (and how I overcame):


## First challange - Segmentation
• Wounds are amorphic and come in many shapes.
• Every image has a different size.
• Very little labled data for masks. (Smart augmentation is needed)
Color classification
• Colors are very subjective and rely heavily on the different lighting in the image. • Very little labled data - and not all color labels are present.
Laying the ground for a Unet NN
I started off with building a small Unet model and train it myself. reasons to use Unet-
• can levarage data augmantations (which is much needed here)
• originally developed for medical purposes
• the skip connections and the decoder encoder structure help a lot with segmentation tasks This proved fustrating because although it worked ok, the segmentation was not precise enough.
next I remembered a trained model for public use called SAM (Segment Anything Model) which can segment an area by clicking on any part of the wound. for this to be automatic I created a small model to capture a change in coloring as a closed shaped around the middle of the picture. I then chose its centroid and it was the point for the SAM to operat. When the point was in the wound it worked great. but it wasn't always the case.
Finally- I decided to go with a pretrained Unet model with Resnet encoder, that has some 'knowledge' on wounds. With the labled mask and their augmentation as help, (And a lot of Hyper Parameters tuning) Really good results were obtained.


## Second challange - Color classification
A naive approach to color classification.
This is because the labeled data don't contain all the labels.
• Lack of adequate color labeling was the biggest challange.
• The second challenge was matching the h-s-v color ranges to what seen in the picture.
for calibration I used a simple interactive tool (made with chat's help) to match h-s-v values to the colors in the image. the code opens up a picture and the user chooses points and classifies them as yellow red black or pink. This allowed me to choose the color ranges for this function. IMPORTANT TO NOTE: This is a rule based solution I turned to- because I think the labels were not good enough so I couldn't train a good model for that. I think the results are pretty good.

reasons to use h-s-v:

- better than rgb for lighting differences
- more intuitive when switching colors 


For all unlabled images- 
For classification- I decided on a simple rule- if a dominant color is more than twice as high as the socond dominant color, it's classified as the dominant. If not, than it's 'mixed'

- Important note:
The color ranges can be changed according to what really is known about the tissues color and kind.
Black is challanging. a hole (no light is arriving to the wound) can be interperated as black when actually it's unknown. seing more examples of black tissue could help with this.

* For a future better tissue classification I would have:
- obtain more examples
- train a model that will not rely on colors alone but also on the texture of the wound / tissue

regarding color certainty - I created a function to calculate:
For single-color classification: it takes into account how dominant the color is and how much of the wound is classified.
For mixed: How close the top colors are (more equal = higher confidence it's truly mixed)

but- the black problem kind of overshadows (wink ;) this confidence level. For a first code it's not bad.