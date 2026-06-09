<h2>TensorFlow-FlexUNet-Image-Segmentation-Synapse-Abdominal-MultiOrgan (Updated: 2026/06/09)</h2>

Sarah T. Arai<br>
Software Laboratory antillia.com<br>
<ul>
<li>2026/06/9: Updated <a href="./src/TensorFlowFlexModel.py">TensorFlowFlexModel.py</a></li>
<li>2026/06/9: Added 3D Volume Segmentation</li>
</ul>
This is the first experiment of Image Segmentation for <b>Synapse-Abdominal-MultiOrgan</b>,
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels PNG 
<a href="https://drive.google.com/file/d/1tfXRc8AlP8YhxI6yIb7T67SzcXGl9xUX/view?usp=sharing">
Augmented-Synapse-Abdominal-MultiOrgan-ImageMask-Dataset.zip
</a> (<a href="https://www.apache.org/licenses/LICENSE-2.0">Apache 2.0</a>), 
which was derived by us from <br><br> 

<a href="https://www.kaggle.com/datasets/shinjinidey/synapse-dataset/data">
<b>
synapse-dataset
</b>
</a> by Shinjini Dey.
<br>
<br>
<hr>
<b>Acutual Image Segmentation for  Synapse-Abdominal-MultiOrgan Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.<br><br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10002_102.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10002_102.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10002_102.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10003_147.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10003_147.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10003_147.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10005_60.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10005_60.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10005_60.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from <br><br> 
<a href="https://www.kaggle.com/datasets/shinjinidey/synapse-dataset/data">
<b>
synapse-dataset
</b>
</a> by Shinjini Dey.
<br><br>
For more information on <b>synapse-dataset</b>, please refer to 
<a href="https://www.synapse.org/Synapse:syn3193805/files/">
Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge
</a>
<br><br>
The following description  was taken from the web site 
<a href="https://www.kaggle.com/datasets/shinjinidey/synapse-dataset/data">
<b>
synapse-dataset
</b>
</a>
<br>
<br>
<b>About Dataset</b><br>
Abdomen Raw-Data from Multi-Atlas Labeling Beyond the Cranial Vault dataset containing 30 
image files and 30 label files each in .nii format.
<br><br>
<b>Source:</b><br>
<a href="https://www.synapse.org/Synapse:syn3193805/files/">https://www.synapse.org/Synapse:syn3193805/files/</a>
<br><br>
<b>Cite:</b><br>
harrigr. (2015). Segmentation Outside the Cranial Vault Challenge [Dataset]. Synapse. <br>
<a href="https://doi.org/10.7303/SYN3193805">https://doi.org/10.7303/SYN3193805</a>
<br><br>
<b>License</b><br>
<a href="https://www.apache.org/licenses/LICENSE-2.0">Apache 2.0</a>)
<br><br>
<h3>
2 Synapse-Abdominal-MultiOrgan ImageMask Dataset
</h3>
<h4>2.1 Download ImageMask Dataset</h4>
 If you would like to train this Synapse-Abdominal-MultiOrgan Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1tfXRc8AlP8YhxI6yIb7T67SzcXGl9xUX/view?usp=sharing">
Augmented-Synapse-Abdominal-MultiOrgan-ImageMask-Dataset.zip
</a> (<a href="https://www.apache.org/licenses/LICENSE-2.0">Apache 2.0</a>), 
which was derived by us from <br><br> , 
expand the downloaded dataset, and put it under <b>./dataset</b> folder to be:
<pre>
./dataset
└─Synapse-Abdominal-MultiOrgan
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Synapse-Abdominal-MultiOrgan Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/Synapse-Abdominal-MultiOrgan_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<h4>2.2 ImageMask Dataset Derivation</h4>
The folder structure of <b>synapse dataset</b> is the following.<br>
<pre>
./archive
  ├─img
  │  ├─img0001.nii
  │  ├─img0002.nii  
...
  │  └─img0030.nii
  └─label
      ├─label0001.nii
      ├─label0002.nii
... 
      └─label0030.nii
</pre>
<b>Step 1</b><br>
We used a simple Python script and the following color-class-mapping table to generatea a 512x512 pixels PNG master dataset 
with colorized masks from all pairs of image NIfTI file in <b>img</b> folder and 
their corresponding label NIfTI file in <b>label</b> folder.<br><br>
<a id="color-class-mapping-table"><b>Synapse-Abdominal-MultiOrgan color class mapping table</b></a>
<br> 
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>0</td><td with='80' height='auto'><img src='./color_class_mapping/Background.png' widith='40' height='25'></td><td>(0, 0, 0)</td><td>Background</td></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/Spleen.png' widith='40' height='25'></td><td>(128, 30, 30)</td><td>Spleen</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/Right Kidney.png' widith='40' height='25'></td><td>(30, 128, 30)</td><td>Right Kidney</td></tr>
<tr><td>3</td><td with='80' height='auto'><img src='./color_class_mapping/Left Kidney.png' widith='40' height='25'></td><td>(30, 30, 128)</td><td>Left Kidney</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/Gallbladder.png' widith='40' height='25'></td><td>(255, 0, 0)</td><td>Gallbladder</td></tr>
<tr><td>5</td><td with='80' height='auto'><img src='./color_class_mapping/Esophagus.png' widith='40' height='25'></td><td>(0, 255, 0)</td><td>Esophagus</td></tr>
<tr><td>6</td><td with='80' height='auto'><img src='./color_class_mapping/Liver.png' widith='40' height='25'></td><td>(0, 0, 255)</td><td>Liver</td></tr>
<tr><td>7</td><td with='80' height='auto'><img src='./color_class_mapping/Stomach.png' widith='40' height='25'></td><td>(255, 255, 0)</td><td>Stomach</td></tr>
<tr><td>8</td><td with='80' height='auto'><img src='./color_class_mapping/Aorta.png' widith='40' height='25'></td><td>(255, 0, 255)</td><td>Aorta</td></tr>
<tr><td>9</td><td with='80' height='auto'><img src='./color_class_mapping/Inferior Vena Cava.png' widith='40' height='25'></td><td>(0, 255, 255)</td><td>Inferior Vena Cava</td></tr>
<tr><td>10</td><td with='80' height='auto'><img src='./color_class_mapping/Portal & Splenic Vein.png' widith='40' height='25'></td><td>(128, 128, 0)</td><td>Portal & Splenic Vein</td></tr>
<tr><td>11</td><td with='80' height='auto'><img src='./color_class_mapping/Pancreas.png' widith='40' height='25'></td><td>(128, 0, 128)</td><td>Pancreas</td></tr>
<tr><td>12</td><td with='80' height='auto'><img src='./color_class_mapping/Right Adrenal Gland.png' widith='40' height='25'></td><td>(128, 128, 128)</td><td>Right Adrenal Gland</td></tr>
<tr><td>13</td><td with='80' height='auto'><img src='./color_class_mapping/Left Adrenal Gland.png' widith='40' height='25'></td><td>(255, 255, 255)</td><td>Left Adrenal Gland</td></tr>
</table>
<br>
For simplicity, we excluded all empty black label slices and their corresponding image slices to generate the PNG master, 
because they were irrelevant to training our segmentation model.
<br><br>
<b>Step 2</b><br>
We generated our augmented dataset from the PNG master by using the following image deformation and distortion tools.<br>
<a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a><br>
<a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a> <br>
<br>
<h4>2.3 Image and Mask samples</h4>
<b>Train sample images</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train sample masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Synapse-Abdominal-MultiOrgan TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers = 8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 14

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Synapse-Abdominal-MultiOrgan 1+13 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
; 1+13 classes
rgb_map = {(0,0,0):0, (128,30,30):1, (30,128,30):2, (30,30,128):3, (255,0,0):4, (0,255,0):5, (0,0,255):6, (255,255,0):7, \ 
           (255,0,255):8, (0,255,255):9, (128,128,0):10, (128,0,128):11, (128,128,128):12, (255,255,255):13 }
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 40.<br><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/train_console_output_at_epoch40.png" width="1024" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Synapse-Abdominal-MultiOrgan.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/evaluate_console_output_at_epoch40.png" width="1024" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Synapse-Abdominal-MultiOrgan/test was low, and dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.0223
dice_coef_multiclass,0.99
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Synapse-Abdominal-MultiOrgan.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Synapse-Abdominal-MultiOrgan Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.<br><br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10002_109.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10002_109.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10002_109.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10003_147.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10003_147.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10003_147.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10003_161.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10003_161.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10003_161.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10005_60.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10005_60.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10005_60.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10006_87.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10006_87.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10006_87.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/images/10007_108.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test/masks/10007_108.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_output/10007_108.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
6 3D Volume Segmentation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan</b> folder<br>
,and run the following bat file to infer images segmentation for 2D slices of 3D volume NIfTI files
 by the Trained-TensorFlowFlexUNet model for Synapse-Abdominal-MultiOrgan.<br>
<pre>
>./5.infer3d.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNet3DInferencer.py ./train_eval_infer.config
</pre>
<b>infer3d section </b> in <a href="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/train_eval_infer.config">
train_eval_infer.config
<a></b>
<pre>
[infer3d] 
;Specify an images_dir which contains NIfTI files
images_dir    = "./mini_test_3d/images/"
output_dir    = "./mini_test_3d_output/"
slice_shape_order = "hwd"
slice_normalize = True
slice_resize   = (512,512)
slice_rotation = "cv2.ROTATE_90_CLOCKWISE"
mask_overlay  = True
</pre>
<hr>
<b>Acutual Image Segmentation for 2D Slices of a Synapse-Abdominal-MultiOrgan NIfTI</b><br>
Some Slices, Inferred Masks and Mask overlays for a 3D volume <b>img0008.nii</b> file in <b>archive/img</b> folder.
 folder.<br>
<br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<br>
<table>
<tr>
<th>Image</th>
<th>Inferred-mask</th>
<th>Mask overlay</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/slices/10085.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/masks/10085.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/overlays/10085.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/slices/10097.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/masks/10097.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/overlays/10097.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/slices/10109.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/masks/10109.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/overlays/10109.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/slices/10121.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/masks/10121.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/overlays/10121.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/slices/10133.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/masks/10133.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/overlays/10133.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/slices/10145.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/masks/10145.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/mini_test_3d_output/img0008.nii/overlays/10145.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>
7 MaskOverlay Video of 3D Volume Segmentation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan</b> folder, and run the following bat file 
to generate <b>overlays.mp4</b> or <b>overlay.gif</b> for MaskOverlays of 3D Volume Segmentation. <br>
<pre>
>./6.video3d.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/MaskOverlayVideoGenerator.py ./train_eval_infer.config
</pre>
<br>
<b>infer3d section </b> in <a href="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/train_eval_infer.config">
train_eval_infer.config
<a></b>
<pre>
[infer3d] 
mask_overlay  = True
;Specify ".mp4" or ".gif".
;video_fileformat  = ".mp4"
video_fileformat  = ".gif"
</pre>
<br>
<b>overlays.gif</b><br>
<img src="./projects/TensorFlowFlexUNet/Synapse-Abdominal-MultiOrgan/video_3d/overlays.gif">
<br>

<br>
<h3>
References
</h3>
<b>1. SEF-UNet: advancing abdominal multi-organ segmentation with SEFormer and depthwise cascaded upsampling
</b><br>
Yaping Zhao, Yizhang Jiang, Lijun Huang, Kaijian Xia<br>
<a href="https://peerj.com/articles/cs-2238/">
https://peerj.com/articles/cs-2238/
</a>
<br><br>
<b>2. An Efficient Vision Mamba–Transformer Hybrid Architecture for Abdominal Multi-Organ Image Segmentation</b><br>
Fang Lu, Jingyu Xu, Qinxiu Sun and Qiong Lou<br>
<a href="https://www.mdpi.com/1424-8220/25/21/6785">https://www.mdpi.com/1424-8220/25/21/6785</a>
<br><br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-CURVAS-MICCAI2024-Abdominal-MultiOrgan</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-CURVAS-MICCAI2024-Abdominal-MultiOrgan">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-CURVAS-MICCAI2024-Abdominal-MultiOrgan
</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-MICCAI-FLARE22-Abdominal-Organ</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-MICCAI-FLARE22-Abdominal-Organ">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-MICCAI-FLARE22-Abdominal-Organ
</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>

