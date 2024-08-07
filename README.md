# Soiling detection for Advanced Driver Assistance Systems

## Overview
Here is breakdown of parts in the repository

### Root 
In the root you can find requirements with libraries used in our environment.

### data_statistics
networks_evaluations.ipynb -> There is the script used for network results evaluations.
basic_statistics.ipynb -> And script that wasnt used for the article, but creates histogram of all images and distrbution of classes accross dataset.

### helpers
<b>filter_of_files.csv</b> -> This is table with our data splits. You can find there all images with 3 columns. 
<ul>
<li> To be deleted - Those are obvious errors. By removing files with nok (not ok), you get <i>"Correct files"</i> dataset. </li>
<li> Unclear - Unclear boundaries of annotations or simplifications in annotations per images. By removing those and <i>"Correct files"</i> you get dataset <i>Correct clear files</i></li>
<li>Strict delete - Combination of two above + added check accross multiple images, so the annotations should follow same annotations logic and definition in all images. By removing those files you get <i>"Correct clear strict files"</i></li>
data_selection.ipynb -> Script to create txt files with list of files for each set of data.</li>
</ul>

### networks_run
train_all.py -> Script which starts training of all networks. 
pytorch_networks/train/pytorch_l_base_train.py -> Universal script to train one segmentation network from framework
pytorch_networks/predict/pytorch_l_base_pred.py -> Universal script to run prediction of one network.  
evaluation/evaluation.py -> Evaluation script to use per one network prediction

## Manual
<ol>
  <li>Download concatenated woodscape dataset https://drive.google.com/file/d/1WNlDBADwlaheMaVpIjEeAMklw7jle9Ja/view?usp=sharing and unzip it into repository</li>
  <li>You have structure git_root/woodscape_input/{gtLabels},{rgbImages},{rgbLabels}/*.png</li>
  <li>Download trained models and their outputs https://drive.google.com/file/d/13k17SjgQHZCO-1Ctr3DY_bW6DGvQZZie/view?usp=sharing and unzip it into repository</li>
  <li>You have structure git_root/model_outputs/{*netowork_name_encoder_loss_dataset*}/{evaluations},{model}, {predictions}</li>
  <li>Run helpers/data_selection.ipynb. This will create folder "woodscape_preprocessed" with txt files for training. Your data are ready for training</li>
  <li>Run networks_run/train_all.py. You can comment out some setting you dont want. Results are stored in models_output.</li>
  <li>Once you have trained model, run networks_run/pytorch_networks/predict/pytorch_l_base_pred.py per trained model with its setup using test images. Outputs should be stored next to the traines model into folder "predictions"</li>
  <li>After you do this for all networks and configuration. Run for each predictions networks_run/evaluation/evaluation.py and store results again next to the models in folder "evaluations"</li>
  <li>You can run data_statistics/netoworks_evaluation.ipynb. If you removed some configurations, removed them also from list of settings contained in jupyter notebook.</li>
</ol> 