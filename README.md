# Melanoma Detection Models by "RECOD Titans"

This repository is a branch of [Tensorflow/models/slim](https://github.com/tensorflow/models/tree/master/slim) containing the
models implemented by [RECOD "Titans"](https://recodbr.wordpress.com/) for the article ["Data, Depth, and Design: Learning Reliable Models for Melanoma Screening"](https://arxiv.org/abs/1711.00441).

This repository has also a branch/release for the models implemented by RECOD "Titans" for the IEEE ISBI 2017 Challenge presented by ISIC ([ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a) challenge / Part 3:
Lesion Classification). There's a separate repository for the [models used in Part 1: Lesion Segmentation](https://github.com/learningtitans/isbi2017-part1), and a [technical report](https://arxiv.org/abs/1703.04819) detailing our participation on both tasks. RECOD "Titans" got the best ROC AUC for melanoma classification (87.4%), 3rd best ROC AUC for seborrheic keratosis classification (94.3%), and 3rd best combined/mean ROC AUC (90.8%).

## Foreword

**Please bear with us as we finish this code repository. Launch date is: March 31st 2018.** The code and procedure here are already usable, but we are assembling and checking the final details. We will freeze the first release on the launch date.

Please, help us to improve this code, by [submitting an issue](https://github.com/learningtitans/data-depth-design/issues) if you find any problems.

Despite the best effort of authors, reproducing results of todays' Machine Learning is challenging, due to the complexity of the machinery, involving millions of lines of code distributed among thousands of packages — and the management of hundreds of random factors.

We are committed to alleviate that problem. We are a very small team, and unfortunately, cannot provide help with technical issues (e.g., procuring data, installing hardware or software, etc.), but we'll do our best to share the technical and scientific details needed to reproduce the results. Please, see our contacts at the end of this document.

Most of the code is a direct copy of the models posted in Tensorflow/Slim, adjusted to fit the melanoma detection task (dataset, data preparation, results formatting, etc.). We created the code needed for the SVM decision layers, the ensembles, the statistical analyses and graphs.

**If you use our procedures, protocols, or code in an academic context, please cite us.** The main reference is the "Data, Depth, and Design: Learning Reliable Models for Melanoma Screening" article. The "RECOD Titans at ISIC Challenge 2017" report may also be of interest. If the transfer learning aspects of this work are important to your context, you might find appropriate to cite the ISBI 2017 paper "Knowledge transfer for melanoma screening with deep learning" as well. All references are linked at the end of this document.

## Requirements

*Hardware:* You'll need a CUDA/cuDNN compatible GPU card with enough RAM. We tested our models on NVIDIA GeForce Titan X, Titan X (Pascal), Titan Xp, and Tesla K40c cards, all with 12 GiB of VRAM.

*Software:* All our tests used Linux. We ran most experiments on Ubuntu 14.04.5 LTS, and 16.04.3 LTS. You'll need Python v3.5+, with packages tensorflow-gpu v1.3.0, numpy, scipy, and sklearn. You can install those packages with pip. You'll also need wget, curl, git, ImageMagick, and bc (the command-line basic calculator).

*Docker installation:* The easiest way to install the needed software is to create a [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) container from image tensorflow/tensorflow:1.3.0-gpu-py3, and then add the remaining packages:

```bash
nvidia-docker pull tensorflow/tensorflow:1.3.0-gpu-py3

mkdir ~/ddd

nvidia-docker run -ti -e OUTSIDE_USER=$USER  -e OUTSIDE_UID=$UID -e OUTSIDE_GROUP=`/usr/bin/id -ng $USER` -e OUTSIDE_GID=`/usr/bin/id -g $USER` -v $HOME/ddd:/ddd --name data-depth-design tensorflow/tensorflow:1.3.0-gpu-py3 /bin/bash

# Type those commands in the shell inside the container:
apt-get update
apt-get install -y curl git imagemagick wget

groupadd --gid "$OUTSIDE_GID" "$OUTSIDE_GROUP"
useradd --create-home --uid "$OUTSIDE_UID" --gid "$OUTSIDE_GID" "$OUTSIDE_USER"

su -l $OUTSIDE_USER
ln -s /ddd  ~/ddd
```

The procedure above creates a user inside the container equivalent to your external user, and maps the external directory ~/ddd into /ddd inside the container. That is highly recommended because Docker filesystem isn't fit for extensive data manipulation.

## Cloning this repository

We're assuming that you'll use the commands below to clone this repository. If you use a different path than ~/ddd, adapt the instructions that follow as needed.

```bash
cd ~/ddd
git clone https://github.com/learningtitans/data-depth-design data-depth-design
```

## Obtaining and preparing the images

This repository contains all needed metadata, but no actual images. We collected data from several sources, listed below. Those sources are publicly obtainable — more or less easily — some requiring a license agreement, some requiring payment, some requiring both.

If you're going to use our pre-trained models (see below) and test on your own data, you don't have to procure any images. If you want to train the models "from scratch", or reproduce the tests in the article, you'll have to procure all datasets.

### Official ISIC 2017 challenge dataset

The official [ISIC 2017 Challenge](https://challenge.kitware.com/#phase/5840f53ccad3a51cc66c8dab) challenge dataset has 2,000 dermoscopic images (374 melanomas, 254 seborrheic keratoses, and 1,372 benign nevi). It's freely available, after signing up at the challenge website.

You have to download all three splits: training, validation, and test, and unzip them to the ~/ddd/sourceData/challenge directory (flat in the same directory, or in subdirectories, it does not matter). You'll also need the ground truth segmentation (Part 1) files for the three splits.

*The procedure:* (1) Download/unzip the challenge files into ~/ddd/sourceData/challenge. One way to accomplish it:

```bash
mkdir -p ~/ddd/sourceData/challenge
cd ~/ddd/sourceData/challenge

# Download and unzip all files here

mkdir -p ~/ddd/extras/challenge/superpixels
find . -name '*.png' -exec mv -v "{}" ~/ddd/extras/challenge/superpixels \;
```

### Additional ISIC Archive Images

We used additional images from the [ISIC Archive](http://isdis.net/isic-project/) an international consortium to improve melanoma diagnosis, containing over 13,000 dermoscopic images. They're freely available.

*The procedure:* Download all images to ~/ddd/sourceData/isic/images and all segmentation masks to ~/ddd/sourceData/isic/masksImages. If an image has several segmentation masks, download all of them.

```bash
mkdir -p ~/ddd/sourceData/isic
cd ~/ddd/sourceData/isic

# Get list of all image ids
wget "https://isic-archive.com:443/api/v1/image?limit=50000&offset=0&sort=name&sortdir=1" -O isicRawList
tr , "\n" < isicRawList > isicFullList
sed -n 's/.*"_id": "\(.*\)".*/\1/p' < isicFullList > isicAllIDs

# Download all images
mkdir images
cd images
cat ../isicAllIDs | while read imgid; do wget "https://isic-archive.com:443/api/v1/image/$imgid/download?contentDisposition=attachment" -O "$imgid.jpg"; sleep 1; done

# Check all image files were downloaded
cat ../isicAllIDs | while read imgid; do [[ -s "$imgid.jpg" ]] || echo "ERROR: $imgid.jpg not downloaded!"; done

cd ..

# Download all masks metadata
mkdir masksMeta
cd masksMeta
cat ../isicAllIDs | while read imgid; do curl -X GET --header 'Accept: application/json' 'https://isic-archive.com/api/v1/segmentation?limit=50&offset=0&sort=created&sortdir=-1&imageId='"$imgid" -o "$imgid.masks.json"; sleep 0.5; done

# Check all masks metadata were downloaded
cat ../isicAllIDs | while read imgid; do [[ -s "$imgid.masks.json" ]] || echo "ERROR: $imgid.masks.json not downloaded!"; done

cd ..

# Download all masks images

mkdir masksImages
cd masksImages
```

Create a file ```download_masks.sh``` in the current path with the following contents:

```bash
#!/bin/bash
while read imgid; do
    maskfile="../masksMeta/$imgid.masks.json"
    echo "Image: $imgid";
    cat "$maskfile" | tr , "\n" | sed -n 's/.*"_id": "\(.*\)".*/\1/p' | while read maskid; do
        echo "    Mask: $maskid"
        curl -X GET --header 'Accept: image/png' 'https://isic-archive.com/api/v1/segmentation/'"$maskid"'/mask?contentDisposition=inline' -o "$imgid.$maskid.png"
        sleep 1
    done
    echo
done
```

Continue with the following commands:

```bash
chmod u+x download_masks.sh
cat ../isicAllIDs | ./download_masks.sh
```

### Interactive Atlas of Dermoscopy

The [Interactive Atlas of Dermoscopy](http://www.dermoscopy.org/) has 1,000+ clinical cases (270 melanomas, 49 seborrheic keratoses), each with at least two images: dermoscopic, and close-up clinical. It's available for anyone to buy for ~250€.

*The procedure:* (1) Insert/Mount the Atlas CD-ROM. (2) Copy all image files to ~/ddd/sourceData/atlas. (3) Rename all files to lowercase. One way to accomplish it:

```bash
mkdir -p ~/ddd/sourceData/atlas
cd ~/ddd/sourceData/atlas

# Adapt the path /media/cdrom below to the CD mount point
find /media/cdrom/Images -name '*.jpg' -exec sh -c 'cp -v "{}" `basename "{}" | tr "[A-Z]" "[a-z]"`' \;
```

### Dermofit Image Library

The [Dermofit Image Library](https://licensing.eri.ed.ac.uk/i/software/dermofit-image-library.html) has 1,300 images (76 melanomas, 257 seborrheic keratoses). It's available after signing a license agreement, for a fee of ~50€.

*The procedure:* (1) Download/unzip all .zip dataset files to ~/ddd/sourceData/dermofit. One way to accomplish it:

```bash
mkdir -p ~/ddd/sourceData/dermofit
cd ~/ddd/sourceData/dermofit
# Download and unzip all files here
```

### The PH2 Dataset

The [PH2 Dataset](http://www.fc.up.pt/addi/ph2%20database.html) has 200 dermoscopic images (40 melanomas). It's freely available after signing a short online registration form.

*The procedure:* (1) Download/unzip all the images to ~/ddd/sourceData/ph2. One way to accomplish it:

```bash
mkdir -p ~/ddd/sourceData/ph2
cd ~/ddd/sourceData/ph2
# Download and unzip all images here
```

### Integrating the dataset

Start creating a flat folder with all lesion images, and another with all ground-truth segmentation masks:

```bash
mkdir -p ~/ddd/data/lesions
cd ~/ddd/data/lesions

find ~/ddd/sourceData -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' \) -not \( -iname '*_segmentation.png' -o -iname '*_superpixels.png' -o -iname '*mask.png' -o -path '*masksImages/*' -o -path '*_lesion/*' -o -path '*_roi/*' \) -exec bash -c 'source="{}"; dest=$(basename "$source"); echo "$source" "=>" "$dest"; ln "$source" "$dest"' \;

mkdir -p ~/ddd/data/masks
cd ~/ddd/data/masks

find ~/ddd/sourceData -type f \( -iname '*_segmentation.png' -o -iname '*mask.png' -o -path '*masksImages/*' -o -path '*_lesion/*' \) -exec bash -c 'source="{}"; dest=$(basename "$source"); echo "$source" "=>" "$dest"; ln "$source" "$dest"' \;
```

Now, create three sets of resized square images with sizes 598, 305 and 299. This is the only procedure where you'll need the ImageMagick dependency.

Create the file ~/ddd/data/resize_images.sh with the following contents:

```bash
#!/bin/bash
set -euo pipefail
for IMAGE_SIZE in 598 305 299; do

    export NEW_SIZE="$IMAGE_SIZE"

    mkdir -p ~/ddd/data/lesions"$NEW_SIZE"
    cd ~/ddd/data/lesions"$NEW_SIZE"

    find ~/ddd/data/lesions -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' \) -exec bash -c 'source="{}"; dest=$(basename "$source"); dest="${dest%.*}".jpg; echo "$source" "=>" "$dest"; convert "$source" -resize "$NEW_SIZE"x"$NEW_SIZE"\! "$dest"' \;

    mkdir -p ~/ddd/data/masks"$NEW_SIZE"
    cd ~/ddd/data/masks"$NEW_SIZE"

    find ~/ddd/data/masks -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' \) -exec bash -c 'source="{}"; dest=$(basename "$source"); dest="${dest%.*}".png; echo "$source" "=>" "$dest"; convert "$source" -resize "$NEW_SIZE"x"$NEW_SIZE"\! "$dest"' \;
done
```

Continue with the following commands:

```bash
cd ~/ddd/data/
chmod u+x resize_images.sh
./resize_images.sh
```

Finally, compute the average ground-truth mask (for the images with more than one mask):

```bash
cd ~/ddd/data/

mkdir averageMasks598
python -u ../data-depth-design/etc/compute_average_mask.py lesions598 masks598 averageMasks598

mkdir averageMasks305
python -u ../data-depth-design/etc/compute_average_mask.py lesions305 masks305 averageMasks305

mkdir averageMasks299
python -u ../data-depth-design/etc/compute_average_mask.py lesions299 masks299 averageMasks299
```

During the procedure above, some warnings are expected, because the EDRA Atlas images do not have ground-truth masks.

### Computing the predicted masks

The experiments evaluate two training sets: the same training split used in ISIC Challenge 2017 (isic), and a much larger one with images from several datasets (full).

Those training sets are used to create two segmentation models, using the code available in [TODO: link to segmentation model repo]. Instructions for training the segmentation models from scratch are available there.

Pre-trained segmentation models are also available:

* Segmentation model trained on isic: [TODO: link]
* Segmentation model trained on full: [TODO: link]

Instructions on how to install the models with pre-trained weights are included on the segumentation model README.

Below are the instructions for creating the predicted masks, once the models are area either trained or loaded:

-
-
-
-
## Main Factorial Experiment

Design of Experiments [TODO: Link] nomenclature in half a nutshell: *factor* is an explanatory variable (independent variable) manipulated by the experimenter; each factor assumes values called *levels*. A *treatment* is a choice of a level for each studied factor.

The table below explains the factors and levels chosen for this study. The levels of binary factors are, by convention, labelled -1 and 1.

|Symbol|Factor|Levels|
| :-:  |:-----|:-----|
|a|Model|(-1) Resnet-101 v2; (1) Inception v4|
|b|Training dataset|(-1) Training split of ISIC Challenge 2017; (1) Level 1 + Entire ISIC Archive + University of  Porto PH² + University of Edinburgh Dermofit|
|c|Input resolution|(-1) 299×299 px (305×305 if segmentation on); (1) 598×598 pixels|
|d|Training augmentation|(-1) Tensorflow/SLIM default; (1) Customized|
|e|Input normalization|(-1) None; (1) Subtract mean of each samples' pixels|
|f|Segmentation|(-1) No segmentation information; (1) Segmentation pre-encoded at input|
|g|Training length|(-1) Short (about half); (1) Full (30k batches for Resnet/40k for Inception)|
|h|SVM decision layer|(-1) Absent; (1) Present|
|i|Test augmentation post-deep|(-1) No; (1) Yes — decision is mean of 50 augmented samples|
|j|Test dataset|(0) Sample/split from Train/Full; (1) Validation split of ISIC Challenge 2017; (2) Test split of ISIC Challenge 2017; (3) Dermoscopic images from EDRA Atlas; (4) Clinical images from EDRA Atlas|

### Experiment steps

1. Create a Tensorflow-format dataset from the lesion images (and segmentation masks, when appropriate) for the training set;
2. Create a Tensorflow-format dataset from the lesion images (and segmentation masks, when appropriate) for the test set;
3. Train the deep classification model with the training set;
4. If there is no SVM layer:
    1. Make the predictions for the test set from the deep classification models;
5. If there is an SVM layer:
    1. Extract the features from the deep classification model for the training set;
    2. Train the SVM decision layer with those features;
    3. Make the predictions for the test set from the SVM model;

You run the steps using the appropriate scripts in ~/ddd/data-depth-design/main.doe/, according to the following table:

|Step|Name|Subdirectory|
|:-: |:---|:-----|
|1|Training dataset convertion|jbhi-0-1-create-train.run.dir/|
|2|Test dataset convertion|jbhi-0-2-create-test.run.dir/|
|3|Deep model training|jbhi-1-2-deep-train.run.dir/|
|4.i|Deep model prediction|jbhi-4-1-test-no-svm.run.dir/|
|5.i|Training features extraction|jbhi-3-1-svm-feature-train.run.dir/|
|5.ii|SVM layer training|jbhi-3-2-svm-train.run.dir/|
|5.iii|SVM layer prediction|jbhi-4-2-test-svm.run.dir/|

Each of the subdirectories above has a helper script ./main.run.sh which runs the step for all treatments. You can use it if you want to run all treatments (e.g., to perform the ANOVA analysis as described in the article).

If you want to run a particular treatment, you have to run the steps above in order, picking at each step a particular script ./experiment_XXX.run.sh, where XXX is the number indicated at the attached table.

### Analysis steps

0. Collect all results;
0. Perform the ANOVA and print the tables;
0. Create the correlograms;
0. Perform the simulations of the ensemble experiments;
0. Perform the simulations of the sequential procedures;


|Step|Name|Subdirectory|
|:-: |:---|:-----|
|6|Metrics collection|jbhi-5-1-metrics-anova.run.dir/|


## Secondary Experiment on the Effects of Transfer Learning

We ran a secondary experiment to assess the importance of transfer learning. The experiment was smaller than the full experiment, as we fixed factors f=-1 (i.e., no segmentation) and h=-1 (i.e., no SVM decision layer). The remainder of the experiment remained the same.

You can run the main experiment before the secondary experiment or vice-versa, indifferently, but when you move between them you'll have to rename some directories that both experiments use:

[TODO: instructions on which directories to move]

### Experiment steps

0. Create a Tensorflow-format dataset from the lesion images (and segmentation masks, when appropriate) for the training set;
0. Create a Tensorflow-format dataset from the lesion images (and segmentation masks, when appropriate) for the test set;
0. Train the deep classification model with the training set;
0. Make the predictions for the test set from the deep classification models;

You run the steps using the appropriate scripts in ~/ddd/data-depth-design/main.doe/, according to the following table:

|Step|Name|Subdirectory|
|:-: |:---|:-----|
|1|Training dataset convertion|jbhi-0-1-create-train.noseg.select.dir/|
|2|Test dataset convertion|jbhi-0-2-create-test.noseg.select.dir/|
|3|Deep model training|jbhi-1-2-deep-train.noseg.select.dir/|
|4|Deep model prediction|jbhi-4-1-test-no-svm.noseg.select.dir/|

Each of the subdirectories above has a helper script ./main.run.sh which runs the step for all treatments. You can use it if you want to run all treatments (e.g., to perform the ANOVA analysis as described in the article).

If you want to run a particular treatment, you have to run the steps above in order, picking at each step a particular script ./experiment_XXX.run.sh, where XXX is the number indicated at the attached table.

### Analysis steps

0. Collect all results;
0. Perform the ANOVA and print the tables;


## About us

The Learning Titans are a team of researchers lead by [Prof. Eduardo Valle](http://eduardovalle.com/) and hosted by the [RECOD Lab](https://recodbr.wordpress.com/), at the [University of Campinas](http://www.unicamp.br/), in Brazil.


### Our papers and reports

A Menegola, J Tavares, M Fornaciali, LT Li, S Avila, E Valle. RECOD Titans at ISIC Challenge 2017. [arXiv preprint arXiv:1703.04819](https://arxiv.org/abs/1703.04819) | [Video presentation](https://www.youtube.com/watch?v=DFrJeh6LkE4) | [PDF Presentation](http://eduardovalle.com/wordpress/wp-content/uploads/2017/05/menegola2017isbi-RECOD-ISIC-Challenge-slides.pdf)

A Menegola, M Fornaciali, R Pires, FV Bittencourt, S Avila, E Valle. Knowledge transfer for melanoma screening with deep learning. IEEE International Symposium on Biomedical Images (ISBI) 2017. [arXiv preprint arXiv:1703.07479](https://arxiv.org/abs/1703.07479) | [Video presentation](https://www.youtube.com/watch?v=upJApUVCWJY)
| [PDF Presentation](http://eduardovalle.com/wordpress/wp-content/uploads/2017/05/menegola2017isbi-TransferLearningMelanomaScreening-slides.pdf)

M Fornaciali, M Carvalho, FV Bittencourt, S Avila, E Valle. Towards automated melanoma screening: Proper computer vision & reliable results. [arXiv preprint arXiv:1604.04024](https://arxiv.org/abs/1604.04024).

M Fornaciali, S Avila, M Carvalho, E Valle. Statistical learning approach for robust melanoma screening. SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI) 2014. [DOI: 10.1109/SIBGRAPI.2014.48](https://scholar.google.com.br/scholar?cluster=3052571560066780582&hl=en&as_sdt=0,5&sciodt=0,5) | [PDF Presentation](https://sites.google.com/site/robustmelanomascreening/SIBGRAPI_Slides_MichelFornaciali.pdf?attredirects=0)

[Robust Melanoma Screening Minisite](https://sites.google.com/site/robustmelanomascreening/)


## Copyright and license

Please check files LICENSE/AUTHORS, LICENSE/CONTRIBUTORS, and LICENSE/LICENSE.

