# Generation of data and classification

## Topics Set
This step requires having an available word2vec model trained on a twitter sample. We did this by training a model with an embedding dimension of 50. Similar models can  be collected [here](http://fredericgodin.com/software/) if you don't have a large enough collection of tweets on which to train. Once this is done, topics set can be computed via the [initial_computations](../bash_scripts/initial_computations.sh) script.

## Census Data

Before running the location pipeline, the census data needs to be stored locally. Depending on the resolution of the data wanted for the later training, you may use the ``IRIS`` or ``insee`` flag. The __IRIS__ dataset is already provided in the [INSEE IRIS](../data_files/INSEE_IRIS) while the __INSEE\_200m__ needs to be downloaded following the link in the README file. Similarly,  annotation of linkedin users for professional occupation was done using the census data provided in the [INSEE_SOC](../data_files/INSEE SOC) folder.

## Location/Occupation Pipeline

We start by extracting location data from our twitter sample from users with geolocations. Information on this set of users is then augmented by crawling the last 3,240 tweets that they posted as well as their profile information. 
This is done via the [location_pipeline](../bash_scripts/location_pipeline.sh) script. 

Once the Twitter data has been extracted and parsed, model training can be performed from [here](../python_scripts/location_pipeline.py).

An identical procedure must be followed for the occupation pipeline with the added constraint of having to self-annotate the dataset.

## Remote-Sensing Pipeline

Additional steps must be taken to obtain the data needed for this pipeline. First, coordinates for the set of locations for which the GSV Satellite and Street View must be stored. In order to be able to collect this a Google Maps Static API key must be obtained and introduced in the [get\_gsv\_images.py](../python_scripts/data_coll_process/get_gsv_images.py). The collection must then be run as follows:

``python get_gsv_images.py -input_file -of output_file  -gsv_met metadata_file -log log_file``

Non-Residential locations must then be filtered out. To do so, we train a ResNet50 on the UCMerced Dataset (download [here](http://weegee.vision.ucmerced.edu/datasets/landuse.html)) to filter out satellite tiles recognized as something different to a residential location. Training of the model can be achieved with the [UCMERCED_train.py](../python_scripts/data_coll_process/UCMERCED_train.py) script. Filtering and generation of the annotation-ready dataset is then performed in this [notebook](../ipynb_notebooks/Recognize_Residential_Areas.ipynb)

Final plots are then done here.