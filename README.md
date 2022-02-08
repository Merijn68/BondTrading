# Tijdreeksanalyse van bondprijzen

Voorspellen van prijzen van staatsobligaties op basis van marktdata met behulp van Neurale Netwerken

Notebooks \ 01_tijdreeksanalyse Bondprijzen     -> Introductie / Business understanding / Management samenvatting
Notebooks \ 02_data_voorbereiding               -> Data Preperation
Notebooks \ 03_data_exploratie                  -> Data Exploratie
04_base_model                                   -> Eerste modellen alleen getrained op het signaal
05_toevoegen_features                           -> Toevoegen van features om bondprijzen te voorspellen
06_richting_voorspellen                         -> Een andere aanpak waarbij de richting van de bondprijsbeweging wordt voorspeld

Project Organization
------------
        
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           
        │   └── make_dataset.py <- Scripts to generate datasets
        |   └── join_data.py    <- Join datasets
        |   └── split.py        <- Split test/train data
        |   └── window.py       <- Windowing timeseries data
        │
        ├── features       
        │   └── build_features.py   <-Adding features to the data
        │
        ├── models         
        │   │              
        │   ├── base_model.py   <- Base RNNModels and hypertrainable models
        │   └── evaluate.py     <- Metrics and custom loss function for models
        |   └── hyper.py        <- code for hypertuning the models
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
  --------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
