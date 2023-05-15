## The inflation project

This project is focused on investigating the inflation phenomenon in Poland over the last years and its impact on the consumers' situation.

### Motivation

Beginning from 2022, we can observe an unusual increases of the inflation level in many countries around the world. The stable evolution of prices is a natural process in healthy and developing economies, but when the pace of growths becomes too high, especially exceeding the rises of people's wages, the situation gets challenging. The analysis of the inflation data allows to study the problem, uncovering not only how the prices changed over the last years, but also what was their impact on the consumers.

The aim of this project is to investigate and describe the dynamics of inflation in Poland, using the official data provided by the [Polish Central Statistical Office](https://stat.gov.pl/en/). The considered datasets include both the price indices and their absolute values (for particular products and services), but also the wages of consumers. The data are split into various sections (including voivodeships and households) and cover the time frame from January 2010 to March 2023. Due to the wide range of the available records, we can compare the current situation on the market with the one before the latest growths.

### Possible questions to answer

* How the inflation changed over time - in general, in reference to particular good(s), for different types of households?
* Which groups/types of products recorded the highest increases in price in recent years?
* Are there any significant differences in price levels between various regions of Poland? Where the prices change similarly?
* What is the dynamics of wages? Do they increase proportionally to the inflation?
* How the consumers' wealth changed in time? What are the most recent results?

### Tools and methods

The analysis has been performed using the Python language, and enclosed in the form of Jupyter notebooks as follows:

* the `0-data_cleaning.ipynb` notebook includes the description of data collection and cleaning operations,
* the `1-inflation_analysis.ipynb` notebook contains the investigations on the inflation (price indices) dynamics, and
* the `2-wages_analysis.ipynb` notebook includes the analysis of the wages and their relations with the prices.

The raw data files (.csv) can be found in the `raw_data` catalogue, with all the cleaned datasets stored in the `inflation_database.db` database. The `voivodeships_map_data` catalogue contains the geospatial data of Polish voivodeships, which are used to present the results on maps.

The code has been run under `python ver. 3.11.0` and the libraries listed in the `requirements.txt` file. All the utility functions used in the notebooks can be found in the `utils.py` file.
