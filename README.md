# Movie Genre Predictor

![Project Logo](https://img.icons8.com/?size=100&id=80698&format=png)

## Overview

Movie Genre Predictor is a web application built using Streamlit that predicts the genre of a movie based on its title and description. It utilizes machine learning models trained on movie data to provide genre predictions.

# Links
- Link to the website : https://movie-genre-predictor-m.streamlit.app/
- Data and Models can be downloaded from: https://github.com/pankil-soni/codsoft/tree/master/Genre%20Classification

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Models and Accuracy](#models-and-accuracy)
- [Input Options](#input-options)
- [About the Project](#about-the-project)
- [License](#license)
- [Contact](#contact)

## Demo
![ezgif-7-13f70baab3](https://github.com/pankil-soni/movie-genre-predictor/assets/116267467/fe3d25b8-5ea9-4e31-b8ef-c6578f6879ac)


## Features

- Predict movie genre from both manually entered and pre-existing movie titles.
- User-friendly web interface with movie information display.
- Multiple machine learning models for genre prediction.
- genre predictions with high model accuracy.

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

- Clone the repository:

  ```
  git clone https://github.com/pankil-soni/movie-genre-predictor.git
  ```

  ```
  cd movie-genre-predictor
  ```

## Install dependencies:

- install the requirements
  ```
  pip install -r requirements.txt
  ```

## Usage

- Run the following command to start the Streamlit app:

  ```
  streamlit run app.py
  ```

## Models and Accuracy

- Logistic Regression :
  1. Train Accuracy - **94.69%**
  2. Test Accuracy - **51.59%**
- BernoulliNB :
  1. Train Accuracy - **91.34%**
  2. Test Accuracy - **52.72%**
- MultinomialNB :
  1. Train Accuracy - **93.19%**
  2. Test Accuracy - **51.65%**
- Random Forest :
  1. Train Accuracy - **100%**
  2. st Accuracy - **48.84%**
- Support Vector :
  1. Train Accuracy - **99.09%**
  2. Test Accuracy - **48.31%**

## Input Options

The application provides two options for input:

1. Search Movie: Choose from pre-existing movie titles with actual genre information.
2. Manually Enter Title and Description: Enter your own movie title and description for genre prediction.

## About the Project

This project is a Movie Genre Predictor web application made using Streamlit and deployed on Streamlit Server. The movie data is sourced from IMDB, providing genre information for predictions.

Made by **Pankil Soni**

## License

No license

## Contact

For inquiries or feedback, please contact pmsoni2016@gmail.com.
