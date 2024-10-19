# Optimization of Piezoelectric Energy Harvesting Systems (Stun Shoe)

This project aims to optimize the energy harvesting capability of piezoelectric systems embedded in shoes (Stun Shoe). By utilizing machine learning models, the system predicts optimal configurations and adjustments to maximize energy harvested from footfalls. The hardware collects data related to the energy output and factors influencing it, and the ML models analyze and optimize the process.


## Table of Contents
- [Introduction](#introduction)
- [Objective](#objective)
- [System Overview](#system-overview)
- [Machine Learning Models](#machine-learning-models)
- [Data Collection](#data-collection)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Introduction
In my first year of college, I initiated a project titled Stun Shoe, which I led in collaboration with two fellow freshmen. This endeavor was inspired by the troubling news articles I frequently encountered regarding the exploitation of women, motivating me to create a meaningful impact.

Stun Shoe employed piezoelectric cells that functioned in conjunction with a machine learning model to optimize energy harvesting. This marked my initial exposure to machine learning, and I was captivated by how a straightforward XGBoost model could analyze over 120,000 data points to predict and enhance the final energy output from the piezoelectric discs. One significant challenge we faced was the presence of noise in the footfall data. I recognized that these patterns varied based on distinct activities, points of impact, and surfaces. To address this, we innovatively applied Reinforcement Learning techniques utilizing DQN and PPO algorithms.

Our project garnered recognition at my department's expo, where we achieved first runner-up. This experience profoundly deepened my interest in machine learning, illuminating its vast potential and diverse applications.

Piezoelectric materials generate electricity when subjected to mechanical stress, such as walking or running. The **Stun Shoe** project incorporates piezoelectric systems into footwear, aiming to optimize the amount of energy harvested from each footfall using advanced machine learning techniques. This optimization seeks to improve the efficiency of energy harvesting systems in practical, real-world scenarios.

## Objective
The primary goal of this project is to:
- Develop an energy-efficient piezoelectric harvesting system embedded in shoes.
- Create a machine learning model that can predict and optimize the energy output based on various parameters such as stride length, force, material properties, and frequency of steps.

## System Overview
The system consists of:
1. **Piezoelectric Sensors**: Embedded in the shoe soles to capture mechanical energy.
2. **Data Acquisition System**: Collects real-time data such as footfall impact, voltage output, and other variables.
3. **Machine Learning Model**: Predicts optimal configurations for maximizing energy harvested.
4. **Energy Storage System**: Stores the generated energy efficiently.

## Machine Learning Models
To optimize the energy harvesting process, we implemented and tested several machine learning models, including:
- **Random Forest Regression**: Used for predicting energy output based on various footfall characteristics.
- **K-Nearest Neighbors (KNN)**: Applied to identify patterns in data related to walking habits.
- **Support Vector Machines (SVM)**: Utilized to classify different walking patterns and their effect on energy output.
- **ARIMA**: Analyzed to predict future footfalls and their energy potential.
- **Gradient Boosting (XGBoost)**: Fine-tuned for maximizing the accuracy of the energy predictions.
- **Neural Networks**: For deep learning-based predictions based on complex input patterns.

## Data Collection
Data collection was conducted using:
- Footfall impact (force, frequency)
- Walking speed
- Step length
- Piezoelectric voltage output
- Material properties of the piezoelectric system

The data was collected over several walking trials and was preprocessed for training and validation of the machine learning models.

## Results
After training and testing the machine learning models, we achieved:
- **37%** improvement in energy harvesting efficiency over baseline.
- **83%** prediction accuracy for footfall energy output.
- Real-time adaptability to different walking styles and materials.

## Future Work
Future improvements include:
- Expanding the dataset to include a broader range of users with different walking styles.
- Improving the real-time performance of the system for practical deployment.
- Integrating the system with smart devices for continuous monitoring and energy storage.

## Acknowledgments
This project was developed as part of the **Stun Shoe** initiative. Special thanks to all contributors who provided valuable feedback and resources to make this project successful.

---

<i>Feel free to reach out for any questions or collaboration opportunities.</i>
