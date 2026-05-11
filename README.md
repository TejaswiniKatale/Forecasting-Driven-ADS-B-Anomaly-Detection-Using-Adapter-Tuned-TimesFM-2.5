# Low-Altitude ADS-B Anomaly Detection with Adapter-Tuned TimesFM 2.5

## Overview

This project presents a forecasting-driven anomaly detection framework for low-altitude ADS-B aircraft trajectories using Google's TimesFM 2.5 time-series foundation model. The main goal is to detect cyberattack-induced anomalies in aircraft trajectory data by learning normal flight behavior and identifying abnormal deviations caused by different attack scenarios.

Automatic Dependent Surveillance–Broadcast, or ADS-B, is widely used in modern aviation surveillance. Aircraft continuously broadcast information such as position, altitude, velocity, heading, and vertical rate. However, because ADS-B messages are often unauthenticated and openly broadcast, they are vulnerable to cyberattacks such as spoofing, replay, saturation, and trajectory manipulation. These attacks can affect situational awareness, air traffic monitoring, and aviation safety.

This project investigates whether a time-series foundation model can be adapted for ADS-B anomaly detection, especially in low-altitude flight trajectories where cyber-induced deviations may have significant safety implications. The framework uses TimesFM 2.5 as the forecasting backbone and explores adapter tuning with a CNN-based classifier head to detect abnormal trajectory behavior across attack-specific scenarios.

---

## Project Motivation

Traditional ADS-B anomaly detection methods often rely on hand-crafted rules, statistical thresholds, or supervised deep learning models trained from scratch. These approaches may struggle when attack patterns vary across flights, time windows, or trajectory conditions.

This project is motivated by three key ideas:

1. **ADS-B trajectories are time-series data**  
   Aircraft movement evolves over time, so anomaly detection should capture temporal patterns instead of treating each message independently.

2. **Cyberattacks create abnormal trajectory deviations**  
   Spoofing, replay, saturation, and interpolation attacks can change the expected behavior of flight features such as altitude, speed, heading, and vertical rate.

3. **Time-series foundation models can improve generalization**  
   TimesFM 2.5 is pretrained for time-series forecasting, making it suitable for learning normal temporal behavior and detecting deviations from expected patterns.

---

## Objectives

The main objectives of this project are:

- Develop a forecasting-driven anomaly detection pipeline for ADS-B aircraft trajectory data.
- Use Google's TimesFM 2.5 as a time-series forecasting backbone.
- Detect multiple cyberattack types, including spoofing, replay, saturation, and interpolation attacks.
- Evaluate anomaly detection performance using attack-specific metrics.
- Explore adapter tuning and CNN-based classification for efficient model adaptation.
- Compare message/window-level and flight/attack-level performance separately.
- Build a reproducible pipeline for preprocessing, prediction generation, and consolidated evaluation.

---

## Dataset

The project uses ADS-B aircraft trajectory data from OpenSky-style flight records. Each flight contains time-ordered aircraft state information.

Typical features used in the dataset include:

- `flight_id`
- `time`
- `icao24`
- `callsign`
- `lat`
- `lon`
- `velocity`
- `heading`
- `vertrate`
- `baroaltitude`
- `geoaltitude`
- `altitude_ft_used`
- `onground`
- `alert`
- `spi`
- `squawk`

The project focuses on **low-altitude trajectories**, where anomaly detection is especially important for aviation safety and cyber-resilience analysis.

---

## Attack Scenarios

This project evaluates multiple ADS-B cyberattack scenarios.

### 1. Spoofing Attack

A spoofing attack injects false aircraft trajectory information into the ADS-B data stream. This can create misleading aircraft positions or movement patterns.

### 2. Replay Attack

A replay attack repeats previously observed ADS-B messages. This can make an aircraft appear to follow an old or duplicated trajectory.

### 3. Saturation Attack

A saturation attack introduces excessive or abnormal message patterns that can disrupt normal surveillance interpretation.

### 4. Interpolation Attack

An interpolation attack modifies trajectory values smoothly between points, making the anomaly harder to detect because the trajectory may still appear realistic.

---

## Methodology

The proposed framework follows a forecasting-driven anomaly detection approach.

### Step 1: Data Preprocessing

Raw ADS-B trajectory data is cleaned, filtered, and organized by flight. The preprocessing stage includes:

- Sorting messages by flight and timestamp.
- Filtering low-altitude aircraft trajectories.
- Selecting relevant time-series features.
- Handling missing or invalid values.
- Preparing normal and attacked trajectory sequences.
- Creating consistent train and evaluation splits.

### Step 2: Window-Based Time-Series Construction

Each flight trajectory is divided into fixed-length time-series windows. These windows are used as the basic unit for forecasting and classification.

A window-based approach helps the model capture short-term temporal behavior in aircraft movement.

Example features inside each window may include:

- Latitude
- Longitude
- Altitude
- Velocity
- Heading
- Vertical rate

### Step 3: TimesFM 2.5 Forecasting Backbone

TimesFM 2.5 is used as the time-series foundation model. The model forecasts future trajectory behavior from historical time-series windows.
Past ADS-B trajectory window → TimesFM forecast → Expected future behavior
