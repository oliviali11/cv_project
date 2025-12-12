## CS153 Computer Vision - Final Project

This folder contains our final project for CS153 Computer Vision at Harvey Mudd College.

### **Folder Structure and Contents**

Preprocessing data

- **organize.py/**
  - places all validation image data into corresponding WNID labeled folders based on ground truth class indices

Running models (Goes through 4 attacks per model and using 25 batches)

- **resnet.py/**
  - Run with python3 resnet.py > resnet.txt for robustness accuracy results
- **deit.py/**
  - Run with python3 resnet.py > resnet.txt for robustness accuracy results
- **convit.py/**
  - Run with python3 resnet.py > resnet.txt for robustness accuracy results
- **convnext.py/**
  - Run with python3 resnet.py > resnet.txt for robustness accuracy results

Analysis - **plot.py/** - Creates plot of model architecture versus robustness accuracy for the 4 attacks (in legend)
