# Project report guide

Use this as a report outline and fill in results only after running your experiments.

## Title
Real-Time Facial Emotion Recognition Using Deep Learning

## Abstract
This project develops a computer-vision system for classifying visible facial expressions into seven FER-2013 categories using a convolutional neural network. The training pipeline performs image normalization and data augmentation, while early stopping and learning-rate reduction are used to control overfitting. A webcam application uses OpenCV to detect face regions and applies the trained model to each crop. Performance is evaluated using accuracy, per-class precision and recall, macro F1-score, and a confusion matrix. The system is a demonstration of expression classification and is not intended to infer a person's true emotional state or support high-stakes decisions.

## Chapters
1. Introduction and problem statement
2. Literature review (CNNs, facial expression datasets, face detection)
3. Requirements and system design
4. Dataset description and preprocessing
5. Proposed CNN architecture and training procedure
6. Implementation details
7. Experimental setup and results
8. Limitations, ethical considerations, and future work
9. Conclusion
10. References

## Problem statement
Manual observation of facial-expression images is subjective and difficult to scale. The project investigates whether a CNN can learn visual patterns from labeled images and classify facial-expression categories in a live camera application.

## Objectives
- Prepare labeled facial images for supervised learning.
- Design and train a CNN for seven-class classification.
- Evaluate performance on held-out data.
- Integrate inference with OpenCV face detection.
- Measure limitations and document reproducible experiments.

## Suggested experiments
- Baseline CNN versus the supplied CNN.
- With versus without data augmentation.
- Compare accuracy, macro F1, per-class recall, parameter count, and inference latency.
- Test varied lighting, head pose, occlusion, glasses, and camera quality.

## Results table (fill with your actual measurements)
| Metric | Measured result |
|---|---|
| Test accuracy | Fill after evaluation |
| Macro precision | Fill after evaluation |
| Macro recall | Fill after evaluation |
| Macro F1-score | Fill after evaluation |
| Average inference time/frame | Measure on your hardware |
| Training duration | Record from your run |

## Limitations and responsible use
FER-2013 labels are noisy and expressions are culturally and contextually variable. Performance may degrade with lighting, pose, occlusion, camera differences, and demographic differences. Softmax scores are not necessarily calibrated probabilities. Obtain consent before capturing people, avoid retaining images without permission, and do not use outputs for hiring, grading, policing, medical diagnosis, or other consequential decisions.

## Conclusion template
The project implemented a CNN-based facial-expression classification pipeline and connected it to a webcam interface. Summarize the measured test metrics, the strongest and weakest classes, observed real-time speed, and limitations. Do not claim a target accuracy unless supported by your actual evaluation.
