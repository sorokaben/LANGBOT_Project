# LANGbot
### ASL hand sign translator bot

## Installation
No special installation instructions. Simply download all .py and .csv files.

## Description
This project trains a model using a dataset of about 6000 ASL hand sign images. The program takes these images and preprocesses them using 
mediapipe to extract 21 landmarks and their respective x and y coordinates. We then used this dataset of coordinates to train an AI model
to recognize static ASL hand signs. After training, the program takes video feed, preprocesses it through mediapipe, and gives the model 
the corresponding set of coordinates for each landmark, which it then uses to predict the most likely answer. 

## Purpose
The purpose of this project was to create a solution to translating ASL hand signs that was much more accessible to the average person.
By using this program, a deaf or hard-of-hearing person can easily communicate with a non-ASL speaker. Originally, this project
was meant to include real-time translation for full words and ASL motions, but due to limited knowledge the idea was modified to
this alternative. 

## Limitations
The current model is not capable of doing real-time ASL translations as it was not trained on the appropriate data and it is not 
sophisticated enough to do so. 

## Future Improvements
- Support for dynamic ASL gestures (not just static signs)  
- Improved  detection
- Expanded dataset for higher accuracy and robustness  
- Mobile or web-based deployment for broader accessibility  
