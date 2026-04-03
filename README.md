# LANGbot
### ASL hand sign translator bot

## Installation
No special installation instructions. Simply download all .py and .csv files.

## Description
This project takes static images of ASL hand signs, puts them through mediapipe, which detects 21 hand landmarkers and their
x and y coordinates. These coordinates are then given to the model to interpret the corresponding English alphabet letter. 
Using Gemini, the program will be able to detect once a full word has been translated, which it will then hand off to a text-to-speech
portion of the program.

## Purpose
The purpose of this project was to create a solution to translating ASL hand signs that was much more accessible to the average person.
By using this program, a deaf or hard-of-hearing person can easily communicate with a non-ASL speaker. Originally, this project
was meant to include real-time translation for full words and ASL motions, but due to limited knowledge the idea was modified to
this alternative. 

## Future Improvements
- Support for dynamic ASL gestures (not just static signs)  
- Improved  detection
- Expanded dataset for higher accuracy and robustness  
- Mobile or web-based deployment for broader accessibility  
