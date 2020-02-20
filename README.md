# eye-contact-cnn
This repository provides a deep convolutional neural network model trained to detect moments of eye contact in egocentric view. The model was trained on over 4 millions of facial images of > 100 young individuals during natural social interactions, and achives an accuracy comaprable to that of trained clinical human coders.


## To run
### Running on Webcam
```
python demo.py
```
### Running on Video
```
python demo.py --video yourvideofile.avi
```

Hit 'q' to quit the program.


## Flags
- `--face`: Path to pre-processed face detection file of format [frame#, min_x, min_y, max_x, max_y]. If not specified, dlib's face detector will be used.
- `-save_vis`: Saves the output as an avi video file.
- `-save_text`: Saves the output as a text file (Format: [frame#, eye_contact_score]).
- `-display_off`: Turn off display window.


## Notes
- Output eye contact score ranges [0, 1] and score above 0.9 is considered confident.
- To further improve the result, smoothing the output is encouraged as it can help removing outliers caused by eye blinks, motion blur etc.
- Code has been tested with PyTorch 0.4 and Python 2.7
