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


## Citation
Please cite this paper in any publications that make use of this software.

```
@article{chong2017detecting,
  title={Detecting gaze towards eyes in natural social interactions and its use in child assessment},
  author={Chong, Eunji and Chanda, Katha and Ye, Zhefan and Southerland, Audrey and Ruiz, Nataniel and Jones, Rebecca M and Rozga, Agata and Rehg, James M},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={1},
  number={3},
  pages={43},
  year={2017},
  publisher={ACM}
}
```

Link to the paper:
[Detecting gaze towards eyes in natural social interactions and its use in child assessment](https://arxiv.org/pdf/1902.00607.pdf)
