# eye-contact-cnn
This repository provides a deep convolutional neural network model trained to detect moments of eye contact in egocentric view. The model was trained on over 4 millions of facial images of > 100 young individuals during natural social interactions, and achives an accuracy comaprable to that of trained clinical human annotators.

![](teaser.gif)

## Libraries used in our experiment
- PyTorch 0.4.0
- opencv 4.0.0
- numpy 1.16.2
- PIL 5.3.0
- pandas 0.23.4
- dlib 19.13.0 (optional if you want live face detection)


## To run
### with a webcam
```
python demo.py
```
### on a video file
```
python demo.py --video yourvideofile.avi
```
### on our sample video
Try this if you don't want to use dlib's face and instead test with pre-detected faces.

Comment out the first line of demo.py "import dlib" if you didn't install dlib.

```
python demo.py --video demo_video.avi --face demo_face_detections.txt
```

Demo video has been downloaded from [here](https://youtu.be/5wFyxihwQiI). I used this [face detector](https://github.com/natanielruiz/dockerface) to generate the face detection file.


## Flags
- `--face`: Path to pre-processed face detection file of format [frame#, min_x, min_y, max_x, max_y]. If not specified, dlib's face detector will be used.
- `-save_vis`: Saves the output as an avi video file.
- `-save_text`: Saves the output as a text file (Format: [frame#, eye_contact_score]).
- `-display_off`: Turn off display window.
- Hit 'q' to quit the program.


## Notes
- Output eye contact score ranges [0, 1] and score above 0.9 is considered confident.
- To further improve the result, smoothing the output is encouraged as it can help removing outliers caused by eye blinks, motion blur etc.


## Citation
Please cite this paper in any publications that make use of this software.

```
@article{chong2020,
 title={Detection of eye contact with deep neural networks is as accurate as human experts},
 url={osf.io/5a6m7},
 DOI={10.31219/osf.io/5a6m7},
 publisher={OSF Preprints},
 author={Chong, Eunji and Clark-Whitney, Elysha and Southerland, Audrey and Stubbs, Elizabeth and Miller, Chanel and Ajodan, Eliana L and Silverman, Melanie R and Lord, Catherine and Rozga, Agata and Jones, Rebecca M and et al.},
 year={2020}
}
```

Link to the paper:
[here](https://nature-research-under-consideration.nature.com/users/37265-nature-communications/posts/60730-detection-of-eye-contact-with-deep-neural-networks-is-as-accurate-as-human-experts)
