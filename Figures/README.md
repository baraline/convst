Those script containt the methods used for generating figures present in the paper. 
If you wish to use some of them, modify the base path and any path variable that is present in the start of each script

## How to compile scene animations
To compile the scene animations, you need to install [ManimCE](https://docs.manim.community/en/stable/installation.html) and its requirerments.
Once this is done, you can launch the compilation in either high quality with the `-pqh` command, or low quality with the `-pql`command as : 

```bash
manim -pql scene.py Slide1 
```
This will compile and play the first class `Slide1` into a mp4 file, you can compile all the animations this way (from `Slide1` to `Slide7`)
