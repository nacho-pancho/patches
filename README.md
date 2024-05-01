# Patch extraction and stitching

This little project is about extracting patches from 2D images and stitching them back in an efficient way.
It supports masks so that the only patches that are extracted are those that "touch" the mask.
It is a stripped down version of a more complex project I did on inpainting a while ago.
It supports memory mapping, which is nice if you have many images and you want to not fill your memory with tons of patches

The file `patch_extraction_in_python.py` does the same, but in Python. It is slower but works, if you don't want to compile the C version.
BTW, to compile you need to run:
```
python3 setup.py develop
```

