# Guibbon with U
![Tests](https://github.com/ManuGira/Guibbon/actions/workflows/tests.yml/badge.svg)

High-level GUI with an API similar to the HighGUI of OpenCV. It allows to display an image and add GUI controllers such as
 - Sliders (trackbar)
 - Buttons
 - Radio buttons
 - Check boxes
 - Color picker
 - Draggable points and polygons on the displayed image
 - Any custom widget that you write in Tkinter

If you know how to use the GUI of OpenCV, then you already know how to use Guibbon. 

Reasons why you want to use Guibbon:
 - It's using Tkinter which is natively installed in your python distribution
 - Beside Tkinter, it only has 3 dependencies: numpy, opencv-python and pillow
 - You just need to display an image and add a few controllers to play with parameters


## User Installation
This package is hosted on PyPl, you can install the latests stable release with pip:
```
pip install guibbon
```
If you feel adventurous you can also install the most recent commit of this repo:
```
pip install git+https://github.com/ManuGira/Guibbon.git@master
```
## Development 

The project is configured in `pyproject.toml`. It contains dependecies, and configs for continuous integration.

### Dev Installation
Install development requirements with poetry:
```
$ poetry install
```
This will create a new venv for this project and install dependencies according to poetry.lock. If you prefer to manage your venv yourself, you can install with pip:
```
$ pip install -r requirements_dev.txt
$ pip install -e .  # Install this package in editable mode
``` 
All requirements files and poetry.lock have been generated with the `prepare_python.sh` script.
```
./prepare_python.sh
```

### Continuous Integration
#### Testing with pytest 
Run tests and generate coverage report:
```
$ pytest 
```
#### Type checking with mypy
Run it with:
```
$ mypy .
```
#### Lintering with ruff
Linter check:
```
$ ruff check .
```
Linter fix:
```
$ ruff check --fix .
```

### Publishing to PyPI with poetry
First, update the version number in the `pyproject.toml`  
Then build tarball and wheel:
```
$ poetry build
```
Publish to PyPI:
```
$ poetry publish -r pypi -u __token__ -p <paste the secret token here (very long string starting with "pypi-")>
```

## TODO

#### Image Viewer
* **Feature**: Handle double clicks
* **Feature**: Handle **param** field of mouse callback
* **Feature**: Scroll delta is missing
* **Feature**: Mac support

#### Demos
* **Feature**: Make sure that the failing demo also fails with cv2

#### Interactive Overlays
* **Feature**: Make sure that the failing demo also fails with cv2


## Documentation

#### Class hierarchy
* Guibbon
  * keyboard_event_hander: *static KeyboardEventHandler*
  * root: *static tk.Tk*
  * self.window: *tk.Frame*
  * frame: *tk.Frame*
  * ctrl_frame: *tk.Frame*
  * image_viewer: *ImageViewer*
    * canvas: *tk.Canvas*
    * imgtk: *PIL.ImageTk*
    * interactive_overlays: *List(InteractiveOverlay)*
      * canvas*
        * circle_id: *id of canvas oval*
        * 

#### Tkinter hierarchy
* Guibbon.root: *static Tk*
  * guibbon.window: *TopLevel*
    * tkcv2.frame: *Frame*
      * tkcv2.ctrl_frame: *Frame*
        * controller: *Frame*
        * controller: *Frame*
        * ...
      * tkcv2.image_viewer.canvas: *Canvas*
        * tkcv2.image_viewer.imgtk: *PIL.ImageTk*
