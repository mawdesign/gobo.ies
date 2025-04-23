# gobo.ies

## Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)

## Introduction

The `gobo.ies` project is a Python-based tool that generates IES (Illuminating Engineering Society) files from image 
inputs. IES files are a standard format used to represent the photometric distribution of a light source, essentially 
describing how the light is emitted. This tool allows users to create these files from images, offering a way to derive 
light emission characteristics from visual data. It leverages the `polarTransform` library to convert the image into a 
polar coordinate system, which is essential for representing the light distribution around a source. The script then 
calculates the luminous intensity based on the pixel values in the transformed image and generates the IES file.

## Installation

To install the required dependencies, use pip:

```bash
pip install -r requirements.txt
```

This will install the following packages:
* polarTransform
* Pillow
* numpy

## Usage

1. Ensure you have the required libraries installed (see [Installation](#installation)).
2. Run the `image2ies.py` script, providing the path to your image file, the desired beam angle, and the maximum candela 
   value.  You can also optionally specify the output filename, and the angle steps for theta and phi.
   ```bash
   python src/image2ies.py <image_path> <beam_angle_deg> <max_candela> [<filename>] [<theta_step>] [<phi_step>]
   ```
   * `image_path`: Path to the input image file (e.g., JPEG, PNG).
   * `beam_angle_deg`: The beam angle of the light source in degrees.
   * `max_candela`: The maximum luminous intensity (candela) of the light source.
   * `filename` (optional): The name of the output IES file. Defaults to "output.ies".
   * `theta_step` (optional): Step size for vertical angles (theta) in degrees. Defaults to 1.0 degree.
   * `phi_step` (optional): Step size for horizontal angles (phi) in degrees. Defaults to 5.0 degrees.
   For example:
   ```bash
   python src/image2ies.py example_image.jpg 60.0 1000.0 my_light.ies 5.0 10.0
   ```
   This will generate an IES file named `my_light.ies` based on the image `example_image.jpg`, with a beam angle of 60 
   degrees and a maximum candela of 1000.

## Contributing

We welcome contributions to this project! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Ensure your code follows the project's style guidelines (Black).
5. Submit a pull request

## License

This project is released under the Creative Commons Zero v1.0 Universal license. See the [LICENSE](LICENSE) file for more 
information.
