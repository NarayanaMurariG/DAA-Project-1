# Image Quilting

Dependencies Needed
1. Matplotlib
2. OpenCv2
3. Numpy

## Description

Small blocks from the example image are replicated to the output image in image quilting. 
The initial block is copied at random, and then following blocks are placed in such a way
that they partially overlap with previously inserted pixel blocks. The width of the overlap
between old and new blocks is usually determined by the block size used while replication. 
To solve problem of choosing which pixels from the overlap region between the two overlap patches,
I have created a graph from the overlap region and generated the weights using the absolute
pixel differences of the two patches in overlap region and then performed a graph cut using 
Ford Fulkerson algorithm and then based on graph cut output, decided on which pixels go from which
patch and then copied those pixels to the final output.

## Installing the dependencies

All the required dependencies are in the Requirements.txt file and can be 
installing using the pip install command.

## Statement of help
All the sample outputs and intermediate outputs generated previously
are stored in IntermediateOutputs directory.

All the working screenshots and output images are located in a directiry called WorkingScreenshots

When application is executed all the adjacency matrices and seam cuts are 
stored in the state_files directory, which will be created and cleared when the 
script is run.

Similarly, all the intermediate outputs for a particular run will be stored in
image_files directory, which will be created and cleared when the script is run.

Only the inital input image and final output image are displayed as plots.

All the python scripts and classes are located in the src folder

How to run the application and arguments needed:
    **ImageQuilting.py <path-to-sample-image> x y tileSize**

Here, x, y are no of rows and columns of the final output image and they contain x*y image segments 
of tileSize (approximately)


Examples with actual files

**python ImageQuilting.py test.png 3 3 20**
<br>
**python ImageQuilting.py tomato.png 4 4 50**
<br>
**python ImageQuilting.py nuts.png 5 5 30**

The output is as follows: 

(venv) narayanamurari@MacBook-Pro-16 src % python ImageQuilting.py test.png 3 3 20
<br>
Number of arguments: 5 arguments.
<br>
 Program Running......
<br>
 Program Completed......
<br>
States saved in state_files directory
<br>
Images saved in image_files directory
<br>


## State files
All the intermediate adjacency matrices built produced are saved in a directory called
state_files and it is cleared before each run. It also stores which pixels are used from which overlap and
they are named by the iteration in which they are computed and are stored in state_files directory

All the intermediate image outputs are stored in image_files directory.

Note: Both state_files and image_files directories will be cleared before each run

## Resources and References used
1. https://www.geeksforgeeks.org/graph-and-its-representations/
2. https://www.geeksforgeeks.org/minimum-cut-in-a-directed-graph/?ref=lbp
3. https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf
4. https://github.com/axu2/image-quilting
