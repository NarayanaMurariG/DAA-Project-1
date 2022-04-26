# Image Quilting

Dependencies Needed
1. Matplotlib
2. OpenCv2
3. Numpy

##Description

Small blocks from the example image are replicated to the output image in image quilting. 
The initial block is copied at random, and then following blocks are placed in such a way
that they partially overlap with previously inserted pixel blocks. The width of the overlap
between old and new blocks is usually determined by the block size. To solve problem of 
choosing which pixels from the overlap region between the two overlap patches,
I have created a graph from the overlap region and generated the weights using the absolute
pixel differences of the two patches in overlap region and then performed a graph cut using 
Ford Fulkerson algorithm and then based on graph cut output, decided on which pixels go from which
patch and then copied those pixels to the final output.

##Commands to Run
How to run the application and arguments needed:
    ImageQuilting.py <path-to-sample-image> x y tileSize

Here, x, y are no of rows and columns of the final output image and they contain x*y image segments 
of tileSize (approximately)


Examples with actual files

python ImageQuilting.py test.png 3 3 20 T
<br>
python ImageQuilting.py tomato.png 4 4 50 T
<br>
python ImageQuilting.py nuts.png 5 5 30 T
<br>
python ImageQuilting.py nuts.png 4 4 60 T


#State files
All the intermediate adjacency matrices built produced are saved in a directory called
state_files and it is cleared before each run. It also stores which pixels are used from which overlap 

They are named by the iteration in which they are computed and are stored in image_files directory

All the intermediate image outputs are stored in image_files directory.

Note: Both state_files and image_files directories will be cleared before each run

## Resources used
1. https://www.geeksforgeeks.org/graph-and-its-representations/
2. GeeksForGeeks for Ford Fulkerson : https://www.geeksforgeeks.org/minimum-cut-in-a-directed-graph/?ref=lbp
3. https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf
4. 
