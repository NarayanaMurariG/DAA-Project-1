import numpy as np
from AdjacencyMatrix import AdjacencyMatrix
import cv2 as cv2
import matplotlib.pyplot as plt
import sys
import Util


class ImageQuilting:

    def __init__(self):
        self.res = None
        self.pos_matrix = None
        self.pos_matrix_map = None
        self.count = 1
        self.image_name = None

    # Generating a matrix with node postions
    def generatePos(self, height, width):
        self.pos_matrix = np.zeros([height, width], dtype=int)

        count = 1
        for i in range(width):
            for j in range(height):
                self.pos_matrix[j][i] = count
                count = count + 1

        return self.pos_matrix

    # Check pos of two indices to determine if they are in same row/column based on cut
    def checkPos(self, pos1, pos2, cut):

        if cut == "vertical":
            x1, y1 = pos1
            x2, y2 = pos2

            if x1 != x2:
                return False
            else:
                return True
        else:
            x1, y1 = pos1
            x2, y2 = pos2

            if y1 != y2:
                return False
            else:
                return True

        return False

    def get_pos_matrix_map(self):

        self.pos_matrix_map = {}

        for i in range(self.pos_matrix.shape[0]):
            for j in range(self.pos_matrix.shape[1]):
                pos = (i, j)
                val = self.pos_matrix[i][j]
                self.pos_matrix_map[val] = pos

        return self.pos_matrix_map


    # Getting random patch of patchlength
    def getRandomPatch(self, image, patchLength):
        height = image.shape[0]
        width = image.shape[1]
        i = np.random.randint(height - patchLength)
        j = np.random.randint(width - patchLength)

        return image[i:i + patchLength, j:j + patchLength]

    # Calculating error value for two patches
    def leastOverlapError(self, random_sample_patch, samplePatchDimensions, overlap, y, x):
        error = 0

        if x > 0:
            left = random_sample_patch[:, :overlap] - self.res[y:y + samplePatchDimensions, x:x + overlap]
            error = error + np.sum(left * left)

        if y > 0:
            up = random_sample_patch[:overlap, :] - self.res[y:y + overlap, x:x + samplePatchDimensions]
            error = error + np.sum(up * up)

        if x > 0 and y > 0:
            corner = random_sample_patch[:overlap, :overlap] - self.res[y:y + overlap, x:x + overlap]
            error = error - np.sum(corner * corner)

        return error

    # Getting the randomly picked patch of best match
    def getRandomBestPatch(self, image, samplePatchDimensions, overlap, y, x):
        height = image.shape[0]
        width = image.shape[1]

        errors = np.zeros((height - samplePatchDimensions, width - samplePatchDimensions))

        for i in range(height - samplePatchDimensions):
            for j in range(width - samplePatchDimensions):
                random_sample_patch = image[i:i + samplePatchDimensions, j:j + samplePatchDimensions]
                e = self.leastOverlapError(random_sample_patch, samplePatchDimensions, overlap, y, x)
                errors[i, j] = e


        # Now that we have computed the errors for different patches across the input image
        # We have to get the start index for the patch which has least error and it is
        # Acheived by getting the indices of min error in error matrix using argmin and
        # unravel_index numpy functions
        i, j = np.unravel_index(np.argmin(errors), errors.shape)
        return image[i:i + samplePatchDimensions, j:j + samplePatchDimensions]

    def plotGraph(self, flag):
        if flag:
            # plt.imshow(self.res)
            # plt.show()
            Util.save_plots(self.res,self.image_name+'-'+str(self.count))
            self.count = self.count + 1
        pass

    def get_final_overlap_patch(self, cut_pos, cut, left_overlap, right_overlap, i, j):
        right_copy = np.zeros([left_overlap.shape[0], left_overlap.shape[1], 3])
        right_copy[:, :] = [1, 0, 0]
        left_copy = right_copy.copy()
        left_copy[:, :] = [0, 0, 1]
        if cut == "vertical":

            for cut in enumerate(cut_pos):
                pos = cut[1].split("-")
                pos_left = self.pos_matrix_map[int(pos[0])]  # only interested in the column
                pos_right = self.pos_matrix_map[int(pos[1])]  # only interested in the column
                if self.checkPos(pos_left, pos_right, cut):  # Checking if they are in same row

                    x1, y1 = pos_left
                    x2, y2 = pos_right
                    #
                    right_copy[x1, :y1 + 1] = left_copy[x1, :y1 + 1]
                    #
                    right_overlap[x1, :y1 + 1] = left_overlap[x1, :y1 + 1]

            # Always send from right overlap as it is from the closest patch
            Util.save_image(right_copy, i, j)
            return right_overlap
        else:
            top_overlap = left_overlap
            bottom_overlap = right_overlap
            overlap = top_overlap.shape[1]

            for cut in enumerate(cut_pos):
                pos = cut[1].split("-")
                pos_top = self.pos_matrix_map[int(pos[0])]  # only interested in the column
                pos_bottom = self.pos_matrix_map[int(pos[1])]  # only interested in the column
                if self.checkPos(pos_top, pos_bottom, cut):  # Checking if they are in same row

                    x1, y1 = pos_top
                    x2, y2 = pos_bottom
                    #
                    right_copy[:x1 + 1, y1] = left_copy[:x1 + 1, y1]
                    #
                    bottom_overlap[:x1 + 1, y1] = top_overlap[:x1 + 1, y1]

            # Always send from bottom overlap as it is from the closest patch
            Util.save_image(right_copy, i, j)
            return bottom_overlap

    def graphMinCutPatch(self, patch, patchLength, overlap, y, x, i, j, flag):
        patch = patch.copy()

        if y == 0:  # First Row, simple vertical cuts
            left = self.res[:patchLength, x:x + patchLength]
            right = patch

            # First Fill The Non Overlapping part
            self.res[:patchLength, x + overlap:x + patchLength] = patch[:, overlap:]  # 0-17 20:37 Plotted Correctly

            left_overlap = left[:, patchLength - overlap:patchLength]
            right_overlap = right[:, :overlap]

            # Generating the position matrix for the nodes with node numbers
            self.pos_matrix = self.generatePos(left_overlap.shape[0], left_overlap.shape[1])
            # Creating a dict where keys are node numbers and values are its positions in the graph position matrix
            self.pos_matrix_map = self.get_pos_matrix_map()

            cut = "vertical"
            adjacencyMatrix = AdjacencyMatrix(left_overlap.shape[0], left_overlap.shape[1], self.pos_matrix)

            # Building an adjacency matrix and then calculating weights and performing graph cut
            # It returns the postions where the graph has to be cut
            cut_pos = adjacencyMatrix.calculateGraph(left_overlap, right_overlap, cut)
            matrix_to_print = adjacencyMatrix.adjacency_matrix
            Util.save_graph(matrix_to_print, i, j)

            # Getting the final overlap patch consisting of pixels from both first and second overlap
            final_overlap_patch = self.get_final_overlap_patch(cut_pos, cut, left_overlap, right_overlap, i, j)

            self.res[:patchLength, x:x + overlap] = final_overlap_patch[:, :]  # Copying final overlap patch to result
            self.plotGraph(flag)
        elif x == 0:  # First Column, simple horizontal cuts
            top_overlap = self.res[patchLength - overlap:patchLength, :patchLength]
            bottom_overlap = patch[:overlap, :]

            # First Fill The Non Overlapping part
            self.res[y + overlap:y + patchLength, :patchLength] = patch[overlap:, :]
            self.pos_matrix = self.generatePos(top_overlap.shape[0], top_overlap.shape[1])
            self.pos_matrix_map = self.get_pos_matrix_map()

            cut = "horizontal"
            adjacencyMatrix = AdjacencyMatrix(top_overlap.shape[0], top_overlap.shape[1], self.pos_matrix)

            # Building an adjacency matrix and then calculating weights and performing graph cut
            # It returns the postions where the graph has to be cut
            cut_pos = adjacencyMatrix.calculateGraph(top_overlap, bottom_overlap, cut)
            matrix_to_print = adjacencyMatrix.adjacency_matrix
            Util.save_graph(matrix_to_print, i, j)

            # Getting the final overlap patch consisting of pixels from both first and second overlap
            final_overlap_patch = self.get_final_overlap_patch(cut_pos, cut, top_overlap, bottom_overlap, i, j)

            self.res[y:y + overlap:, :patchLength] = final_overlap_patch[:, :]  # Copying final overlap patch to result
            self.plotGraph(flag)

            # Working till here
        else:
            # Do vertical for full length
            left = self.res[y:y + patchLength, x:x + patchLength]
            right = patch

            left_overlap = left[:, patchLength - overlap:patchLength]
            right_overlap = right[:, :overlap]

            self.pos_matrix = self.generatePos(left_overlap.shape[0], left_overlap.shape[1])
            self.pos_matrix_map = self.get_pos_matrix_map()

            cut = "vertical"
            adjacencyMatrix = AdjacencyMatrix(left_overlap.shape[0], left_overlap.shape[1], self.pos_matrix)

            # Building an adjacency matrix and then calculating weights and performing graph cut
            # It returns the postions where the graph has to be cut
            cut_pos = adjacencyMatrix.calculateGraph(left_overlap, right_overlap, cut)
            matrix_to_print = adjacencyMatrix.adjacency_matrix
            Util.save_graph(matrix_to_print, i, j)

            # Getting the final overlap patch consisting of pixels from both first and second overlap
            final_overlap_patch = self.get_final_overlap_patch(cut_pos, cut, left_overlap, right_overlap, i, j)

            self.res[y:y + patchLength, x:x + overlap] = final_overlap_patch[:,:]  # Copying final overlap patch to result

            # First Fill The Non Overlapping part
            self.res[y + overlap:y + patchLength, x + overlap:x + patchLength] = patch[overlap:, overlap:]
            self.plotGraph(flag)

        return patch

    def myQuiltFunction(self, image, samplePatchDimensions, noOfRows, noOfColumns, flag):
        overlap = samplePatchDimensions // 6

        height = (noOfRows * samplePatchDimensions) - (noOfRows - 1) * overlap
        width = (noOfColumns * samplePatchDimensions) - (noOfColumns - 1) * overlap

        self.res = np.zeros((height, width, image.shape[2]))


        # Looping over rows and columns and choosing random best patch compared to resultant and then
        # Joining both the patches using graphcuts
        for i in range(noOfRows):
            for j in range(noOfColumns):
                y = i * (samplePatchDimensions - overlap)  # Row start for patch
                x = j * (samplePatchDimensions - overlap)  # Column start for patch

                if i == 0 and j == 0:
                    # First row first column, no need of any graph cut, can use as it is
                    patch = self.getRandomPatch(image, samplePatchDimensions)
                    self.res[y:y + samplePatchDimensions, x:x + samplePatchDimensions] = patch
                else:
                    #Getting the best random patch for the resultant
                    patch = self.getRandomBestPatch(image, samplePatchDimensions, overlap, y, x)
                    self.graphMinCutPatch(patch, samplePatchDimensions, overlap, y, x, i, j, flag)

        return self.res

    def start(self, x, y, path, tileSize, printEveryPhase):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        plt.imshow(image)
        plt.show()

        self.myQuiltFunction(image, tileSize, x, y, printEveryPhase)
        plt.imshow(self.res)
        plt.show()


if __name__ == "__main__":
    print('Number of arguments:', len(sys.argv), 'arguments.')
    if len(sys.argv) < 6:
        print("Invalid Arguments")
        sys.exit()
    Util.clear_state_files()
    image_path = sys.argv[1]  # Image Path
    x = int(sys.argv[2])  # No of rows of tiles
    y = int(sys.argv[3])  # No of columns of tiles
    tile_size = int(sys.argv[4])

    print_intermediate_states = sys.argv[5]
    flag = False
    if print_intermediate_states == 'T':
        flag = True

    imageQuilting = ImageQuilting()
    imageQuilting.image_name = image_path.split('.')[0]
    print(" Program Running......")
    imageQuilting.start(x, y, image_path, tile_size, flag)

    print(" Program Completed......")
    print("States saved in state_files directory")
    print("Images saved in image_files directory")
