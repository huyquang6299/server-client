import cv2
import math
import numpy as np

# import subprocess          #Linux/Ubuntu system sideload lib

min_point = ()
max_point = ()


def mouseevent(image, scale):
    # Function: click_and_crop
    def click_and_crop(event, x, y, flags, param):
        global min_point, max_point
        # Mouse-click activate
        if event == cv2.EVENT_LBUTTONDOWN:
            min_point = (x, y)
        # Mouse-release activate
        elif event == cv2.EVENT_LBUTTONUP:
            max_point = (x, y)
            # draw a rectangle around the region of interest
            cv2.rectangle(image, min_point, max_point, (0, 255, 0), 2)
            cv2.imshow('image', image)

    ###

    # Main function:

    # Open storage file, Slot coordinates & Landmark coordinates:
    f1 = open('Slot coordinate.txt', 'w+')
    f2 = open('Mark coordinate.txt', 'w+')

    # Draw text box, describing the functions
    cv2.rectangle(image, (1480, 940), (1910, 1070), (255, 255, 255), -1)
    cv2.rectangle(image, (1480, 940), (1910, 1070), (0, 127, 0), 3)
    text_01 = "Drag mouse over region of interest"
    text_02 = "Press M to save to Landmarks"
    text_03 = "Press S to save to Parking Slots"
    text_04 = "Press B to end the program"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text_01, (1495, 965), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, text_02, (1515, 995), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image, text_03, (1515, 1025), font, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, text_04, (1515, 1055), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    ##

    # Display GUI, set click_and_crop
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    slotNumber = 0
    image = cv2.resize(image,dsize=None,fx=scale,fy=scale)
    imageCopy = image.copy()
    while True:
        # Display the image for selection purpose only (used for PyCharm)
        # image = cv2.resize(image, (scrWidth, scrHeight))
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # If the 'b' key is pressed, break from the program
        if key == ord("q"):
            break
        # If the 's' key is pressed, write the coordinates of SLOTS' vertices to txt file
        elif key == ord("c"):
            image = imageCopy.copy()
        elif key == ord("s"):
            f1.write('%d ' % slotNumber)
            slotNumber = slotNumber + 1
            f1.write('%d %d ' % (int(min_point[0] / scale ), int(min_point[1] / scale)))
            f1.write('%d %d\n' % (int(max_point[0] / scale), int(max_point[1] / scale)))
        # If the 'm' key is pressed, write the coordinates of LANDMARKS' centroids to txt file
        elif key == ord("m"):
            f2.write('%d ' % (int((min_point[0] + max_point[0]) / 2 / scale)))  # 1 2   # Left to right
            f2.write('%d ' % (int((min_point[1] + max_point[1]) / 2 / scale)))  # 3 4   # Top to bottom
    ###
    # Close files, kill GUI
    f1.close()
    f2.close()
    cv2.destroyAllWindows()


def perspectivewarp(im):

    def rotate(angle, center=None, scale=1.0):
        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return M

    def My2pointWarp(img, ref_x1, ref_y1, ref_x2, ref_y2, cur_x1, cur_y1, cur_x2, cur_y2):
        cur_copy = img.copy()
        # Find center points
        ref_mid_x = (ref_x1 + ref_x2) // 2;
        ref_mid_y = (ref_y1 + ref_y2) // 2
        cur_mid_x = (cur_x1 + cur_x2) // 2;
        cur_mid_y = (cur_y1 + cur_y2) // 2

        vector_1 = [ref_x2 - ref_x1, ref_y2 - ref_y1]
        vector_2 = [cur_x2 - cur_x1, cur_y2 - cur_y1]

        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)

        # Translate to match center points
        T = np.float32([[1, 0, ref_mid_x - cur_mid_x], [0, 1, ref_mid_y - cur_mid_y]])
        # Scale
        line_1 = math.sqrt((ref_x2 - ref_x1) ** 2 + (ref_y2 - ref_y1) ** 2)
        line_2 = math.sqrt((cur_x2 - cur_x1) ** 2 + (cur_y2 - cur_y1) ** 2)
        M = rotate(-180 * angle / 3.1415926, center=(ref_mid_x, ref_mid_y), scale=line_1 / line_2)
        T = np.vstack((T, [0, 0, 1]))
        M = np.vstack((M, [0, 0, 1]))
        C = np.dot(M, T)
        out = cv2.warpAffine(cur_copy, C[0: 2, :], (cur_copy.shape[1], cur_copy.shape[0]))
        return out, C[0: 2, :]

    ref_x1 = 453;
    ref_y1 = 568
    ref_x2 = 1107;
    ref_y2 = 573
    ref_x3 = 213;
    ref_y3 = 836
    ref_x4 = 1227;
    ref_y4 = 860

    # Read slots and landmarks coordinates from files -------
    f1 = open('Slot coordinate.txt', 'r')
    f2 = open('Mark coordinate.txt', 'r')
    # result = open('old.Result1.txt', 'w+')

    # Read landmarks' coordinates
    for line in f2:
        ref_x1 = int(float(line.split()[0]))
        ref_y1 = int(float(line.split()[1]))
        ref_x2 = int(float(line.split()[2]))
        ref_y2 = int(float(line.split()[3]))
        ref_x3 = int(float(line.split()[4]))
        ref_y3 = int(float(line.split()[5]))
        ref_x4 = int(float(line.split()[6]))
        ref_y4 = int(float(line.split()[7]))

    cur_x1_tmp = 0;
    cur_y1_tmp = 0
    cur_x2_tmp = 0;
    cur_y2_tmp = 0
    cur_x3_tmp = 0;
    cur_y3_tmp = 0
    cur_x4_tmp = 0;
    cur_y4_tmp = 0

    M_tmp = 0
    M_4point = 0;
    M_3point = 0

    cur_x1 = ref_x1;
    cur_y1 = ref_y1
    cur_x2 = ref_x2;
    cur_y2 = ref_y2
    cur_x3 = ref_x3;
    cur_y3 = ref_y3
    cur_x4 = ref_x4;
    cur_y4 = ref_y4

    out_tmp = 0

    im_lower = np.array([20, 75, 75], dtype="uint8")  # Lower threshold
    im_upper = np.array([35, 255, 255], dtype="uint8")  # Upper threshold
    kernel = np.ones((3, 3), np.uint8)

    # index = 0

    for filename in range(0, 1):
        # index = index + 1

        # im = cv2.imread(os.path.join('./DATA/201230_rename', filename))

        #  Landmarks detection and warping ############
        im_copy = im
        im_hsv = cv2.cvtColor(im_copy, cv2.COLOR_BGR2HSV)  # Convert image from RGB space to HSV space
        im_mask = cv2.inRange(im_hsv, im_lower, im_upper)  # Create mask
        im_mask = cv2.morphologyEx(im_mask, cv2.MORPH_OPEN, kernel)

        _,cur_cnt, _ = cv2.findContours(im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cur_1 = 0
        cur_2 = 0
        cur_3 = 0
        cur_4 = 0

        for c in cur_cnt:
            x, y, w, h = cv2.boundingRect(c)
            if (280 < (x + (w // 2)) < 630) and (450 < (y + (h // 2)) < 700) and (15 < w < 70) and (
                    7 < h < 60):  # Defined range for landmarks 1
                cur_x1 = x + (w // 2)
                cur_y1 = y + (h // 2)
                cur_1 = cur_1 + 1
            elif (920 < (x + (w // 2)) < 1300) and (460 < (y + (h // 2)) < 700) and (15 < w < 70) and (
                    7 < h < 60):  # Defined range for landmarks 2
                cur_x2 = x + (w // 2)
                cur_y2 = y + (h // 2)
                cur_2 = cur_2 + 1
            elif (50 < (x + (w // 2)) < 450) and (700 < (y + (h // 2)) < 950) and (20 < w < 80) and (
                    10 < h < 60):  # Defined range for landmarks 3
                cur_x3 = x + (w // 2)
                cur_y3 = y + (h // 2)
                cur_3 = cur_3 + 1
            elif (1050 < (x + (w // 2)) < 1450) and (700 < (y + (h // 2)) < 1050) and (20 < w < 90) and (
                    10 < h < 90):  # Defined range for landmarks 4
                cur_x4 = x + (w // 2)
                cur_y4 = y + (h // 2)
                cur_4 = cur_4 + 1

        if cur_1 == 1:
            im_copy = cv2.circle(im_copy, (cur_x1, cur_y1), 10, (255, 0, 0), 2)
        if cur_2 == 1:
            im_copy = cv2.circle(im_copy, (cur_x2, cur_y2), 10, (255, 0, 0), 2)
        if cur_3 == 1:
            im_copy = cv2.circle(im_copy, (cur_x3, cur_y3), 10, (255, 0, 0), 2)
        if cur_4 == 1:
            im_copy = cv2.circle(im_copy, (cur_x4, cur_y4), 10, (255, 0, 0), 2)

        # Check if all 4 landmarks are detected
        # CASE: ALL 4 LANDMARKS DETECTED SUCCESSFULLY
        if (cur_1 == 1) and (cur_2 == 1) and (cur_3 == 1) and (cur_4 == 1):  # 1-1-1-1
            # Only need to have one pair of unchanged coordinates
            if ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_x2 - 10 < cur_x2 < ref_x2 + 10) and
                (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10)) or \
                    ((ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and
                     (ref_y3 - 10 < cur_y3 < ref_y3 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)) or \
                    ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_x3 - 10 < cur_x3 < ref_x3 + 10) and
                     (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10)) or \
                    ((ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and
                     (ref_y2 - 10 < cur_y2 < ref_y2 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)) or \
                    ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and
                     (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)) or \
                    ((ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_x3 - 10 < cur_x3 < ref_x3 + 10) and
                     (ref_y2 - 10 < cur_y2 < ref_y2 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10)):
                out = im_copy
                pass
            else:
                # Specify input and output coordinates that is used to calculate the transformation matrix
                input_pts = np.float32([[cur_x1, cur_y1], [cur_x2, cur_y2], [cur_x3, cur_y3], [cur_x4, cur_y4]])
                output_pts = np.float32([[ref_x1, ref_y1], [ref_x2, ref_y2], [ref_x3, ref_y3], [ref_x4, ref_y4]])
                # Compute the perspective transform M
                M = cv2.getPerspectiveTransform(input_pts, output_pts)
                M_tmp = M
                M_4point = 1;
                M_3point = 0
                # Apply the perspective transformation to the image
                out = cv2.warpPerspective(im_copy, M, (im_copy.shape[1], im_copy.shape[0]), flags=cv2.INTER_LINEAR)

        # CASE: ONLY 3 LANDMARKS DETECTED
        elif (cur_1 != 1) and (cur_2 == 1) and (cur_3 == 1) and (cur_4 == 1):  # x-1-1-1
            # Case: Camera's angle is unchanged
            # Only need to have one pair of unchanged coordinates
            if ((ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10) and
                (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)) or \
                    ((ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10) and
                     (ref_y3 - 10 < cur_y3 < ref_y3 + 10) and (ref_x3 - 10 < cur_x3 < ref_x3 + 10)) or \
                    ((ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10) and
                     (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)):
                cur_x1 = ref_x1;
                cur_y1 = ref_y1
                out = im_copy
                pass
            # Case: The camera's angle is changed
            else:
                # 3-point Homography
                input_pts = np.float32([[cur_x2, cur_y2], [cur_x3, cur_y3], [cur_x4, cur_y4]])
                output_pts = np.float32([[ref_x2, ref_y2], [ref_x3, ref_y3], [ref_x4, ref_y4]])
                M = cv2.getAffineTransform(input_pts, output_pts)
                M_tmp = M
                M_4point = 0;
                M_3point = 1
                out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                # Interpolate the 4th point
                point = (ref_x1, ref_y1)
                warped_point = M.dot(np.array(point + (1,)))
                cur_x1 = int(warped_point[0]);
                cur_y1 = int(warped_point[1])

        elif (cur_1 == 1) and (cur_2 != 1) and (cur_3 == 1) and (cur_4 == 1):  # 1-x-1-1
            # Case: The camera angle is unchanged
            # Only need to have one pair of unchanged coordinates
            if ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and
                (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)) or \
                    ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and
                     (ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10)) or \
                    ((ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10) and
                     (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)):
                cur_x2 = ref_x2;
                cur_y2 = ref_y2
                out = im_copy
                pass
            # Case: The camera angle is changed
            else:
                # 3-point Homography
                input_pts = np.float32([[cur_x1, cur_y1], [cur_x3, cur_y3], [cur_x4, cur_y4]])
                output_pts = np.float32([[ref_x1, ref_y1], [ref_x3, ref_y3], [ref_x4, ref_y4]])
                M = cv2.getAffineTransform(input_pts, output_pts)
                M_tmp = M
                M_4point = 0;
                M_3point = 1
                out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                # Interpolate the 4th point
                point = (ref_x2, ref_y2)
                warped_point = M.dot(np.array(point + (1,)))
                cur_x2 = int(warped_point[0]);
                cur_y2 = int(warped_point[1])

        elif (cur_1 == 1) and (cur_2 == 1) and (cur_3 != 1) and (cur_4 == 1):  # 1-1-x-1
            # Case: The camera angle is unchanged
            # Only need to have one pair of unchanged coordinates
            if ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and
                (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)) or \
                    ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and
                     (ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10)) or \
                    ((ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10) and
                     (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10)):
                cur_x3 = ref_x3;
                cur_y3 = ref_y3
                out = im_copy
                pass
            # Case: The camera angle is changed
            else:
                # 3-point Homography
                input_pts = np.float32([[cur_x1, cur_y1], [cur_x2, cur_y2], [cur_x4, cur_y4]])
                output_pts = np.float32([[ref_x1, ref_y1], [ref_x2, ref_y2], [ref_x4, ref_y4]])
                M = cv2.getAffineTransform(input_pts, output_pts)
                M_tmp = M
                M_4point = 0;
                M_3point = 1
                out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                # Interpolate the 4th point
                point = (ref_x3, ref_y3)
                warped_point = M.dot(np.array(point + (1,)))
                cur_x3 = int(warped_point[0]);
                cur_y3 = int(warped_point[1])

        elif (cur_1 == 1) and (cur_2 == 1) and (cur_3 == 1) and (cur_4 != 1):  # 1-1-1-x
            # Case: The camera angle is unchanged
            # Only need to have one pair of unchanged coordinates
            if ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and
                (ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10)) or \
                    ((ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and
                     (ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10)) or \
                    ((ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10) and
                     (ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10)):
                cur_x4 = ref_x4;
                cur_y4 = ref_y4
                out = im_copy
                pass
            # Case: The camera angle is changed
            else:
                # 3-point Homography
                input_pts = np.float32([[cur_x1, cur_y1], [cur_x2, cur_y2], [cur_x3, cur_y3]])
                output_pts = np.float32([[ref_x1, ref_y1], [ref_x2, ref_y2], [ref_x3, ref_y3]])
                M = cv2.getAffineTransform(input_pts, output_pts)
                M_tmp = M
                M_4point = 0;
                M_3point = 1
                out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                # Interpolate the 4th point
                point = (ref_x4, ref_y4)
                warped_point = M.dot(np.array(point + (1,)))
                cur_x4 = int(warped_point[0]);
                cur_y4 = int(warped_point[1])

        # CASE: ONLY 2 LANDMARKS DETECTED
        # Add another case comparing with previous coordinates and matrix
        # In previous cases, this step isn't necessary
        elif (cur_1 != 1) and (cur_2 != 1) and (cur_3 == 1) and (cur_4 == 1):  # x-x-1-1
            if (ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10) and \
                    (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10):
                cur_x1 = ref_x1;
                cur_y1 = ref_y1
                cur_x2 = ref_x2;
                cur_y2 = ref_y2
                out = im_copy
                pass
            elif (cur_x3_tmp - 10 < cur_x3 < cur_x3_tmp + 10) and (cur_y3_tmp - 10 < cur_y3 < cur_y3_tmp + 10) and \
                    (cur_x4_tmp - 10 < cur_x4 < cur_x4_tmp + 10) and (cur_y4_tmp - 10 < cur_y4 < cur_y4_tmp + 10):
                cur_x1 = cur_x1_tmp;
                cur_y1 = cur_y1_tmp
                cur_x2 = cur_x2_tmp;
                cur_y2 = cur_y2_tmp
                if (M_4point == 1) and (M_3point == 0):
                    M = M_tmp
                    M_4point = 1;
                    M_3point = 0
                    M_tmp = M
                    out = cv2.warpPerspective(im_copy, M, (im_copy.shape[1], im_copy.shape[0]), flags=cv2.INTER_LINEAR)
                elif (M_4point == 0) and (M_3point == 1):
                    M = M_tmp
                    M_4point = 0;
                    M_3point = 1
                    M_tmp = M
                    out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                else:
                    cv2.putText(out, 'CODE ERROR', (20, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                out, M = My2pointWarp(im_copy, ref_x3, ref_y3, ref_x4, ref_y4, cur_x3, cur_y3, cur_x4, cur_y4)
                # 1st landmark coordinates interpolation
                point_1 = (ref_x1, ref_y1)
                warped_point_1 = M.dot(np.array(point_1 + (1,)))
                cur_x1 = int(warped_point_1[0]);
                cur_y1 = int(warped_point_1[1])
                # 2nd landmark coordinates interpolation
                point_2 = (ref_x2, ref_y2)
                warped_point_2 = M.dot(np.array(point_2 + (1,)))
                cur_x2 = int(warped_point_2[0]);
                cur_y2 = int(warped_point_2[1])
            # print('Only 2 landmarks detected in C9 (%d), please check the camera' % ii)
            cv2.putText(out, 'Only 2 landmarks detected, please check the camera', (20, 1000), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            var = 386

        elif (cur_1 != 1) and (cur_2 == 1) and (cur_3 != 1) and (cur_4 == 1):  # x-1-x-1
            if (ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10) and \
                    (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10):
                cur_x1 = ref_x1;
                cur_y1 = ref_y1
                cur_x3 = ref_x3;
                cur_y3 = ref_y3
                out = im_copy
                pass
            elif (cur_x2_tmp - 10 < cur_x2 < cur_x2_tmp + 10) and (cur_y2_tmp - 10 < cur_y2 < cur_y2_tmp + 10) and \
                    (cur_x4_tmp - 10 < cur_x4 < cur_x4_tmp + 10) and (cur_y4_tmp - 10 < cur_y4 < cur_y4_tmp + 10):
                cur_x1 = cur_x1_tmp;
                cur_y1 = cur_y1_tmp
                cur_x3 = cur_x3_tmp;
                cur_y3 = cur_y3_tmp
                if (M_4point == 1) and (M_3point == 0):
                    M = M_tmp
                    M_4point = 1;
                    M_3point = 0
                    M_tmp = M
                    out = cv2.warpPerspective(im_copy, M, (im_copy.shape[1], im_copy.shape[0]), flags=cv2.INTER_LINEAR)
                elif (M_4point == 0) and (M_3point == 1):
                    M = M_tmp
                    M_4point = 0;
                    M_3point = 1
                    M_tmp = M
                    out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                else:
                    cv2.putText(out, 'CODE ERROR', (20, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                out, M = My2pointWarp(im_copy, ref_x2, ref_y2, ref_x4, ref_y4, cur_x2, cur_y2, cur_x4, cur_y4)
                M_tmp = M
                # 1st landmark coordinates interpolation
                point_1 = (ref_x1, ref_y1)
                warped_point_1 = M.dot(np.array(point_1 + (1,)))
                cur_x1 = int(warped_point_1[0]);
                cur_y1 = int(warped_point_1[1])
                # 3rd landmark coordinates interpolation
                point_3 = (ref_x3, ref_y3)
                warped_point_3 = M.dot(np.array(point_3 + (1,)))
                cur_x3 = int(warped_point_3[0]);
                cur_y3 = int(warped_point_3[1])
            # print('Only 2 landmarks detected in C9 (%d), please check the camera' % ii)
            cv2.putText(out, 'Only 2 landmarks detected, please check the camera', (20, 1000), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        elif (cur_1 != 1) and (cur_2 == 1) and (cur_3 == 1) and (cur_4 != 1):  # x-1-1-x
            if (ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10) and \
                    (ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10):
                cur_x1 = ref_x1;
                cur_y1 = ref_y1
                cur_x4 = ref_x4;
                cur_y4 = ref_y4
                out = im_copy
                pass
            elif (cur_x3_tmp - 10 < cur_x3 < cur_x3_tmp + 10) and (cur_y3_tmp - 10 < cur_y3 < cur_y3_tmp + 10) and \
                    (cur_x2_tmp - 10 < cur_x2 < cur_x2_tmp + 10) and (cur_y2_tmp - 10 < cur_y2 < cur_y2_tmp + 10):
                cur_x1 = cur_x1_tmp;
                cur_y1 = cur_y1_tmp
                cur_x4 = cur_x4_tmp;
                cur_y4 = cur_y4_tmp
                if (M_4point == 1) and (M_3point == 0):
                    M = M_tmp
                    M_4point = 1;
                    M_3point = 0
                    M_tmp = M
                    out = cv2.warpPerspective(im_copy, M, (im_copy.shape[1], im_copy.shape[0]), flags=cv2.INTER_LINEAR)
                elif (M_4point == 0) and (M_3point == 1):
                    M = M_tmp
                    M_4point = 0;
                    M_3point = 1
                    M_tmp = M
                    out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                else:
                    cv2.putText(out, 'CODE ERROR', (20, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                out, M = My2pointWarp(im_copy, ref_x2, ref_y2, ref_x3, ref_y3, cur_x2, cur_y2, cur_x3, cur_y3)
                M_tmp = M
                # 1st landmark coordinates interpolation
                point_1 = (ref_x1, ref_y1)
                warped_point_1 = M.dot(np.array(point_1 + (1,)))
                cur_x1 = int(warped_point_1[0]);
                cur_y1 = int(warped_point_1[1])
                # 4th landmark coordinates interpolation
                point_4 = (ref_x4, ref_y4)
                warped_point_4 = M.dot(np.array(point_4 + (1,)))
                cur_x4 = int(warped_point_4[0]);
                cur_y4 = int(warped_point_4[1])
            # print('Only 2 landmarks detected in C9 (%d), please check the camera' % ii)
            cv2.putText(out, 'Only 2 landmarks detected, please check the camera', (20, 1000), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        elif (cur_1 == 1) and (cur_2 != 1) and (cur_3 != 1) and (cur_4 == 1):  # 1-x-x-1
            if (ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and \
                    (ref_x4 - 10 < cur_x4 < ref_x4 + 10) and (ref_y4 - 10 < cur_y4 < ref_y4 + 10):
                cur_x3 = ref_x3;
                cur_y3 = ref_y3
                cur_x2 = ref_x2;
                cur_y2 = ref_y2
                out = im_copy
                pass
            elif (cur_x1_tmp - 10 < cur_x1 < cur_x1_tmp + 10) and (cur_y1_tmp - 10 < cur_y1 < cur_y1_tmp + 10) and \
                    (cur_x4_tmp - 10 < cur_x4 < cur_x4_tmp + 10) and (cur_y4_tmp - 10 < cur_y4 < cur_y4_tmp + 10):
                cur_x1 = cur_x1_tmp;
                cur_y1 = cur_y1_tmp
                cur_x4 = cur_x4_tmp;
                cur_y4 = cur_y4_tmp
                if (M_4point == 1) and (M_3point == 0):
                    M = M_tmp
                    M_4point = 1;
                    M_3point = 0
                    M_tmp = M
                    out = cv2.warpPerspective(im_copy, M, (im_copy.shape[1], im_copy.shape[0]), flags=cv2.INTER_LINEAR)
                elif (M_4point == 0) and (M_3point == 1):
                    M = M_tmp
                    M_4point = 0;
                    M_3point = 1
                    M_tmp = M
                    out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                else:
                    cv2.putText(out, 'CODE ERROR', (20, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                out, M = My2pointWarp(im_copy, ref_x1, ref_y1, ref_x4, ref_y4, cur_x1, cur_y1, cur_x4, cur_y4)
                M_tmp = M
                # 3rd landmark coordinates interpolation
                point_3 = (ref_x3, ref_y3)
                warped_point_3 = M.dot(np.array(point_3 + (1,)))
                cur_x3 = int(warped_point_3[0]);
                cur_y3 = int(warped_point_3[1])
                # 2nd landmark coordinates interpolation
                point_2 = (ref_x2, ref_y2)
                warped_point_2 = M.dot(np.array(point_2 + (1,)))
                cur_x2 = int(warped_point_2[0]);
                cur_y2 = int(warped_point_2[1])
            # print('Only 2 landmarks detected in C9 (%d), please check the camera' % ii)
            cv2.putText(out, 'Only 2 landmarks detected, please check the camera', (20, 1000), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        elif (cur_1 == 1) and (cur_2 != 1) and (cur_3 == 1) and (cur_4 != 1):  # 1-x-1-x
            if (ref_x3 - 10 < cur_x3 < ref_x3 + 10) and (ref_y3 - 10 < cur_y3 < ref_y3 + 10) and \
                    (ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10):
                cur_x4 = ref_x4;
                cur_y4 = ref_y4
                cur_x2 = ref_x2;
                cur_y2 = ref_y2
                out = im_copy
                pass
            elif (cur_x1_tmp - 10 < cur_x1 < cur_x1_tmp + 10) and (cur_y1_tmp - 10 < cur_y1 < cur_y1_tmp + 10) and \
                    (cur_x3_tmp - 10 < cur_x3 < cur_x3_tmp + 10) and (cur_y3_tmp - 10 < cur_y3 < cur_y3_tmp + 10):
                cur_x4 = cur_x4_tmp;
                cur_y4 = cur_y4_tmp
                cur_x2 = cur_x2_tmp;
                cur_y2 = cur_y2_tmp
                if (M_4point == 1) and (M_3point == 0):
                    M = M_tmp
                    M_4point = 1;
                    M_3point = 0
                    M_tmp = M
                    out = cv2.warpPerspective(im_copy, M, (im_copy.shape[1], im_copy.shape[0]), flags=cv2.INTER_LINEAR)
                elif (M_4point == 0) and (M_3point == 1):
                    M = M_tmp
                    M_4point = 0;
                    M_3point = 1
                    M_tmp = M
                    out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                else:
                    cv2.putText(out, 'CODE ERROR', (20, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                out, M = My2pointWarp(im_copy, ref_x1, ref_y1, ref_x3, ref_y3, cur_x1, cur_y1, cur_x3, cur_y3)
                M_tmp = M
                # 4th landmark coordinates interpolation
                point_4 = (ref_x4, ref_y4)
                warped_point_4 = M.dot(np.array(point_4 + (1,)))
                cur_x4 = int(warped_point_4[0]);
                cur_y4 = int(warped_point_4[1])
                # 2nd landmark coordinates interpolation
                point_2 = (ref_x2, ref_y2)
                warped_point_2 = M.dot(np.array(point_2 + (1,)))
                cur_x2 = int(warped_point_2[0]);
                cur_y2 = int(warped_point_2[1])
            # print('Only 2 landmarks detected in C9 (%d), please check the camera' % ii)
            cv2.putText(out, 'Only 2 landmarks detected, please check the camera', (20, 1000), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        elif (cur_1 == 1) and (cur_2 == 1) and (cur_3 != 1) and (cur_4 != 1):  # 1-1-x-x
            if (ref_x1 - 10 < cur_x1 < ref_x1 + 10) and (ref_y1 - 10 < cur_y1 < ref_y1 + 10) and \
                    (ref_x2 - 10 < cur_x2 < ref_x2 + 10) and (ref_y2 - 10 < cur_y2 < ref_y2 + 10):
                cur_x3 = ref_x3;
                cur_y3 = ref_y3
                cur_x4 = ref_x4;
                cur_y4 = ref_y4
                out = im_copy
                pass
            elif (cur_x1_tmp - 10 < cur_x1 < cur_x1_tmp + 10) and (cur_y1_tmp - 10 < cur_y1 < cur_y1_tmp + 10) and \
                    (cur_x2_tmp - 10 < cur_x2 < cur_x2_tmp + 10) and (cur_y2_tmp - 10 < cur_y2 < cur_y2_tmp + 10):
                cur_x3 = cur_x3_tmp;
                cur_y3 = cur_y3_tmp
                cur_x4 = cur_x4_tmp;
                cur_y4 = cur_y4_tmp
                if (M_4point == 1) and (M_3point == 0):
                    M = M_tmp
                    M_4point = 1;
                    M_3point = 0
                    M_tmp = M
                    out = cv2.warpPerspective(im_copy, M, (im_copy.shape[1], im_copy.shape[0]), flags=cv2.INTER_LINEAR)
                elif (M_4point == 0) and (M_3point == 1):
                    M = M_tmp
                    M_4point = 0;
                    M_3point = 1
                    M_tmp = M
                    out = cv2.warpAffine(im_copy, M, (im_copy.shape[1], im_copy.shape[0]))
                else:
                    cv2.putText(out, 'CODE ERROR', (20, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                out, M = My2pointWarp(im_copy, ref_x1, ref_y1, ref_x2, ref_y2, cur_x1, cur_y1, cur_x2, cur_y2)
                M_tmp = M
                # 3rd landmark coordinates interpolation
                point_3 = (ref_x3, ref_y3)
                warped_point_3 = M.dot(np.array(point_3 + (1,)))
                cur_x3 = int(warped_point_3[0]);
                cur_y3 = int(warped_point_3[1])
                # 4th landmark coordinates interpolation
                point_4 = (ref_x4, ref_y4)
                warped_point_4 = M.dot(np.array(point_4 + (1,)))
                cur_x4 = int(warped_point_4[0]);
                cur_y4 = int(warped_point_4[1])
            # print('Only 2 landmarks detected in C9 (%d), please check the camera' % ii)
            cv2.putText(out, 'Only 2 landmarks detected, please check the camera', (20, 1000), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        # CASE: ONLY 1 OR NO LANDMARK DETECTED
        else:
            # Use previous coordinates and images
            cur_x1 = cur_x1_tmp;
            cur_y1 = cur_y1_tmp
            cur_x2 = cur_x2_tmp;
            cur_y2 = cur_y2_tmp
            cur_x3 = cur_x3_tmp;
            cur_y3 = cur_y3_tmp
            cur_x4 = cur_x4_tmp;
            cur_y4 = cur_y4_tmp
            out = out_tmp
            # print('DETECTION ERROR IN C9 (%d): %d %d %d %d' % (ii, cur_1, cur_2, cur_3, cur_4))  # Error in detection

        # Store coordinate and output to use later in the next frame
        cur_x1_tmp = cur_x1;
        cur_y1_tmp = cur_y1
        cur_x2_tmp = cur_x2;
        cur_y2_tmp = cur_y2
        cur_x3_tmp = cur_x3;
        cur_y3_tmp = cur_y3
        cur_x4_tmp = cur_x4;
        cur_y4_tmp = cur_y4
        out_tmp = out

        # cv2.rectangle(out, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        cv2.imwrite('ImageCali.jpg', out)  # Name of files

    f1.close()
    f2.close()


def imagecut(imageCali):
    imageArray = []
    # Open slot coordinate file, read in
    f2 = open('Slot coordinate.txt', 'r')
    lis = [line.split() for line in f2]
    # print(lis)
    coord = []
    n = len(lis[0])
    fileLength = len(lis)
    f2.close()
    f2 = open('Slot coordinate.txt', 'r')
    countLine = len(f2.read().split('\n')) - 1
    f2.close()
    for j in range(n):
        tempArray = []
        for i in range(fileLength):
            # print(lis[i][j])
            tempArray.append(int(lis[i][j]))
        coord.append(tempArray)
        # print(tempArray)
    # fSlot = open('Slot coordinate.txt')
    # text = fSlot.read().split('\n')[:-1]
    # coordition = []
    # for line in text:
    #     coordArray = []
    #     for coord2 in line.split(' '):
    #         coordArray.append(int(coord2))
    #
    #     coordition.append(coordArray)
    # fSlot.close()
    coordition = []
    image = cv2.imread(imageCali)
    cut = image
    for index in range(0, countLine):
        posX = int((coord[1][index] + coord[3][index]) / 2)
        posY = int((coord[2][index] + coord[4][index]) / 2)
        if index in range(0, 12):
            img = cut[(posY - 65):(posY + 65), (posX - 65):(posX + 65)]
            coordition.append(((posX - 65, posY - 65),(posX + 65, posY + 65)))
        elif index in range(12, 16):
            img = cut[(posY - 45):(posY + 45), (posX - 45):(posX + 45)]
            coordition.append(((posX - 45, posY - 45),(posX + 45, posY + 45)))
        else:
            img = cut[(posY - 38):(posY + 38), (posX - 38):(posX + 38)]
            coordition.append(((posX - 38, posY - 38),(posX + 38, posY + 38)))
        imageArray.append(img)
    return imageArray, coordition, countLine
