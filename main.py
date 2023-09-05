import os

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy.ndimage.filters import uniform_filter as filter, gaussian_filter

np.set_printoptions(threshold=np.inf)

kernel_size = (11, 11)


def census(im, kernel_):
    height, width = im.shape

    kh, kw = kernel_.shape
    hh, hw = kh // 2, kw // 2
    res_transform = np.zeros((height, width), dtype=np.int64)

    for i in range(hh, height - hh):
        for j in range(hw, width - hw):
            center_pixel = im[i, j]

            code = 0
            bit_position = 0

            for m in range(-hh, hh + 1):
                for n in range(-hw, hw + 1):
                    if m == 0 and n == 0:
                        continue

                    neighbor_pixel = im[i + m, j + n]

                    if neighbor_pixel >= center_pixel:
                        code |= (1 << bit_position)

                    bit_position += 1

            res_transform[i, j] = code

    return res_transform


def cost_vol_left(left_image, right_image, max_disparity):
    height, width = left_image.shape

    cost_volume = np.full((height, width, max_disparity), 9, dtype=np.float64)

    for d in range(max_disparity):
        for y in range(height):
            for x in range(width):
                if x - d >= 0:
                    cost = np.bitwise_xor(left_image[y, x], right_image[y, x - d])
                    cost_volume[y, x, d] = bin(cost)[2:].count('1')

    return cost_volume


# def cost_vol_left(left_image, right_image, max_disparity):
#     height, width = left_image.shape
#     padded_right = np.pad(right_image, ((0, 0), (0,max_disparity)), mode='constant')
#     cost_volume = np.bitwise_xor(left_image[:, :, np.newaxis], padded_right[:, max_disparity:max_disparity+width, np.newaxis])
#     return cost_volume.astype(np.float32)

def cost_vol_right(right_image, left_image, max_disparity):
    height, width = right_image.shape

    cost_volume = np.full((height, width, max_disparity), 9, dtype=np.float64)

    for d in range(max_disparity):
        for y in range(height):
            for x in range(width):
                if x + d < width:
                    cost = np.bitwise_xor((left_image[y, x + d]), (right_image[y, x]))
                    cost_volume[y, x, d] = bin(cost)[2:].count('1')

    return cost_volume


# def cost_vol_right(right_image, left_image, max_disparity):
#     height, width = right_image.shape
#     padded_left = np.pad(left_image, ((0, 0), (max_disparity, 0)), mode='constant')
#     cost_volume = np.bitwise_xor(padded_left[:, :width, np.newaxis], right_image[:, :, np.newaxis])
#     return cost_volume.astype(np.float32)


def aggregation(cost_volume, max_disparity):
    aggregated_cost_volume = np.zeros_like(cost_volume)
    for d in range(max_disparity):
        aggregated_cost_volume[:, :, d] = filter(cost_volume[:, :, d], kernel_size)

    for d in range(max_disparity):
        aggregated_cost_volume[:, :, d] = gaussian_filter(cost_volume[:, :, d], kernel_size)

    return aggregated_cost_volume


def winner_takes_all(aggregated_cost_volume):
    return np.argmin(aggregated_cost_volume, axis=2)


def consistency_right(left_disparity_map, right_disparity_map, threshold):
    height, width = left_disparity_map.shape
    filtered_disparity_map = np.zeros_like(left_disparity_map)

    for y in range(height):
        for x in range(width):
            left_disparity = left_disparity_map[y, x]
            left_x = int(x + left_disparity)

            if left_x < width:
                right_disparity = right_disparity_map[y, left_x]

                if abs(left_disparity - right_disparity) == threshold:
                    filtered_disparity_map[y, x] = left_disparity

    return filtered_disparity_map


def consistency_left(left_disparity_map, right_disparity_map, threshold):
    height, width = left_disparity_map.shape
    filtered_disparity_map = np.zeros_like(left_disparity_map)

    for y in range(height):
        for x in range(width):
            left_disparity = left_disparity_map[y, x]
            right_x = int(x - left_disparity)

            if 0 <= right_x:
                right_disparity = right_disparity_map[y, right_x]

                if abs(left_disparity - right_disparity) == threshold:
                    filtered_disparity_map[y, x] = left_disparity

    return filtered_disparity_map


def im_show(im):
    plt.imshow(im)
    plt.show()


def depth_ims(disparity_map, focal_len, T):
    f = focal_len
    height, width = disparity_map.shape
    depth_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            if disparity_map[y, x] != 0:
                depth = f * T / (disparity_map[y, x])
                depth_map[y, x] = depth

    return depth_map


# def reprojection_to_3d(image, depth_map, K, t):
#     height, width = image.shape[:2]
#     matrix_3d = np.zeros((height, width, 4))  # Initialize the 3D matrix
#     p = numpy.matmul(K, np.array([[1, 0, 0, t], [0, 1, 0, t], [0, 0, 1, t]]))
#     U, s, V_T = np.linalg.svd(p.T)
#
#     sigma = np.zeros((U.shape[1], V_T.shape[0]))
#     n = min(U.shape[1], V_T.shape[0])
#     for i in range(n):
#         sigma[i][i] = s[i]
#
#     # print(sigma.astype(np.int64))
#     sig_pinv = np.linalg.pinv(sigma)
#
#     # print(sig_pinv.astype(np.int64))
#     for y in range(height):
#         for x in range(width):
#             homogeneous_coord = np.array([x, y, 1])
#             # print(U.shape ,sigma.T.shape,V_T.shape,homogeneous_coord.shape)
#             transformed_coord = U @ sig_pinv.T @ V_T @ homogeneous_coord
#             # transformed_coord = sigma@ homogeneous_coord
#             # transformed_coord = transformed_coord
#             Z = depth_map[y, x]
#             transformed_coord = transformed_coord * Z
#             # transformed_coord[0] = 0
#             # transformed_coord[1] = 0
#             # transformed_coord[2] =
#             transformed_coord[3] = 1
#             matrix_3d[y, x] = transformed_coord
#
#     return matrix_3d
#
#
# def project(left_image, coor_3D, K, t=0):
#     p = numpy.matmul(K, np.array([[1, 0, 0, t], [0, 1, 0, t], [0, 0, 1, t]]))
#     res_image = np.zeros(left_image.shape)
#     red_channel_res, green_channel_res, blue_channel_res = cv2.split(res_image)
#     red_channel_left, green_channel_left, blue_channel_left = cv2.split(left_image)
#
#     h = left_image.shape[0]
#     w = left_image.shape[1]
#     for i in range(coor_3D.shape[0]):
#         for j in range(coor_3D.shape[1]):
#             X = coor_3D[i, j]
#             v = p @ X
#             if v[2] == 0:
#                 blue_channel_res[i, j], green_channel_res[i, j], red_channel_res[i, j] = 0, 0, 0
#                 continue
#             v = v // v[2]
#             x = v[0]
#             y = v[1]
#
#             if 0 <= y < h and 0 <= x < w:
#                 x = int(v[0])
#                 y = int(v[1])
#                 # print(x, y)
#                 blue_channel_res[i, j] = blue_channel_left[y, x]
#                 green_channel_res[i, j] = green_channel_left[y, x]
#                 red_channel_res[i, j] = red_channel_left[y, x]
#
#     rejoined_image = np.zeros((blue_channel_res.shape[0], blue_channel_res.shape[1], 3), dtype=np.uint8)
#
#     rejoined_image[:, :, 0] = red_channel_res
#     rejoined_image[:, :, 1] = green_channel_res
#     rejoined_image[:, :, 2] = blue_channel_res
#     # plt.imshow(rejoined_image)
#     # plt.show()
#     return rejoined_image
#

def reproject_to_3d(K, im_w, im_h, depth_matrix):
    res = []
    for y in range(im_h):
        for x in range(im_w):
            d = depth_matrix[y, x]
            if d > 0:
                X = (x - K[0, 2]) * d / K[0, 0]
                Y = (y - K[1, 2]) * d / K[1, 1]
                Z = d
                res.append([X, Y, Z])
    return np.array(res)


def program_run():
    kernel_ = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    ])

    for i in range(1, 6):
        print("image_" + str(i) + "  Processing... ")

        file_name = 'set_' + str(i)
        # file_name = 'example'
        left_im = cv2.imread(file_name + '/im_left.jpg', 0)
        right_im = cv2.imread(file_name + '/im_right.jpg', 0)
        K = np.loadtxt(file_name + '/K.txt')
        with open(file_name + "/max_disp.txt", 'r') as file:
            max_disparity = int(file.read().strip())

        # census
        left_con = census(left_im, kernel_)
        right_con = census(right_im, kernel_)
        print("census_" + str(i) + "success")

        # cost volume
        result_cost_volume_left = cost_vol_left(left_con, right_con, int(max_disparity))
        result_cost_volume_right = cost_vol_right(right_con, left_con, int(max_disparity))
        print("cost_volume_" + str(i) + "success")

        # aggregation
        agg_cost_vol_left = aggregation(result_cost_volume_left, int(max_disparity))
        agg_cost_vol_right = aggregation(result_cost_volume_right, int(max_disparity))
        print("agg_" + str(i) + "success")

        # disparity
        disparity_map_left = winner_takes_all(agg_cost_vol_left)
        disparity_map_right = winner_takes_all(agg_cost_vol_right)
        print("winner_" + str(i) + "success")

        # consistency
        disp_left = consistency_left(disparity_map_left, disparity_map_right, 0)
        disp_right = consistency_right(disparity_map_right, disparity_map_left, 0)
        print("disp_" + str(i) + "success")



        # depth calculating
        depth_left_1 = depth_ims(disp_left, K[0][0], 0.1)
        depth_right = depth_ims(disp_right, K[0][0], 0.1)
        print("depth_" + str(i) + "success")

        # normalizing results
        disp_left = cv2.normalize(disp_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_right = cv2.normalize(disp_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # normalizing results
        depth_left = cv2.normalize(depth_left_1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_right = cv2.normalize(depth_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)



        new_file_name = 'set_' + str(i)
        # saving results
        # if os.path.exists(new_file_name):
        #     # Remove the file
        #     os.remove(new_file_name)

        # os.mkdir(new_file_name)

        cv2.imwrite(new_file_name + '/disp_left.jpg', disp_left)
        cv2.imwrite(new_file_name + '/disp_right.jpg', disp_right)

        cv2.imwrite(new_file_name + '/depth_left.jpg', depth_left)
        cv2.imwrite(new_file_name + '/depth_right.jpg', depth_right)
        print("part1_image_" + str(i) + "success")
        # projecting to 3d

        print("part2_image_" + str(i) + "  Processing... ")
        im_left_color = cv2.imread(file_name + '/im_left.jpg')
        # depth_left = np.loadtxt('example/depth_left.txt',delimiter=',')
        # depth_left = cv2.normalize(depth_left, None, alpha=0, beta=5.23, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        points_in_3d = reproject_to_3d(K, im_left_color.shape[1], im_left_color.shape[0], depth_left_1)
        T = 0

        # projecting to 2d with rotation t
        for q_ in range(1, 12):
            transform_matrix = np.array([[1, 0, T], [0, 1, 0], [0, 0, 1]])
            homography_matrix = K @ transform_matrix @ np.linalg.inv(K)
            transformed_image = cv2.warpPerspective(im_left_color, homography_matrix,
                                                    (im_left_color.shape[1], im_left_color.shape[0]))
            P_ = np.array([[1, 0, 0, T], [0, 1, 0, 0], [0, 0, 1, 0]])
            projection_matrix = K @ P_
            im_h, im_w, _ = transformed_image.shape
            res_in_2d = np.zeros_like(transformed_image)
            for point_3d in points_in_3d:
                point_3d_homogeneous = np.append(point_3d, 1)
                transformed_point = projection_matrix @ point_3d_homogeneous
                Mx = transformed_point[0] / transformed_point[2]
                My = transformed_point[1] / transformed_point[2]
                index_x, index_y = int(round(Mx)), int(round(My))
                if 0 <= index_y < im_h and 0 <= index_x < im_w:
                    res_in_2d[index_y, index_x] = transformed_image[index_y, index_x]

            cv2.imwrite(new_file_name + '/synth_' + str(q_) + '.jpg', res_in_2d)
            T -= 0.01

        print("image_" + str(i) + "success")
        print("#################################################")


if __name__ == '__main__':

    program_run()

