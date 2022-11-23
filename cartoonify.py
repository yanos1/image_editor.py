##############################################################################
# FILE: image_editor.py
# WRITER: Yan Nosrati 318862968
# EXERCISE: Intro2cs2 ex5 2022
# DESCRIPTION: An image modification program

##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from ex5_hepler import *
from typing import *
from math import floor
from copy import deepcopy
from math import *
import sys
NUMBERS = "0123456789"

def separate_channels(image: ColoredImage):
    """
    This function seperates a colorful image into 3 channels
    :param image:
    :return: separated list
    """
    separated_list = []
    for col in range(len(image[0][0])):
        outer_list = []
        for row in range(len(image)):
            inner_list = []
            for num in range(len(image[0])):
                inner_list.append(image[row][num][col])
            outer_list.append(inner_list)
            inner_list = []
        separated_list.append(outer_list)
        outer_list = []
    return separated_list


def combine_channels(channels: List[List[List[int]]]):
    """
    This function takes 3 image channels and turns the into one
    :param channels:
    :return: combined list
    """
    combined_list = []
    for col in range(len(channels[0])):
        outer_list = []
        for row in range(len(channels[0][0])):
            inner_list = []
            for num in range(len(channels)):
                inner_list.append(channels[num][col][row])
            outer_list.append(inner_list)
            inner_list = []
        combined_list.append(outer_list)
        outer_list = []
    return combined_list


def RGB2grayscale(colored_image: ColoredImage):
    """
    This function takes a colorful image and turrns it into a grey scale image
    :param colored_image:
    :return: grey scale image
    """
    colored_image_copy = deepcopy(colored_image)
    for col in range(len(colored_image)):
        for row in range(len(colored_image[0])):
            colored_image_copy[col][row] = (
                round(colored_image_copy[col][row][0] * 0.299 +
                      colored_image_copy[col][row][1] * 0.587
                      + colored_image_copy[col][row][2] * 0.114))

    return colored_image_copy


def blur_kernel(size: int) -> Kernel:
    """
    This function determines the size of the blur kernel
    :param size:
    :return: blur kernel
    """
    blurred_kernel = [[1 / size ** 2 for i in range(size)] for j in
                      range(size)]
    return blurred_kernel


def apply_kernel(image: SingleChannelImage, kernel: Kernel):
    """
    This function allows us to apply kernels to the pictures.
    :param image:
    :param kernel:
    :return:
    """
    new_image = []
    current_list = []
    mid = floor(len(kernel) // 2)
    cur_sum = 0
    ker_row = 0
    ker_col = 0

    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(i - mid, i - mid + len(kernel)):
                for t in range(j - mid, j - mid + len(kernel)):
                    if k < 0 or k >= len(image) or t < 0 or t >= len(image[0]):
                        cur_sum += (image[i][j] * kernel[ker_row][ker_col])
                    else:
                        cur_sum += (image[k][t] * kernel[ker_row][ker_col])
                    ker_col+=1
                ker_row+=1
                ker_col = 0
            ker_row = 0
            if 255 >= cur_sum >= 0:
                current_list.append(round(cur_sum))
            elif cur_sum > 255:
                current_list.append(255)
            else:
                current_list.append(0)
            cur_sum = 0
        new_image.append(current_list)
        current_list = []
    return new_image




def average_kernel(size):
    """
    This is the average kernel
    :param size:
    :return:
    """
    averaged_kernel = [[1 / (size * size)] * size for i in range(size)]
    return averaged_kernel


def apply_average(image: SingleChannelImage, kernel: Kernel):
    new_image = []
    current_list = []
    mid = floor(len(kernel) // 2)
    cur_sum = 0

    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(i - mid, i - mid + len(kernel)):
                for t in range(j - mid, j - mid + len(kernel)):
                    if k < 0 or k >= len(image) or t < 0 or t >= len(image[0]):
                        cur_sum += (image[i][j] * kernel[0][0])
                    else:
                        cur_sum += (image[k][t] * kernel[0][0])
            current_list.append(cur_sum)
            cur_sum = 0
        new_image.append(current_list)
        current_list = []
    return new_image


def calculate_nearest_pixels(image, y, x):
    """
    This function calculates the closest pixels to a given floating number
    :param image:
    :param y:
    :param x:
    :return: the edges of the point
    """
    down = ceil(y)
    up = floor(y)
    left = floor(x)
    right = ceil(x)
    sum = 0
    if down == len(image):
        down -=1
    if right == len(image[0]):
        right -= 1
    delta_x = x - floor(x)
    delta_y = y - floor(y)
    right_down_area = ((1-delta_y) * (1-delta_x))
    right_up_area = ((delta_y) * (1-delta_x))
    left_down_area = ((1-delta_y) * delta_x)
    left_up_area = (delta_x * delta_y)

    sum += right_down_area*image[up][left]
    sum += right_up_area* image[down][left]
    sum += left_down_area * image[up][right]
    sum += left_up_area * image[down][right]

    return round(sum)


def bilinear_interpolation(image: SingleChannelImage, y: float, x: float):
    """
    This function calculates the "trapped" pixel based on its relative position
    :param image:
    :param y:
    :param x:
    :return:
    """
    if isinstance(y, int) and isinstance(x, int):
        return image[y][x]

    elif y.is_integer() and x.is_integer():
        return image[int(y)][int(x)]

    else:
        sum = calculate_nearest_pixels(image, y, x)

    return sum


def resize(image: SingleChannelImage, new_height: int, new_width: int):
    """
    This function lets us resize an image based on given parameters.
    :param image:
    :param new_height:
    :param new_width:
    :return: the resized image
    """
    new_image = [[item] * new_width for item in range(new_height)]
    new_image[0][0] = image[0][0]
    new_image[0][-1] = image[0][-1]
    new_image[-1][0] = image[-1][0]
    new_image[-1][-1] = image[-1][-1]

    for line in range(len(new_image)):
        for index in range(len(new_image[0])):
            if (line == 0 and index == 0) or (
                    line == len(new_image) and index == len(new_image[0])) or (
                    line == 0 and index == len(new_image[0])):
                continue
            else:
                map_val_y = (line * (len(image) - 1)) / (len(new_image) - 1)
                map_val_x = (index * (len(image[0]) - 1)) / (len(new_image[0]) - 1)
                new_image[line][index] = bilinear_interpolation(image,
                                                                map_val_y,
                                                                map_val_x)

    return new_image


# def scale_down_colored_image(image: ColoredImage, max_size: int):
#     """
#     This functuon determines if an image is too big. if it is we scale it down.
#     :param image:
#     :param max_size:
#     :return: scaled image
#     """
#     image_copy = deepcopy(image)
#     helping_list = []
#     if len(image) <= max_size and len(image[0]) <= max_size:
#         return
#
#     else:
#         for item in separate_channels(image_copy):
#             helping_list.append(resize(item, max_size, max_size))
#
#         combined_list = combine_channels(helping_list)
#         return combined_list


def rotate_90(image: Image, direction: str) -> Image:
    """
    This function rotates an image to the left or to the right.
    :param image:
    :param direction:
    :return: rotated image
    """
    inner_transposed_list = []
    final_transposed_list = []

    if direction == 'R':
        for i in range(len(image[0])):
            for j in range(len(image)):
                inner_transposed_list.append(image[j][i])
            reversed_list = inner_transposed_list[::-1]
            final_transposed_list.append(reversed_list)
            inner_transposed_list = []
        return final_transposed_list
    else:
        for i in range(len(image[0])):
            for j in range(len(image)):
                inner_transposed_list.append(image[j][i])
            final_transposed_list.append(inner_transposed_list)
            inner_transposed_list = []
        return final_transposed_list[::-1]


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int,
              c: int) -> SingleChannelImage:
    """
    This function turns an image into a black and while image based on a threshold.
    :param image:
    :param blur_size:
    :param block_size:
    :param c:
    :return:
    """
    image_copy = deepcopy(image)
    blurred_list = apply_kernel(image_copy, blur_kernel(blur_size))
    averaged_list = apply_average(blurred_list, average_kernel(block_size))
    for row in range(len(blurred_list)):
        for pixel in range(len(blurred_list[0])):
            if image_copy[row][pixel] + c < averaged_list[row][pixel]:
                image_copy[row][pixel] = 0
            else:
                image_copy[row][pixel] = 255

    return image_copy


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    """
    This function applies a basic quantization to an image
    :param image:
    :param N:
    :return: quantinized image
    """
    image_copy = deepcopy(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            image_copy[i][j] = round(
                floor(image_copy[i][j] * (N / 256)) * 255 / (N - 1))
    return image_copy


def quantize_colored_image(image: ColoredImage, N:int) -> ColoredImage:
    """
    This function applies qunatinazation to a colored image.
    :param image:
    :param N:
    :return: quantinized image
    """
    separated_list = separate_channels(image)
    channels = []
    for channel in separated_list:
        channels.append(quantize(channel, N))

    final_combined_list = combine_channels(channels)
    return final_combined_list

#
# def add_mask(image1: Image, image2: Image, mask: List[List[float]]):
#     """
#     This function is probably written horribly. But what it does is
#      images based on a given mask.
#     :param image1:
#     :param image2:
#     :param mask:
#     :return: new masked image
#     """
#     new_image = []
#     outer_list = []
#     inner_list = []
#     if isinstance(image1[0][0], list):
#         for i in range(len(image1)):
#             for j in range(len(image1[0])):
#                 for k in range(len(image1[0][0])):
#                     inner_list.append(round(
#                         (image1[i][j][k] * (mask[i][j])) + (
#                                 image2[i][j][k] * (1 - (mask[i][j])))))
#                 outer_list.append(inner_list)
#                 inner_list = []
#             new_image.append(outer_list)
#             outer_list = []
#
#     elif isinstance(image1[0][0], int):
#         for i in range(len(image1)):
#             for j in range(len(image1[0])):
#                 inner_list.append(round((image1[i][j] * (mask[i][j])) + (
#                     (image2[i][j] * (1 - (mask[i][j]))))))
#             new_image.append(inner_list)
#             inner_list = []
#
#     return new_image
#
#
# def cartoonify(image: ColoredImage, blur_size: int, th_block_size: int,
#                th_c: int, quant_num_shades: int) -> ColoredImage:
#     """
#     The core of our program. this function creates a ccartoonified image.
#     :param image:
#     :param blur_size:
#     :param th_block_size:
#     :param th_c:
#     :param quant_num_shades:
#     :return:
#     """
#     edged_image = get_edges(RGB2grayscale(image), blur_size, th_block_size,
#                             th_c)
#     quantized_image = quantize_colored_image(image, quant_num_shades)
#     edged_image_copy = deepcopy(edged_image)
#     for index1, row in enumerate(edged_image_copy):
#         for index2, pixel in enumerate(row):
#             edged_image_copy[index1][index2] = (pixel / 255)
#     channels = []
#     for channel in separate_channels(quantized_image):
#         channels.append(add_mask(channel, edged_image, edged_image_copy))
#
#     cartoonified_image = combine_channels(channels)
#
#     return cartoonified_image
def invalid_input(img):
    print("invalid input")
    print_actions()
    return img

def use_grey(img):
    if isinstance(img[0][0], int):
        image = RGB2grayscale(img)
        return image

    else:
        return invalid_input(img)
def use_blur(img,kernel_size):
    if kernel_size in NUMBERS:
        kernel_size = int(kernel_size)
        if kernel_size > 0 and kernel_size == round(kernel_size) and kernel_size % 2 == 1:
            if isinstance(img[0][0], int):
                image = apply_kernel(img, blur_kernel(kernel_size))
                return image
            else:
                separeted = separate_channels(img)
                channels = []
                for channel in separeted:
                    channels.append(apply_kernel(channel, blur_kernel(kernel_size)))
                image = combine_channels(channels)
                return image
        else:
            return invalid_input(img)
    else:
        return invalid_input(img)
def use_resize(img,size):
    if len(size) == 2:
        for i in size:
            if int(i) <= 1 or int(i) != float(i):
                return invalid_input(img)
        if isinstance(img[0][0], int):
            image = resize(img, int(size[0]),
                           int(size[1]))
            return image
        else:
            separeted = separate_channels(img)
            channels = []
            for channel in separeted:
                channels.append(
                    resize(channel,int(size[0]),int(size[1])))
            image = combine_channels(channels)
            return image
    else:
        return invalid_input(img)



def use_rotation(img,direction):
    if direction == "L":
        image = rotate_90(img,direction)
        return image
    elif direction == "R":
        image = rotate_90(img,direction)
        return image
    else:
        return invalid_input(img)


def use_get_edges(img,edge_inpt):
    if len(edge_inpt) == 3:
        for i in edge_input[:2]:
            if int(i) != float(i) or int(i) <= 0 or int(i)%2 == 0:
                return invalid_input(img)

            if int(edge_input[2]) < 0:
                return invalid_input(img)
            if isinstance(img[0][0],int):
                image = get_edges(img, int(edge_inpt[0]),int(edge_inpt[1]),int(edge_inpt[2]))
                return image
            else:
                grayed = RGB2grayscale(img)
                image = get_edges(grayed,int(edge_inpt[0]),int(edge_inpt[1]),int(edge_inpt[2]))
                return image
    else:
        return invalid_input(img)
def use_quantinize(img,quant_inpt):
    if int(quant_inpt) > 1 and int(quant_inpt) == float(quant_inpt):
        if isinstance(img[0][0],int):
            image = quantize(img,int(quant_inpt))
            return image
        else:
            image = quantize_colored_image(img,int(quant_inpt))
            return image
    else:
        return invalid_input(img)


def print_actions():
    print(f"Welcome to your photos editor! "
          f"Please let me know how you would like to edit!\n"
          f"Here's what you can do\n"
          f"1: turning ap colored image to gray scaled one.\n"
          f"2: blurring an image\n"
          f"3: resizing an image\n"
          f"4: Rotating an image\n"
          f"5: marking edges of an image\n"
          f"6: quintinizing an image\n"
          f"7: showing the image\n"
          f"8: exit the editor")


if __name__ == '__main__':
    if len(sys.argv) == 2:
        image = load_image(f"{sys.argv[1]}")
        print_actions()
        while True:
            user_input = input("please choose an action (1-8)")
            if user_input in NUMBERS and len(user_input) !=0:
                user_input = int(user_input)
            if user_input == 1:
                image = use_grey(image)

            elif user_input == 2:
                kernel = input("Please enter the kernel size")
                image = use_blur(image,kernel)

            elif user_input ==3:
                resize_input = input("please enter 'size height,size width'").split(",")
                image = use_resize(image,resize_input)
            elif user_input ==4:
                rotation_input = input("enter L or R")
                image = use_rotation(image, rotation_input)
            elif user_input ==5:
                edge_input = input("Enter in this format: blur_size,block_size,c").split(",")
                image = use_get_edges(image, edge_input)
            elif user_input ==6:
                quant_input = (input("Enter number of shades."))
                image = use_quantinize(image,quant_input)
            elif user_input ==7:
                show_image(image)
            elif user_input ==8:
                path_input = input(
                    "Good job editing. now please enter a valid path to save the phone at")
                save_image(image, path_input)
                break
    else:
        print("invalid input.")
        print_actions()














    # if len(sys.argv) == 8:
    #     image = load_image(f"{sys.argv[1]}")
    #     if len(image) > int(sys.argv[3]) or len(image[0]) > int(sys.argv[3]):
    #         scaled_imgae = scale_down_colored_image(image, int(sys.argv[3]))
    #         cartooned_image = cartoonify(scaled_imgae, int(sys.argv[4]),
    #                                      int(sys.argv[5]), int(sys.argv[6]),
    #                                      int(sys.argv[7]))
    #         save_image(cartooned_image, sys.argv[2])
    # else:
    #      raise ValueError("Wrong number of arguments.")


