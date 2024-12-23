### Check vot2021 keep_track code for details

import numpy as np

def parse_groundtruth_file(string, frame_size):
    # input is a mask - decode it
    m_, offset_ = create_mask_from_string(string[1:].split(','))
    mask = get_array(m_, offset_, frame_size)
    return mask

def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))


def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)

    return mask, (tl_x, tl_y)


def get_array(mask, offset, output_sz=None):
    """
    return an array of 2-D binary mask
    output_sz is in the format: [width, height]
    """
    tl_x, tl_y = offset[0], offset[1]
    region_w, region_h = mask.shape[1], mask.shape[0]
    mask_ = np.zeros((region_h + tl_y, region_w + tl_x), dtype=np.uint8)
    # mask bounds - needed if mask is outside of image
    # TODO: this part of code needs to be tested more with edge cases
    src_x0, src_y0 = 0, 0
    src_x1, src_y1 = mask.shape[1], mask.shape[0]
    dst_x0, dst_y0 = tl_x, tl_y
    dst_x1, dst_y1 = tl_x + region_w, tl_y + region_h
    if dst_x1 > 0 and dst_y1 > 0 and dst_x0 < mask_.shape[1] and dst_y0 < mask_.shape[0]:
        if dst_x0 < 0:
            src_x0 = -dst_x0
            dst_x0 = 0
        if dst_y0 < 0:
            src_y0 = -dst_y0
            dst_y0 = 0
        if dst_x1 > mask_.shape[1]:
            src_x1 -= dst_x1 - mask_.shape[1]# + 1
            dst_x1 = mask_.shape[1]
        if dst_y1 > mask_.shape[0]:
            src_y1 -= dst_y1 - mask_.shape[0]# + 1
            dst_y1 = mask_.shape[0]
        mask_[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]

    # pad with zeros right and down if output size is larger than current mask
    if output_sz is not None:
        pad_x = output_sz[0] - mask_.shape[1]
        if pad_x < 0:
            mask_ = mask_[:, :mask_.shape[1] + pad_x]
            # padding has to be set to zero, otherwise pad function fails
            pad_x = 0
        pad_y = output_sz[1] - mask_.shape[0]
        if pad_y < 0:
            mask_ = mask_[:mask_.shape[0] + pad_y, :]
            # padding has to be set to zero, otherwise pad function fails
            pad_y = 0
        mask_ = np.pad(mask_, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

    return mask_


def mask2bbox(mask):
    """
    mask: 2-D array with a binary mask
    output: coordinates of the top-left and bottom-right corners of the minimal axis-aligned region containing all positive pixels
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rows_i = np.where(rows)[0]
    cols_i = np.where(cols)[0]
    if len(rows_i) > 0 and len(cols_i) > 0:
        rmin, rmax = rows_i[[0, -1]]
        cmin, cmax = cols_i[[0, -1]]
        return (cmin, rmin, cmax - cmin, rmax - rmin)
    else:
        # mask is empty
        return (-1, -1, -1, -1)