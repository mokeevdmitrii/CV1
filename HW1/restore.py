import cv2
import enum
import numpy as np
import typing as tp
import argparse
import time 
from scipy import signal
from scipy.ndimage import median_filter


class InterpolationMode(enum.Enum):
  HORIZONTAL = 0
  VERTICAL = 1


class ImageType(enum.Enum):
  CFA = 0
  RGB_CFA = 1  


def get_rgb(img: np.ndarray, im_type: ImageType) \
      -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if im_type is ImageType.CFA:
      R, G, B = np.zeros_like(img), np.zeros_like(img), np.zeros_like(img)
      R[::2, ::2] = img[::2, ::2]
      G[1::2, ::2] = img[1::2, ::2]
      G[::2, 1::2] = img[::2, 1::2]
      B[1::2, 1::2] = img[1::2, 1::2]
      return R, G, B
    elif im_type is ImageType.RGB_CFA:
      return img[:, :, 0], img[:, :, 1], img[:, :, 2]


# Reconstruct from CFA, not its RGB version!
def interpolate_bayer(img: np.ndarray, mode: InterpolationMode, 
                      img_type: ImageType = ImageType.CFA) -> np.ndarray:
    R, G, B = get_rgb(img, img_type)
    R, G, B = R.astype(np.float32), G.astype(np.float32), B.astype(np.float32)
    h0 = np.array([-0.25, 0, 0.5, 0, -0.25])
    h1 = np.array([0, 0.5, 0, 0.5, 0])
    height, width = R.shape[0], R.shape[1]
    if mode is InterpolationMode.HORIZONTAL:
      conv_g0_even = np.apply_along_axis(lambda r: np.convolve(r, h1, mode='full'), 
                                            axis=1, arr=G[::2, :])[:, 2:width + 2]
      conv_g0_odd = np.apply_along_axis(lambda r: np.convolve(r, h1, mode='full'), 
                                            axis=1, arr=G[1::2, :])[:, 2:width + 2]

      conv_r1_even = np.apply_along_axis(lambda r: np.convolve(r, h0, mode='full'),
                                            axis=1, arr=R[::2, :])[:, 2:width + 2]
      conv_b1_odd = np.apply_along_axis(lambda r: np.convolve(r, h0, mode='full'),
                                            axis=1, arr=B[1::2, :])[:, 2:width+2]
      G[::2, :] += conv_g0_even + conv_r1_even
      G[1::2, :] += conv_g0_odd + conv_b1_odd

    elif mode is InterpolationMode.VERTICAL:
      conv_g0_even = np.apply_along_axis(lambda r: np.convolve(r, h1, mode='full'), 
                                            axis=0, arr=G[:, ::2])[2:height + 2, :]
      conv_g0_odd = np.apply_along_axis(lambda r: np.convolve(r, h1, mode='full'), 
                                            axis=0, arr=G[:, 1::2])[2:height + 2, :]

      conv_r1_even = np.apply_along_axis(lambda r: np.convolve(r, h0, mode='full'),
                                            axis=0, arr=R[:, ::2])[2:height + 2, :]
      conv_b1_odd = np.apply_along_axis(lambda r: np.convolve(r, h0, mode='full'),
                                            axis=0, arr=B[:, 1::2])[2:height + 2, :]
      G[:, ::2] += conv_g0_even + conv_r1_even
      G[:, 1::2] += conv_g0_odd + conv_b1_odd
    else:
      raise RuntimeError("use InterpolationMode.HORIZONTAL | VERTICAL in interpolate")
    # expr. (11) from the article
    RG1_diff = R[::2, ::2] - G[::2, ::2]
    BG1_diff = B[1::2, 1::2] - G[1::2, 1::2]

    # bilinear interpolation. Default for OpenCV, but I'll put it explicitly
    RG_diff = cv2.resize(RG1_diff, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    BG_diff = cv2.resize(BG1_diff, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    R = G + RG_diff
    B = G + BG_diff
    
    return np.stack((R, G, B), axis=-1)


# accepts 1d-arrays and shifts them 1 to left, and 1 to right with edge fill
# for example, given [1, 2, ..., n], will return
# [1, 1, 2, ..., n-1] and [2, ..., n-1, n, n]
def shifted_arrays(arr: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
  if arr.shape[0] < 2:
    return arr
  left = np.roll(arr, 1)
  left[0] = left[1]
  right = np.roll(arr, -1)
  right[-1] = right[-2]
  return left, right

# receives an array and a row index
# returns upper row (if it exists, else the requested one) and same for lower.
def get_ul_row(arr: np.ndarray, r: int) -> tp.Tuple[np.ndarray, np.ndarray]:
  upper = arr[r - 1, ...] if r > 0 else arr[r, ...]
  lower = arr[r + 1, ...] if r < arr.shape[0] - 1 else arr[r, ...]
  return upper, lower 

# returns rows of epsilon_l and epsilon_c for row i given image in LAB format
def find_el_ec(L: np.ndarray, A: np.ndarray, B: np.ndarray, i: int,
               mode: InterpolationMode) -> \
              tp.Tuple[np.ndarray, np.ndarray]:
  left_L, right_L = shifted_arrays(L[i, :])
  left_A, right_A = shifted_arrays(A[i, :])
  left_B, right_B = shifted_arrays(B[i, :])
  upper_L, lower_L = get_ul_row(L, i)
  upper_A, lower_A = get_ul_row(A, i)
  upper_B, lower_B = get_ul_row(B, i)

  # expression (13) and (14) from the article, but rewised
  # elementwise maximum - find it for each element
  if mode is InterpolationMode.HORIZONTAL:
    el_y = np.maximum(np.abs(L[i, :] - upper_L), np.abs(L[i, :] - lower_L))
    ec_y = np.maximum(np.sqrt(np.square(A[i, :] - upper_A) + np.square(B[i, :] - upper_B)),
                    np.sqrt(np.square(A[i, :] - lower_A) + np.square(B[i, :] - lower_B)))
    return el_y, ec_y
  elif mode is InterpolationMode.VERTICAL:
    el_x = np.maximum(np.abs(L[i, :] - left_L), np.abs(L[i, :] - right_L))
    ec_x = np.maximum(np.sqrt(np.square(A[i, :] - left_A) + np.square(B[i, :] - left_B)),
                    np.sqrt(np.square(A[i, :] - right_A) + np.square(B[i, :] - right_B)))
    return el_x, ec_x
  else:
    raise RuntimeError("Only InterpolationMode.HORIZONTAL | VERTICAL are supported")



def homogeneity_map(rgb_img: np.ndarray, mode: InterpolationMode):
  lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
  H, W = rgb_img.shape[0], rgb_img.shape[1]
  L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
  delta = 2
  H_out = np.zeros_like(L, dtype=np.float32)
  # считаем ec, el сразу построчно - не целиком, чтобы не переедать памяти, 
  # но и нe поэлементно, чтобы не тормозить. 
  # Вычисления все векторизуем, чтобы не совсем отставать от C++ реализаций
  col_idx = np.arange(W)
  l, r = col_idx - delta, col_idx + delta
  l[:delta] = 0
  r[-delta:] = W
  for i in range(H):

    el, ec = find_el_ec(L, A, B, i, mode)
    # l = max(col_idx - delta, 0), r = min(col_idx + delta, W), но более
    # эффективно
    top = max(0, i - delta)
    bottom = min(H, i + delta)
    # ex.4: number of neighbours = |B(x, delta)|. 
    # we find it for all elements in row
    total_nbrs = (bottom - top + 1) * (r - l)


    L_curr, A_curr, B_curr = L[top:bottom, :], A[top:bottom, :], B[top:bottom, :]
    # first delta elements
    for j in range(delta):
      L_j, A_j, B_j = L_curr[:, 0:j], A_curr[:, 0:j], B_curr[:, 0:j]
      L_local = np.abs(L_j - L[i][j]) < el[j]
      AB_local = np.sqrt((A_j - A[i][j]) ** 2 + (B_j - B[i][j]) ** 2) < ec[j]
      H_out[i][j] = (np.sum(L_local & AB_local) / total_nbrs[j])


    lab_mid = lab[top:bottom,:,:]
    slice_view = np.lib.stride_tricks.sliding_window_view(lab_mid,(2*delta+1),axis=1)
    # elements in range [delta, W-delta) can be processed vectorized
    # L_mid is (bottom - top + 1, W - 2 * delta, 2 * delta + 1)
    L_mid,A_mid,B_mid = slice_view[:,:,0,:], slice_view[:,:,1,:], slice_view[:,:,2,:]

    
    el_mid, ec_mid = el[delta:W-delta].reshape(1, -1, 1), \
                     ec[delta:W-delta].reshape(1, -1, 1)

  
    L_mid_local = np.abs(L_mid - L[i, delta:W-delta].reshape(1, -1, 1)) < el_mid
    AB_mid_local = np.sqrt((A_mid - A[i, delta:W-delta].reshape(1, -1, 1)) ** 2 + 
                           (B_mid - B[i, delta:W-delta].reshape(1, -1, 1)) ** 2) < ec_mid

    H_out[i][delta:W-delta] = np.sum(L_mid_local & AB_mid_local, axis=(0, 2)) / total_nbrs[delta:W-delta]
    
    # last delta elemens processed
    for j in range(W - delta, W):
      L_j, A_j, B_j = L_curr[:, j:], A_curr[:, j:], B_curr[:, j:]
      L_local = np.abs(L_j - L[i][j]) < el[j]
      AB_local = np.sqrt((A_j - A[i][j]) ** 2 + (B_j - B[i][j]) ** 2) < ec[j]
      H_out[i][j] = np.sum(L_local & AB_local) / total_nbrs[j]
  
  return H_out


def combine_images_with_maps(f_h: np.ndarray, f_v: np.ndarray, 
                             H_fh: np.ndarray, H_fv: np.ndarray) -> np.ndarray:

  assert f_h.shape == f_v.shape and f_v.shape == H_fh.shape + (3,) and \
         H_fh.shape == H_fv.shape, "same shape in first two dims"
  filter = np.ones((3,3), dtype=np.float32) / 9.
  H_fh_conv = signal.convolve2d(H_fh, filter, mode='same')
  H_fv_conv = signal.convolve2d(H_fv, filter, mode='same')
  h_mask = H_fh_conv >= H_fv_conv
  v_mask = ~h_mask
  result = np.zeros_like(f_h,dtype=np.float32)
  result[h_mask] = f_h[h_mask]
  result[v_mask] = f_v[v_mask]
  return result

def apply_median_filter(f: np.ndarray, times: int = 3, filter_size: int = 5) \
                        -> np.ndarray:
  assert filter_size % 2, "choose odd filter size, please"
  res = f.copy()
  R, G, B = res[:,:,0],res[:,:,1],res[:,:,2]
  for _ in range(times):
    RG_diff = median_filter(R - G, size=(filter_size,filter_size))
    BG_diff = median_filter(B - G, size=(filter_size,filter_size))
    GR_diff = -1. * RG_diff
    GB_diff = -1. * BG_diff
    
    R = RG_diff + G
    B = BG_diff + G
    G = 0.5 * (GR_diff + GB_diff + R + B)

  return np.stack((R, G, B), axis=-1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Restore images from CFA bitmap")
  parser.add_argument('input_file', action='store')
  parser.add_argument('output_file', action='store')
  parser.add_argument('--use_rgb', dest='rgb', action='store_true',
                       help="Input file is given in RGB format")
  parser.add_argument('--original_file', dest='orig', action='store', 
                      default="", required=False)

  args = parser.parse_args()

  img_type = ImageType.RGB_CFA if args.rgb else ImageType.CFA
  
  if img_type is ImageType.RGB_CFA:
    in_img = cv2.imread(args.input_file)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
  else:
    in_img = cv2.imread(args.input_file, cv2.IMREAD_GRAYSCALE)

  if args.orig:
    orig = cv2.imread(args.orig)
  start = time.time()

  f_h = interpolate_bayer(in_img, InterpolationMode.HORIZONTAL, img_type=img_type)
  f_v = interpolate_bayer(in_img, InterpolationMode.HORIZONTAL, img_type=img_type)
  
  H_fh = homogeneity_map(f_h, InterpolationMode.HORIZONTAL)
  H_fv = homogeneity_map(f_v, InterpolationMode.VERTICAL)

  pre_out = combine_images_with_maps(f_h, f_v, H_fh, H_fv)
  
  out = apply_median_filter(pre_out)
  out[out < 0] = 0
  out[out > 255] = 255

  end = time.time()
  out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

  cv2.imwrite(args.output_file, out_bgr)

  print("Algorithm processing time {0:0.3f}".format(end - start))
  if args.orig:
    mse = np.mean(np.square(out_bgr.astype(np.float32) 
                            - orig.astype(np.float32)), axis=(0,1))
    max_pix = np.max(out_bgr.astype(np.float32), axis=(0,1))

    color_psnr = 10. * np.log10((max_pix ** 2) / mse)
    print("PSNR for red: {0:0.3f}".format(color_psnr[2]))
    print("PSNR for green: {0:0.3f}".format(color_psnr[1]))
    print("PSNR for blue: {0:0.3f}".format(color_psnr[0]))

