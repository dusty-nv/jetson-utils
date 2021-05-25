#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__  = "hy"
__version__ = "1.00"
__date__    = "5 Sep 2020"

import time
import cv2
import numpy as np
import jetson.utils

# IMG_SIZE = [[640, 360], [720, 405], [800, 450], [864, 486], [1008, 567], [1024, 576], [1152, 648], [1280, 720], [1296, 729], [1440, 810], [1600, 900], [1920, 1080]]
# IMG_SIZE = [[640, 360], [1024, 576], [1280, 720], [1600, 900], [1920, 1080], [3840, 2160]]
# IMG_SIZE = [[640, 360]]
IMG_SIZE = [[1920, 1080]]
# IMG_SIZE = [[160, 90]]

class HogeHogeSplit(object):
    def __init__(self):
        self.src_gpu = cv2.cuda_GpuMat()
        self.dst_gpu = [cv2.cuda_GpuMat() for i in range(4)]

    def Go(self, src, device='cpu'):
        ch = src.shape[2]
        if device == 'cpu':
            # CPU version.
            dst = cv2.split(src)
        elif device == 'numpy':
            # NumPy version.
            dst = [src[:,:,i] for i in range(ch)]
        else:
            # GPU version.
            self.src_gpu.upload(src)
            self.dst_gpu = cv2.cuda.split(self.src_gpu)
            dst = [self.dst_gpu[i].download() for i in range(ch)]

        return dst

    def Go2(self, src, dst):
        # jetson-utils version.
        jetson.utils.cudaSplit(src, dst)
        jetson.utils.cudaDeviceSynchronize()

class HogeHogeMerge(object):
    def __init__(self):
        self.src_gpu = [cv2.cuda_GpuMat() for i in range(4)]
        self.dst_gpu = cv2.cuda_GpuMat()

    def Go(self, src, device='cpu'):
        ch = len(src)
        if device == 'cpu':
            # CPU version.
            dst = cv2.merge(src[::-1])
        elif device == 'numpy':
            # NumPy version.
            dst = np.zeros((src[0].shape[0], src[0].shape[1], ch), dtype = np.uint8)
            for i in range(ch):
                dst[...,i] = src[ch - i - 1]
        else:
            # GPU version.
            for i in range(ch):
                self.src_gpu[i].upload(src[ch - i - 1])
            self.dst_gpu = cv2.cuda.merge(self.src_gpu)
            dst = self.dst_gpu

        return dst

    def Go2(self, src, dst):
        # jetson-utils version.
        jetson.utils.cudaMerge(src[::-1], dst)
        jetson.utils.cudaDeviceSynchronize()

def test_split_merge():
    # 画像を取得
    # cpu, gpu. (BGR)
    img = cv2.imread("src.jpg")
    # jetson-utils. (RGBA)
    img2 = jetson.utils.loadImage('src.jpg', format='rgba8')

    # 3ch(BGR) -> 4ch(BGRA).
    img_a = np.full((img.shape[0], img.shape[1]), 255, dtype = np.uint8)
    img_4ch = np.zeros((img.shape[0], img.shape[1], 4), dtype = np.uint8)
    img_4ch[...,0:3] = img
    img_4ch[...,3]   = img_a

    # input size.
    in_X, in_Y = IMG_SIZE[0]
    # cpu, gpu.
    img_in = cv2.resize(img_4ch, (in_X, in_Y))
    # jetson-utils.
    img_in2 = jetson.utils.cudaAllocMapped(width=in_X, height=in_Y, format=img2.format)
    jetson.utils.cudaResize(img2, img_in2, jetson.utils.INTER_LINEAR)

    hoge_split = HogeHogeSplit()
    dummy_img_split = hoge_split.Go(img_in, 'gpu')   # 時間計測のため。CUDAの初期化時間をキャンセル。

    hoge_merge = HogeHogeMerge()
    dummy_img_out = hoge_merge.Go(dummy_img_split, 'gpu')   # 時間計測のため。CUDAの初期化時間をキャンセル。

    loop_cnt = 10000
    device_list = ['cpu', 'gpu', 'numpy', 'jetson-utils']

    # split(), merge()を計測.
    for d in device_list:   # CPU, CUDA, NumPy, jetson-utils.
        # split.
        if d != 'jetson-utils':
            img_IN = img_in
            start = time.time()
            for i in range(loop_cnt):
                img_SPLIT = hoge_split.Go(img_IN, d)
                # img_SPLIT = hoge_split.Go(img_IN, 'numpy')
            end_time = time.time() - start
        else:
            img_IN = img_in2
            # print(img_IN)
            X = img_IN.width
            Y = img_IN.height
            fmt = 'gray8'
            ch = img_IN.channels
            img_SPLIT = [jetson.utils.cudaAllocMapped(width=X, height=Y, format=fmt) for i in range(ch)]
            # print("img_SPLIT len:", len(img_SPLIT))
            # print(img_SPLIT)
            # for i in range(len(img_SPLIT)):
            #     print(img_SPLIT[i])
            start = time.time()
            for i in range(loop_cnt):
                hoge_split.Go2(img_IN, img_SPLIT)
            end_time = time.time() - start

        if d == 'cpu':
            print('split(CPU) end_time [msec],',end_time*1000/loop_cnt)
        elif d == 'gpu':
            print('split(CUDA) end_time [msec],',end_time*1000/loop_cnt)
        elif d == 'numpy':
            print('split(NumPy) end_time [msec],',end_time*1000/loop_cnt)
        else:
            print('split(jetson-utils) end_time [msec],',end_time*1000/loop_cnt)
            # jetson.utils.saveImage("dst_jetson-utils.jpg", img_SPLIT[0])



        # merge.
        if d != 'jetson-utils':
            start = time.time()
            for i in range(loop_cnt):
                img_OUT = hoge_merge.Go(img_SPLIT, d)
                # img_OUT = hoge_merge.Go(img_SPLIT, 'cpu')
            end_time = time.time() - start
        else:
            X = img_SPLIT[0].width
            Y = img_SPLIT[0].height
            fmt = 'rgba8' if len(img_SPLIT) == 4 else 'rgb8'
            img_OUT = jetson.utils.cudaAllocMapped(width=X, height=Y, format=fmt)
            # print(img_OUT)
            start = time.time()
            for i in range(loop_cnt):
                hoge_merge.Go2(img_SPLIT, img_OUT)
            end_time = time.time() - start

        if d == 'cpu':
            print('merge(CPU) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_cpu.jpg", img_OUT)
        elif d == 'gpu':
            print('merge(CUDA) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_gpu.jpg", img_OUT)
        elif d == 'numpy':
            print('merge(NumPy) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_numpy.jpg", img_OUT)
        else:
            print('merge(jetson-utils) end_time [msec],',end_time*1000/loop_cnt)
            jetson.utils.saveImage("dst_jetson-utils.jpg", img_OUT)



if __name__ == '__main__':
    test_split_merge()
