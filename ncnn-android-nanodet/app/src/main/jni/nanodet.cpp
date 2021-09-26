// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
{
    // 把原有的环境清空一下
    nanodet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodet.opt = ncnn::Option();

#if NCNN_VULKAN
    nanodet.opt.use_vulkan_compute = use_gpu;
#endif

    nanodet.opt.num_threads = ncnn::get_big_cpu_count();
    nanodet.opt.blob_allocator = &blob_pool_allocator;
    nanodet.opt.workspace_allocator = &workspace_pool_allocator;

    // 加载模型和设置模型对应的一些参数
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    nanodet.load_param(mgr, parampath);
    nanodet.load_model(mgr, modelpath);

    return 0;
}

int NanoDet::detect(const cv::Mat& rgb)
{
    ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, rgb.cols, rgb.rows);
    in.substract_mean_normalize(in_mean, in_norm);

    ncnn::Extractor ex = nanodet.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output4", out);

    out.substract_mean_normalize(out_mean, out_norm);
    cv::Mat res(cv::Size(rgb.cols,rgb.rows),CV_8UC1);
    out.to_pixels(res.data, ncnn::Mat::PIXEL_GRAY);

    cv::cvtColor(res,rgb,cv::COLOR_GRAY2RGB);

    return 0;
}
