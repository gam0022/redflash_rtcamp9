# Redflash [4D] Renderer

![pr33_v6_t3000_s1030_1920x1080](https://user-images.githubusercontent.com/759115/64941257-1549c080-d8a1-11e9-9cc6-a145bdaed7d8.png)

Redflash is a physics-based GPU renderer based on Path Tracing implemented in NVIDIA® OptiX 6.5, which can consistently draw scenes with mixed Polygons and **Raymarching**.

Redflash は NVIDIA® OptiX 6.5 上で実装したパストレーシングによる物理ベースのGPUレンダラーで、ポリゴンと **レイマーチング** が混在したシーンを一貫して描画できます。

## Note

This is implemented based on optixPathTracer of NVIDIA official OptiX-Samples.

これは、NVIDIA 公式の OptiX-Samples の optixPathTracer をベースにして実装されています。

The actual implementation is in the [redflash](https://github.com/gam0022/redflash_rtcamp9/tree/master/redflash) directory.

実際の実装は [redflash]([https://github.com/gam0022/redflash/tree/master/redflash](https://github.com/gam0022/redflash_rtcamp9/tree/main/redflash)) ディレクトリ内にあります。

## Features

- Unidirectional Path Tracing
  - Next Event Estimation (Direct Light Sampling)
  - Multiple Importance Sampling
- Materials
  - Disney BRDF
  - Lambert Diffuse
- Primitives
  - Sphere
  - Mesh
  - Distance Function ( **Raymarching** )
- ACES Filmic Tone Mapping
- Deep Learning Denoising

## Development Environment

- Operation confirmed
  - Windows 10 + NVIDIA RTX 2070
  - Windows 11 + NVIDIA RTX 3080
  - Windows Server 2016 Base + NVIDIA Tesla V100 GPUs
  - Windows_Server-2022-English-Full-Base + g4dn.xlarge
- Dependences
  - CUDA 10.1
  - OptiX 6.5.0
  - Cmake 3.24.0-rc5
  - freeglut

## Gallery

### RaytracingCamp9 Submission Version / レイトレ合宿9 提出バージョン

Won 4th prize at [レイトレ合宿9](https://sites.google.com/view/rtcamp9).

- [YouTube](https://www.youtube.com/watch?v=ohbv8_jCQtc)
- [YouTube Short](https://www.youtube.com/shorts/SgPbXt50Jw0)
- [Google Slides](https://docs.google.com/presentation/d/1f05HU58XD2w_71CJOdiEqOsBI8L2TYRTMndNT9MPqpI/edit#slide=id.gbd0ef54b81_0_79)
- [Speaker Deck](https://speakerdeck.com/gam0022/rtcamp9)

![435](https://github.com/gam0022/redflash_rtcamp9/assets/759115/133b831e-9876-4866-af02-09d2aa963f27)

![050](https://github.com/gam0022/redflash_rtcamp9/assets/759115/b57df7f9-ce0d-4b9a-9122-276144f6b741)

### RaytracingCamp8 Submission Version / レイトレ合宿8 提出バージョン

Won 5th prize at [レイトレ合宿8](https://sites.google.com/view/raytracingcamp8/).

- [YouTube](https://www.youtube.com/watch?v=c7JqEpaR658)
- https://github.com/gam0022/redflash_rtcamp8

![light_animation_960](https://user-images.githubusercontent.com/759115/196082478-7956c4f1-b433-49e5-87f8-38e2db83843c.gif)

![zoom-out_960](https://user-images.githubusercontent.com/759115/196172482-8bf54473-6e84-4e36-b167-b3c665d29761.gif)

![raymarching-animation_960](https://user-images.githubusercontent.com/759115/196082497-03638681-b194-43c2-b8e1-32fd8b1cf823.gif)

![menger](https://user-images.githubusercontent.com/759115/196082998-f5fba5ec-21e9-4ae9-a4e2-5cf18127f081.gif)


<!--
![menger_960](https://gam0022.net/images/posts/2022-09-26-rtcamp8/menger_960.gif)
-->


### RaytracingCamp7 Submission Version / レイトレ合宿7 提出バージョン

Won 4th prize at [レイトレ合宿7](https://sites.google.com/site/raytracingcamp7/).

- https://github.com/gam0022/redflash

![pr33_v6_t3000_s1030_1920x1080](https://user-images.githubusercontent.com/759115/64941257-1549c080-d8a1-11e9-9cc6-a145bdaed7d8.png)

#### Camera Angle Variation 1
![cut_far_v1](https://user-images.githubusercontent.com/759115/64941285-272b6380-d8a1-11e9-943c-7bf38f5e9538.png)

#### Camera Angle Variation 2

![cut_far_v2](https://user-images.githubusercontent.com/759115/64941286-2a265400-d8a1-11e9-84a4-245cfe70fed1.png)

#### Camera Angle Variation 3

![cut_far_v3](https://user-images.githubusercontent.com/759115/64941288-2b578100-d8a1-11e9-9494-8395a5310c6f.png)

## Links

- [レイトレ合宿8参加レポート | gam0022.net](https://gam0022.net/blog/2022/10/17/rtcamp8/)
- [レイトレ合宿7でレイマーチング対応のGPUパストレーサーを実装しました！ | gam0022.net](https://gam0022.net/blog/2019/09/18/rtcamp7/)
- [redflash renderer / Raytracing Camp 7 - Speaker Deck](https://speakerdeck.com/gam0022/raytracing-camp-7)
