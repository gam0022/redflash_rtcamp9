set srcDir=D:\Dropbox\RTCamp\redflash_rtcamp9\build\
set srcCudaDir=D:\Dropbox\RTCamp\redflash_rtcamp9\redflash\
set dstDir=D:\Dropbox\RTCamp\rtcamp9_submit_gpu\

cp %srcDir%bin\Release\prod_i0.bat %dstDir%
cp %srcDir%bin\Release\prod_i1.bat %dstDir%
cp %srcDir%bin\Release\redflash.exe %dstDir%
cp %srcDir%bin\Release\sutil_sdk.dll %dstDir%
cp %srcCudaDir%redflash.h %dstDir%cuda\
cp %srcCudaDir%redflash.cu %dstDir%cuda\
cp %srcCudaDir%bsdf_diffuse.cu %dstDir%cuda\
cp %srcCudaDir%bsdf_disney.cu %dstDir%cuda\
cp %srcCudaDir%intersect_raymarching.cu %dstDir%cuda\
cp %srcCudaDir%intersect_sphere.cu %dstDir%cuda\
cp %srcCudaDir%random.h %dstDir%cuda\