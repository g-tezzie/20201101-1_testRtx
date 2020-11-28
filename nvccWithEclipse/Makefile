#1>C:\Users\gtezz\source\repos\20201101-1_testRtx\20201101-1_testRtx>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin\nvcc.exe"
# -gencode=arch=compute_75,code=\"sm_75,compute_75\" --use-local-env
# -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX86\x64" -x cu  
# -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include"
# -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include"
# --source-in-ptx   --keep --keep-dir x64\Debug -maxrregcount=0 --ptxas-options=-v --machine 64 --compile -cudart static  -g  -use_fast_math
# -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Ox /Fdx64\Debug\vc142.pdb /FS /Zi  /MDd "
# -o x64\Debug\kernel.cu.obj "C:\Users\gtezz\source\repos\20201101-1_testRtx\20201101-1_testRtx\kernel.cu"

#1>C:\Users\gtezz\source\repos\20201101-1_testRtx\20201101-1_testRtx>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin\nvcc.exe"
# -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX86\x64" -x cu  
# -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include"
# -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include"
# --source-in-ptx   --keep --keep-dir x64\Debug -maxrregcount=0 --ptxas-options=-v --machine 64 --compile   -g  -use_fast_math
# -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Ox  /FS /Zi  /MDd "
# -o x64\Debug\kernel.cu.obj "C:\Users\gtezz\source\repos\20201101-1_testRtx\20201101-1_testRtx\kernel.cu"
# -gencode=arch=compute_75,code=\"sm_75,compute_75\" -clean

# /OUT:"C:\Users\gtezz\source\repos\20201101-1_testRtx\x64\Debug\20201101-1_testRtx.exe"
# /MANIFEST /NXCOMPAT /PDB:"C:\Users\gtezz\source\repos\20201101-1_testRtx\x64\Debug\20201101-1_testRtx.pdb"
# /DYNAMICBASE "cudart_static.lib" "kernel32.lib" "user32.lib" "gdi32.lib" "winspool.lib" "comdlg32.lib" "advapi32.lib" "shell32.lib" "ole32.lib" "oleaut32.lib" "uuid.lib" "odbc32.lib" "odbccp32.lib" "cudart.lib" "cudadevrt.lib"
# /DEBUG /MACHINE:X64 /INCREMENTAL
# /PGD:"C:\Users\gtezz\source\repos\20201101-1_testRtx\x64\Debug\20201101-1_testRtx.pgd"
# /SUBSYSTEM:CONSOLE /MANIFESTUAC:"level='asInvoker' uiAccess='false'"
# /ManifestFile:"x64\Debug\20201101-1_testRtx.exe.intermediate.manifest" /ERRORREPORT:PROMPT /NOLOGO
# /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64" /TLBID:1 

PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = $(subst $(abspath $(PROJECT_ROOT)..)/,,$(abspath $(PROJECT_ROOT)))

OBJS = nvccWithMake.o

# CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1
NVCC = "$(CUDA_PATH)/bin/nvcc"
# NVCCFLAGS = -ccbin "$(VCPATH)" -x cu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include" --source-in-ptx --keep --keep-dir x64\Debug -maxrregcount=0 --ptxas-options=-v --machine 64 --compile   -g  -use_fast_math -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Ox  /FS /Zi  /MDd " -gencode=arch=compute_75,code=\"sm_75,compute_75\"
#NVCCFLAGS = -ccbin "$(VCPATH)" -x cu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include" --source-in-ptx --keep -maxrregcount=0 --ptxas-options=-v --machine 64 --compile   -g  -use_fast_math -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler " /EHsc /W3 /nologo /Ox  /FS /Zi  /MDd " -gencode=arch=compute_75,code=\"sm_75,compute_75\"
NVCCFLAGS = -ccbin "$(VCPATH)" -x cu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include" --source-in-ptx --keep -maxrregcount=0 --ptxas-options=-v --machine 64 --compile    -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler " /EHsc /W3 /nologo /Ox  /FS /Zi  /MDd " $(NVARCH) 
NVARCH = -g  -use_fast_math -gencode=arch=compute_75,code=\"sm_75,compute_75\"
#NVLDFLAGS = $(NVARCH) --library-path="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64" --library=cudart_static --library=cudart --library=cudadevrt
#NVLDFLAGS = --linker-options '/LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64" /DYNAMICBASE "cudart_static.lib" "kernel32.lib" "user32.lib" "gdi32.lib" "winspool.lib" "comdlg32.lib" "advapi32.lib" "shell32.lib" "ole32.lib" "oleaut32.lib" "uuid.lib" "odbc32.lib" "odbccp32.lib" "cudart.lib" "cudadevrt.lib" "msvcrt.lib"'

LDFLAGS = -lgomp -lstdc++ \
	"$(CUDA_PATH)/lib/x64/cudart_static.lib" \
	"$(CUDA_PATH)/lib/x64/cudart.lib" \
	"$(CUDA_PATH)/lib/x64/cudadevrt.lib"


CFLAGS = -O3 -Ofast -march=native -mtune=native -mfpmath=sse -ffast-math -fopenmp -g3 -Wall -c -fmessage-length=0 

VSPATH = C:\Program Files (x86)\Microsoft Visual Studio\2019\Community
VCPATH = $(VSPATH)\VC\Tools\MSVC\14.27.29110\bin\HostX86\x64
# PATH += C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\\Extensions\Microsoft\IntelliCode\CLI;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\VC\VCPackages;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\TestWindow;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\bin\Roslyn;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Team Tools\Performance Tools\x64;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Team Tools\Performance Tools;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\\x64;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\;C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64;C:\Program Files (x86)\Windows Kits\10\bin\x64;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\\MSBuild\Current\Bin;C:\Windows\Microsoft.NET\Framework64\v4.0.30319;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\libnvvp;;;;;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\libnvvp;;C:\Program Files (x86)\Common Files\Oracle\Java\javapath;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\;C:\Program Files\PuTTY\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\CMake\bin;C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR;C:\Program Files\NVIDIA Corporation\Nsight Compute 2020.2.1\;C:\Users\gtezz\AppData\Local\Microsoft\WindowsApps;;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja
PATH += $(VSPATH)\Common7\IDE\\Extensions\Microsoft\IntelliCode\CLI;$(VSPATH)\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64;$(VSPATH)\Common7\IDE\VC\VCPackages;$(VSPATH)\Common7\IDE\CommonExtensions\Microsoft\TestWindow;$(VSPATH)\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;$(VSPATH)\MSBuild\Current\bin\Roslyn;$(VSPATH)\Team Tools\Performance Tools\x64;$(VSPATH)\Team Tools\Performance Tools;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\\x64;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\;C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64;C:\Program Files (x86)\Windows Kits\10\bin\x64;$(VSPATH)\\MSBuild\Current\Bin;C:\Windows\Microsoft.NET\Framework64\v4.0.30319;$(VSPATH)\Common7\IDE\;$(VSPATH)\Common7\Tools\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\libnvvp;;;;;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\libnvvp;;C:\Program Files (x86)\Common Files\Oracle\Java\javapath;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\;C:\Program Files\PuTTY\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\CMake\bin;C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR;C:\Program Files\NVIDIA Corporation\Nsight Compute 2020.2.1\;C:\Users\gtezz\AppData\Local\Microsoft\WindowsApps;;$(VSPATH)\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$(VSPATH)\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja


ifeq ($(BUILD_MODE),debug)
	CFLAGS += -g
else ifeq ($(BUILD_MODE),run)
	CFLAGS += -O2
else ifeq ($(BUILD_MODE),linuxtools)
	CFLAGS += -g -pg -fprofile-arcs -ftest-coverage
	LDFLAGS += -pg -fprofile-arcs -ftest-coverage
else
    $(error Build mode $(BUILD_MODE) not supported by this Makefile)
endif

.PRECIOUS: %.o %.cu.o

all:	$(PROJECT_NAME).exe

CPP_SOURCES = $(wildcard $(PROJECT_ROOT)*.cpp)
CU_SOURCES = $(wildcard $(PROJECT_ROOT)*.cu)

OBJS = $(subst $(PROJECT_ROOT),,$(CPP_SOURCES:.cpp=.o) $(CU_SOURCES:.cu=.cu.o))

#%.exe: %.cu.o
#nvccWithMake:	$(OBJS)
$(PROJECT_NAME).exe: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^
#	$(NVCC) $(NVLDFLAGS) -o $@ $^
#	echo $(OBJS)
#	echo $@


%.cu.o: $(PROJECT_ROOT)%.cu
#	$(NVCC) -c $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<
	$(NVCC) $(NVCCFLAGS) -o $@ $<

%.o:	$(PROJECT_ROOT)%.cpp
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

#%.o:	$(PROJECT_ROOT)%.c
#	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

#clean:
#	rm -fr nvccWithMake $(OBJS)
