# About app

This application allows you to multiply square matrices of size N x N using CPU and GPU.

For writing, Visual Studio 2015 and Cuda Toolkit 8 were used.

## Using
1. Install CUDA on your computer (required to have a video card from Nvidia).
2. Create new CUDA project in Visual Studio.
3. Copy this code.
4. Run the application.

## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | AMD Ryzen 5 2600 Six-Core Processor 3.4 GHz (Turbo Boost 3.9 GHz) |
| RAM  | 16 GB DDR4 |
| GPU  | GIGABYTE GeForce GTX 550 Ti [GV-N550D5-1GI]  |
| OS   | Windows 10 64-bit  |

## Results

The average time in milliseconds for 5 measurements is presented in the table. Matrix elements are of type float.

|    Size     |          CPU        |         GPU       | Acceleration |
|-------------|---------------------|-------------------|--------------|
| 256 х 256   | 43 ms               | 2.7 ms            |    15.9      |
| 512 х 512   | 340 ms              | 21 ms             |    16.2      |
| 1024 х 1024 | 3183 ms = 3.183 s   | 170 ms            |    18.7      |
| 2048 х 2048 | 31240 ms = 31.240 s | 1365 ms = 1.365 s |    22.8      |
