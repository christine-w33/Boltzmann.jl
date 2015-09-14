
using CUDArt
using CUBLAS

abstract Backend

type CPUBackend <: Backend
end

type GPUBackend <: Backend
end
    


function main()
    N = 1_000
    ca = CudaArray(rand(Float64, 1000, 200));
    cb = CudaArray(rand(Float64, 200, 1000));
    cc = CudaArray(ones(Float64, 1000, 1000));
    @time for i=1:N
        CUBLAS.gemm!('N', 'N', 1.0, ca, cb, 0.0, cc)
    end

    a = rand(Float64, 1000, 200);
    b = rand(Float64, 200, 1000);
    c = ones(Float64, 1000, 1000);
    @time for i=1:N
        BLAS.gemm!('N', 'N', 1.0, a, b, 0.0, c)
    end
end
