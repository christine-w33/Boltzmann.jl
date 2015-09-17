
using CUDArt
using CUBLAS

abstract Backend

## CPU Backend

type CPUBackend <: Backend
end

function make_array{T,N}(backend::CPUBackend, arr::AbstractArray{T,N})
    return arr
end

## GPU Backend

type GPUBackend <: Backend
end


function make_array{T,N}(backend::GPUBackend, arr::AbstractArray{T,N})
    return CudaArray(arr)
end




function main()
    N = 1_000    

    a = rand(Float64, 1000, 200);
    b = rand(Float64, 200, 1000);
    c = ones(Float64, 1000, 1000);
    r = Array(Float64, 0)
    @time for i=1:N
        r = BLAS.gemm!('N', 'N', 1.0, a, b, 0.0, c)
    end

    ca = CudaArray(a);
    cb = CudaArray(b);
    cc = CudaArray(c);
    cr = CudaArray(Array(Float64,0));
    @time for i=1:N
        cr = CUBLAS.gemm!('N', 'N', 1.0, ca, cb, 0.0, cc)
    end
end
