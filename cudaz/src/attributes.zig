const std = @import("std");
const cu = @import("cuda_cimports.zig").cu;

pub const Attribute = enum(c_uint) {
    /// Maximum number of threads per block
    MaxThreadsPerBlock = 1,
    /// Maximum block dimension X
    MaxBlockDimX = 2,
    /// Maximum block dimension Y
    MaxBlockDimY = 3,
    /// Maximum block dimension Z
    MaxBlockDimZ = 4,
    /// Maximum grid dimension X
    MaxGridDimX = 5,
    /// Maximum grid dimension Y
    MaxGridDimY = 6,
    /// Maximum grid dimension Z
    MaxGridDimZ = 7,
    /// Maximum shared memory available per block in bytes
    MaxSharedMemoryPerBlock = 8,
    /// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
    TotalConstantMemory = 9,
    /// Warp size in threads
    WarpSize = 10,
    /// Maximum pitch in bytes allowed by memory copies
    MaxPitch = 11,
    /// Maximum number of 32-bit registers available per block
    MaxRegistersPerBlock = 12,
    /// Typical clock frequency in kilohertz
    ClockRate = 13,
    /// Alignment requirement for textures
    TextureAlignment = 14,
    /// Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead AsyncEngineCount.
    GpuOverlap = 15,
    /// Number of multiprocessors on device
    MultiprocessorCount = 16,
    /// Specifies whether there is a run time limit on kernels
    KernelExecTimeout = 17,
    /// Device is integrated with host memory
    Integrated = 18,
    /// Device can map host memory into CUDA address space
    CanMapHostMemory = 19,
    /// Compute mode (See ::CUcomputemode for details)
    ComputeMode = 20,
    /// Maximum 1D texture width
    MaximumTexture1dWidth = 21,
    /// Maximum 2D texture width
    MaximumTexture2dWidth = 22,
    /// Maximum 2D texture height
    MaximumTexture2dHeight = 23,
    /// Maximum 3D texture width
    MaximumTexture3dWidth = 24,
    /// Maximum 3D texture height
    MaximumTexture3dHeight = 25,
    /// Maximum 3D texture depth
    MaximumTexture3dDepth = 26,
    /// Maximum 2D layered texture width
    MaximumTexture2dLayeredWidth = 27,
    /// Maximum 2D layered texture height
    MaximumTexture2dLayeredHeight = 28,
    /// Maximum layers in a 2D layered texture
    MaximumTexture2dLayeredLayers = 29,
    /// Alignment requirement for surfaces
    SurfaceAlignment = 30,
    /// Device can possibly execute multiple kernels concurrently
    ConcurrentKernels = 31,
    /// Device has ECC support enabled
    EccEnabled = 32,
    /// PCI bus ID of the device
    PciBusID = 33,
    /// PCI device ID of the device
    PciDeviceID = 34,
    /// Device is using TCC driver model
    TccDriver = 35,
    /// Peak memory clock frequency in kilohertz
    MemoryClockRate = 36,
    /// Global memory bus width in bits
    GlobalMemoryBusWidth = 37,
    /// Size of L2 cache in bytes
    L2CacheSize = 38,
    /// Maximum resident threads per multiprocessor
    MaxThreadsPerMultiprocessor = 39,
    /// Number of asynchronous engines
    AsyncEngineCount = 40,
    /// Device shares a unified address space with the host
    UnifiedAddressing = 41,
    /// Maximum 1D layered texture width
    MaximumTexture1dLayeredWidth = 42,
    /// Maximum layers in a 1D layered texture
    MaximumTexture1dLayeredLayers = 43,
    /// Deprecated, do not use.
    CanTex2dGather = 44,
    /// Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
    MaximumTexture2dGatherWidth = 45,
    /// Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
    MaximumTexture2dGatherHeight = 46,
    /// Alternate maximum 3D texture width
    MaximumTexture3dWidthAlternate = 47,
    /// Alternate maximum 3D texture height
    MaximumTexture3dHeightAlternate = 48,
    /// Alternate maximum 3D texture depth
    MaximumTexture3dDepthAlternate = 49,
    /// PCI domain ID of the device
    PciDomainID = 50,
    /// Pitch alignment requirement for textures
    TexturePitchAlignment = 51,
    /// Maximum cubemap texture width/height
    MaximumTexturecubemapWidth = 52,
    /// Maximum cubemap layered texture width/height
    MaximumTexturecubemapLayeredWidth = 53,
    /// Maximum layers in a cubemap layered texture
    MaximumTexturecubemapLayeredLayers = 54,
    /// Maximum 1D surface width
    MaximumSurface1dWidth = 55,
    /// Maximum 2D surface width
    MaximumSurface2dWidth = 56,
    /// Maximum 2D surface height
    MaximumSurface2dHeight = 57,
    /// Maximum 3D surface width
    MaximumSurface3dWidth = 58,
    /// Maximum 3D surface height
    MaximumSurface3dHeight = 59,
    /// Maximum 3D surface depth
    MaximumSurface3dDepth = 60,
    /// Maximum 1D layered surface width
    MaximumSurface1dLayeredWidth = 61,
    /// Maximum layers in a 1D layered surface
    MaximumSurface1dLayeredLayers = 62,
    /// Maximum 2D layered surface width
    MaximumSurface2dLayeredWidth = 63,
    /// Maximum 2D layered surface height
    MaximumSurface2dLayeredHeight = 64,
    /// Maximum layers in a 2D layered surface
    MaximumSurface2dLayeredLayers = 65,
    /// Maximum cubemap surface width
    MaximumSurfacecubemapWidth = 66,
    /// Maximum cubemap layered surface width
    MaximumSurfacecubemapLayeredWidth = 67,
    /// Maximum layers in a cubemap layered surface
    MaximumSurfacecubemapLayeredLayers = 68,
    /// Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
    MaximumTexture1dLinearWidth = 69,
    /// Maximum 2D linear texture width
    MaximumTexture2dLinearWidth = 70,
    /// Maximum 2D linear texture height
    MaximumTexture2dLinearHeight = 71,
    /// Maximum 2D linear texture pitch in bytes
    MaximumTexture2dLinearPitch = 72,
    /// Maximum mipmapped 2D texture width
    MaximumTexture2dMipmappedWidth = 73,
    /// Maximum mipmapped 2D texture height
    MaximumTexture2dMipmappedHeight = 74,
    /// Major compute capability version number
    ComputeCapabilityMajor = 75,
    /// Minor compute capability version number
    ComputeCapabilityMinor = 76,
    /// Maximum mipmapped 1D texture width
    MaximumTexture1dMipmappedWidth = 77,
    /// Device supports stream priorities
    StreamPrioritiesSupported = 78,
    /// Device supports caching globals in L1
    GlobalL1CacheSupported = 79,
    /// Device supports caching locals in L1
    LocalL1CacheSupported = 80,
    /// Maximum shared memory available per multiprocessor in bytes
    MaxSharedMemoryPerMultiprocessor = 81,
    /// Maximum number of 32-bit registers available per multiprocessor
    MaxRegistersPerMultiprocessor = 82,
    /// Device can allocate managed memory on this system
    ManagedMemory = 83,
    /// Device is on a multi-GPU board
    MultiGpuBoard = 84,
    /// Unique id for a group of devices on the same multi-GPU board
    MultiGpuBoardGroupID = 85,
    /// Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware
    HostNativeAtomicSupported = 86,
    /// Ratio of single precision performance (in floating-point operations per second) to double precision performance
    SingleToDoublePrecisionPerfRatio = 87,
    /// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
    PageableMemoryAccess = 88,
    /// Device can coherently access managed memory concurrently with the CPU
    ConcurrentManagedAccess = 89,
    /// Device supports compute preemption.
    ComputePreemptionSupported = 90,
    /// Device can access host registered memory at the same virtual address as the CPU
    CanUseHostPointerForRegisteredMem = 91,
    /// ::cuStreamBatchMemOp and related APIs are supported.
    CanUseStreamMemOps = 92,
    /// 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs.
    CanUse64BitStreamMemOps = 93,
    /// ::CU_STREAM_WAIT_VALUE_NOR is supported.
    CanUseStreamWaitValueNor = 94,
    /// Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel
    CooperativeLaunch = 95,
    /// Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated.
    CooperativeMultiDeviceLaunch = 96,
    /// Maximum optin shared memory per block
    MaxSharedMemoryPerBlockOptin = 97,
    /// The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details.
    CanFlushRemoteWrites = 98,
    /// Device supports host memory registration via ::cudaHostRegister.
    HostRegisterSupported = 99,
    /// Device accesses pageable memory via the host's page tables.
    PageableMemoryAccessUsesHostPageTables = 100,
    /// The host can directly access managed memory on the device without migration.
    DirectManagedMemAccessFromHost = 101,
    /// Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
    VirtualMemoryManagementSupported = 102,
    /// Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    HandleTypePosixFileDescriptorSupported = 103,
    /// Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    HandleTypeWin32HandleSupported = 104,
    /// Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested ::cuMemCreate
    HandleTypeWin32KmtHandleSupported = 105,
    /// Maximum number of blocks per multiprocessor
    MaxBlocksPerMultiprocessor = 106,
    /// Device supports compression of memory
    GenericCompressionSupported = 107,
    /// Maximum L2 persisting lines capacity setting in bytes.
    MaxPersistingL2CacheSize = 108,
    /// Maximum value of CUaccessPolicyWindow::num_bytes.
    MaxAccessPolicyWindowSize = 109,
    /// Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate
    GpuDirectRdmaWithCudaVmmSupported = 110,
    /// Shared memory reserved by CUDA driver per block in bytes
    ReservedSharedMemoryPerBlock = 111,
    /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
    SparseCudaArraySupported = 112,
    /// Device supports using the ::cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU
    ReadOnlyHostRegisterSupported = 113,
    /// External timeline semaphore interop is supported on the device
    TimelineSemaphoreInteropSupported = 114,
    /// Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs
    MemoryPoolsSupported = 115,
    /// Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
    GpuDirectRdmaSupported = 116,
    /// The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum
    GpuDirectRdmaFlushWritesOptions = 117,
    /// GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
    GpuDirectRdmaWritesOrdering = 118,
    /// Handle types supported with mempool based IPC
    MempoolSupportedHandleTypes = 119,
};

// TODO: take a CUdevice here, and expose device in the Stream object
pub fn getAttr(device: u8, attr: Attribute) i32 {
    var d: cu.CUdevice = undefined;
    _ = cu.cuDeviceGet(&d, device);
    var value: i32 = std.math.minInt(i32);
    _ = cu.cuDeviceGetAttribute(&value, @enumToInt(attr), d);
    return value;
}
