const std = @import("std");

const cu = @import("cuda_h");

pub const Attribute = enum(c_uint) {
    /// Maximum number of threads per block
    max_threads_per_block = 1,
    /// Maximum block dimension X
    max_block_dim_x = 2,
    /// Maximum block dimension Y
    max_block_dim_y = 3,
    /// Maximum block dimension Z
    max_block_dim_z = 4,
    /// Maximum grid dimension X
    max_grid_dim_x = 5,
    /// Maximum grid dimension Y
    max_grid_dim_y = 6,
    /// Maximum grid dimension Z
    max_grid_dim_z = 7,
    /// Maximum shared memory available per block in bytes
    max_shared_memory_per_block = 8,
    /// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
    total_constant_memory = 9,
    /// Warp size in threads
    warp_size = 10,
    /// Maximum pitch in bytes allowed by memory copies
    max_pitch = 11,
    /// Maximum number of 32-bit registers available per block
    max_registers_per_block = 12,
    /// Typical clock frequency in kilohertz
    clock_rate = 13,
    /// Alignment requirement for textures
    texture_alignment = 14,
    /// Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead AsyncEngineCount.
    gpu_overlap = 15,
    /// Number of multiprocessors on device
    multiprocessor_count = 16,
    /// Specifies whether there is a run time limit on kernels
    kernel_exec_timeout = 17,
    /// Device is integrated with host memory
    integrated = 18,
    /// Device can map host memory into CUDA address space
    can_map_host_memory = 19,
    /// Compute mode (See ::CUcomputemode for details)
    compute_mode = 20,
    /// Maximum 1D texture width
    maximum_texture1d_width = 21,
    /// Maximum 2D texture width
    maximum_texture2d_width = 22,
    /// Maximum 2D texture height
    maximum_texture2d_height = 23,
    /// Maximum 3D texture width
    maximum_texture3d_width = 24,
    /// Maximum 3D texture height
    maximum_texture3d_height = 25,
    /// Maximum 3D texture depth
    maximum_texture3d_depth = 26,
    /// Maximum 2D layered texture width
    maximum_texture2d_layered_width = 27,
    /// Maximum 2D layered texture height
    maximum_texture2d_layered_height = 28,
    /// Maximum layers in a 2D layered texture
    maximum_texture2d_layered_layers = 29,
    /// Alignment requirement for surfaces
    surface_alignment = 30,
    /// Device can possibly execute multiple kernels concurrently
    concurrent_kernels = 31,
    /// Device has ECC support enabled
    ecc_enabled = 32,
    /// PCI bus ID of the device
    pci_bus_id = 33,
    /// PCI device ID of the device
    pci_device_id = 34,
    /// Device is using TCC driver model
    tcc_driver = 35,
    /// Peak memory clock frequency in kilohertz
    memory_clock_rate = 36,
    /// Global memory bus width in bits
    global_memory_bus_width = 37,
    /// Size of L2 cache in bytes
    l2_cache_size = 38,
    /// Maximum resident threads per multiprocessor
    max_threads_per_multiprocessor = 39,
    /// Number of asynchronous engines
    async_engine_count = 40,
    /// Device shares a unified address space with the host
    unified_addressing = 41,
    /// Maximum 1D layered texture width
    maximum_texture1d_layered_width = 42,
    /// Maximum layers in a 1D layered texture
    maximum_texture1d_layered_layers = 43,
    /// Deprecated, do not use.
    can_tex2d_gather = 44,
    /// Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
    maximum_texture2d_gather_width = 45,
    /// Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
    maximum_texture2d_gather_height = 46,
    /// Alternate maximum 3D texture width
    maximum_texture3d_width_alternate = 47,
    /// Alternate maximum 3D texture height
    maximum_texture3d_height_alternate = 48,
    /// Alternate maximum 3D texture depth
    maximum_texture3d_depth_alternate = 49,
    /// PCI domain ID of the device
    pci_domain_id = 50,
    /// Pitch alignment requirement for textures
    texture_pitch_alignment = 51,
    /// Maximum cubemap texture width/height
    maximum_texturecubemap_width = 52,
    /// Maximum cubemap layered texture width/height
    maximum_texturecubemap_layered_width = 53,
    /// Maximum layers in a cubemap layered texture
    maximum_texturecubemap_layered_layers = 54,
    /// Maximum 1D surface width
    maximum_surface1d_width = 55,
    /// Maximum 2D surface width
    maximum_surface2d_width = 56,
    /// Maximum 2D surface height
    maximum_surface2d_height = 57,
    /// Maximum 3D surface width
    maximum_surface3d_width = 58,
    /// Maximum 3D surface height
    maximum_surface3d_height = 59,
    /// Maximum 3D surface depth
    maximum_surface3d_depth = 60,
    /// Maximum 1D layered surface width
    maximum_surface1d_layered_width = 61,
    /// Maximum layers in a 1D layered surface
    maximum_surface1d_layered_layers = 62,
    /// Maximum 2D layered surface width
    maximum_surface2d_layered_width = 63,
    /// Maximum 2D layered surface height
    maximum_surface2d_layered_height = 64,
    /// Maximum layers in a 2D layered surface
    maximum_surface2d_layered_layers = 65,
    /// Maximum cubemap surface width
    maximum_surfacecubemap_width = 66,
    /// Maximum cubemap layered surface width
    maximum_surfacecubemap_layered_width = 67,
    /// Maximum layers in a cubemap layered surface
    maximum_surfacecubemap_layered_layers = 68,
    /// Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
    maximum_texture1d_linear_width = 69,
    /// Maximum 2D linear texture width
    maximum_texture2d_linear_width = 70,
    /// Maximum 2D linear texture height
    maximum_texture2d_linear_height = 71,
    /// Maximum 2D linear texture pitch in bytes
    maximum_texture2d_linear_pitch = 72,
    /// Maximum mipmapped 2D texture width
    maximum_texture2d_mipmapped_width = 73,
    /// Maximum mipmapped 2D texture height
    maximum_texture2d_mipmapped_height = 74,
    /// Major compute capability version number
    compute_capability_major = 75,
    /// Minor compute capability version number
    compute_capability_minor = 76,
    /// Maximum mipmapped 1D texture width
    maximum_texture1d_mipmapped_width = 77,
    /// Device supports stream priorities
    stream_priorities_supported = 78,
    /// Device supports caching globals in L1
    global_l1_cache_supported = 79,
    /// Device supports caching locals in L1
    local_l1_cache_supported = 80,
    /// Maximum shared memory available per multiprocessor in bytes
    max_shared_memory_per_multiprocessor = 81,
    /// Maximum number of 32-bit registers available per multiprocessor
    max_registers_per_multiprocessor = 82,
    /// Device can allocate managed memory on this system
    managed_memory = 83,
    /// Device is on a multi-GPU board
    multi_gpu_board = 84,
    /// Unique id for a group of devices on the same multi-GPU board
    multi_gpu_board_group_id = 85,
    /// Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware
    host_native_atomic_supported = 86,
    /// Ratio of single precision performance (in floating-point operations per second) to double precision performance
    single_to_double_precision_perf_ratio = 87,
    /// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
    pageable_memory_access = 88,
    /// Device can coherently access managed memory concurrently with the CPU
    concurrent_managed_access = 89,
    /// Device supports compute preemption.
    compute_preemption_supported = 90,
    /// Device can access host registered memory at the same virtual address as the CPU
    can_use_host_pointer_for_registered_mem = 91,
    /// ::cuStreamBatchMemOp and related APIs are supported.
    can_use_stream_mem_ops_v1 = 92,
    /// 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs.
    can_use64_bit_stream_mem_ops_v1 = 93,
    /// ::CU_STREAM_WAIT_VALUE_NOR is supported.
    can_use_stream_wait_value_nor_v1 = 94,
    /// Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel
    cooperative_launch = 95,
    /// Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated.
    cooperative_multi_device_launch = 96,
    /// Maximum optin shared memory per block
    max_shared_memory_per_block_optin = 97,
    /// The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details.
    can_flush_remote_writes = 98,
    /// Device supports host memory registration via ::cudaHostRegister.
    host_register_supported = 99,
    /// Device accesses pageable memory via the host's page tables.
    pageable_memory_access_uses_host_page_tables = 100,
    /// The host can directly access managed memory on the device without migration.
    direct_managed_mem_access_from_host = 101,
    /// Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
    virtual_memory_management_supported = 102,
    /// Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    handle_type_posix_file_descriptor_supported = 103,
    /// Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
    handle_type_win32_handle_supported = 104,
    /// Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested ::cuMemCreate
    handle_type_win32_kmt_handle_supported = 105,
    /// Maximum number of blocks per multiprocessor
    max_blocks_per_multiprocessor = 106,
    /// Device supports compression of memory
    generic_compression_supported = 107,
    /// Maximum L2 persisting lines capacity setting in bytes.
    max_persisting_l2_cache_size = 108,
    /// Maximum value of CUaccessPolicyWindow::num_bytes.
    max_access_policy_window_size = 109,
    /// Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate
    gpu_direct_rdma_with_cuda_vmm_supported = 110,
    /// Shared memory reserved by CUDA driver per block in bytes
    reserved_shared_memory_per_block = 111,
    /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
    sparse_cuda_array_supported = 112,
    /// Device supports using the ::cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU
    read_only_host_register_supported = 113,
    /// External timeline semaphore interop is supported on the device
    timeline_semaphore_interop_supported = 114,
    /// Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs
    memory_pools_supported = 115,
    /// Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
    gpu_direct_rdma_supported = 116,
    /// The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum
    gpu_direct_rdma_flush_writes_options = 117,
    /// GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
    gpu_direct_rdma_writes_ordering = 118,
    /// Handle types supported with mempool based IPC
    mempool_supported_handle_types = 119,
    /// Indicates device supports cluster launch
    cluster_launch = 120,
    /// Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
    deferred_mapping_cuda_array_supported = 121,
    /// 64-bit operations are supported in ::cuStreamBatchMemOp and related MemOp APIs.
    can_use_64_bit_stream_mem_ops = 122,
    /// ::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
    can_use_stream_wait_value_nor = 123,
    /// Device supports buffer sharing with dma_buf mechanism.
    dma_buf_supported = 124,
    /// Device supports IPC Events.
    ipc_event_supported = 125,
    /// Number of memory domains the device supports.
    mem_sync_domain_count = 126,
    /// Device supports accessing memory using Tensor Map.
    tensor_map_access_supported = 127,
    /// Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() requested with cuMemCreate()
    handle_type_fabric_supported = 128,
    /// Device supports unified function pointers.
    unified_function_pointers = 129,
    /// NUMA configuration of a device: value is of type ::CUdeviceNumaConfig enum
    numa_config = 130,
    /// NUMA node ID of the GPU memory
    numa_id = 131,
    /// Device supports switch multicast and reduction operations.
    multicast_supported = 132,
    /// Indicates if contexts created on this device will be shared via MPS
    mps_enabled = 133,
    /// NUMA ID of the host node closest to the device. Returns -1 when system does not suppoNUMA.
    host_numa_id = 134,
    /// Device supports CIG with D3D12.
    d3d12_cig_supported = 135,
    /// The returned valued shall be interpreted as a bitmask, where the individual bits adescribed by the ::CUmemDecompressAlgorithm enum.
    mem_decompress_algorithm_mask = 136,
    /// The returned valued is the maximum length in bytes of a single decompress operation that allowed.
    mem_decompress_maximum_length = 137,
    /// Device supports CIG with Vulkan.
    vulkan_cig_supported = 138,
    /// The combined 16-bit PCI device ID and 16-bit PCI vendor ID.
    gpu_pci_device_id = 139,
    /// The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.
    gpu_pci_subsystem_id = 140,
    /// Device supports HOST_NUMA location with the virtual memory management APIs li::cuMemCreate, ::cuMemMap and related APIs
    host_numa_virtual_memory_management_supported = 141,
    /// Device supports HOST_NUMA location with the ::cuMemAllocAsync and ::cuMemPool family APIs
    host_numa_memory_pools_supported = 142,
    /// Device supports HOST_NUMA location IPC between nodes in a multi-node system.
    host_numa_multinode_ipc_supported = 143,
};

// TODO: take a CUdevice here, and expose device in the Stream object
pub fn getAttr(device: cu.CUdevice, attr: Attribute) u32 {
    var value: c_int = 0;
    _ = cu.cuDeviceGetAttribute(&value, @intFromEnum(attr), device);
    return @intCast(value);
}
