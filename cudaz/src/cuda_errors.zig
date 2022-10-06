const std = @import("std");
const log = std.log.scoped(.Cuda);

pub const cu = @import("cuda_cimports.zig").cu;

pub const Error = error{
    OutOfMemory,
    LaunchOutOfResources,
    UnexpectedByCudaz,
    NotSupported,
    NotReady,
    // TODO: add more generic errors to allow forward compatibilities with more
    // Cuda versions / Cudaz development.
};

/// Converts the Cuda result code into a simplified Zig error.
/// The goal is to only surface errors that are recoverable and aren't due to a
/// bug in caller code, library code or kernel code.
/// Cuda can yield a lot of different errors, but there are different categories.
/// Note that in any case we will log the detailed error and its message.
/// * Resource errors: typically out of memory.
///     -> original error (OutOfMemory, ...)
/// * Device capability errors: the current device doesn't support this function.
///     -> NotSupported
/// * API usage errors: errors that will deterministically be returned
///     when the API is misused. Cudaz is allowed to panic here.
///     Cudaz aims at preventing this kind of error by providing an API harder to misuses.
///     Typically calling Cuda methods before calling cuInit().
///     This can also be due to passing host pointer to functions expecting device pointer.
///     -> @panic
/// * Message passing: not actual error, but things that should be retried.
///     eg: CUDA_ERROR_NOT_READY
///     -> NotReady
/// * Bug in Cudaz: Cudaz is creating a default context and loading the compiled
///     Cuda code for you. Those operation shouldn't fail unless bug in Cudaz.
///     -> @panic
/// * Kernel execution error:
///     There was a bug during kernel execution (stackoverflow, segfault, ...)
///     Typically those leave the driver in unusable state, the process must be restarted.
///     Cudaz will panic here.
///     This will typically not trigger at the expected place because they will
///     happen asynchronously to the host code execution.
///     -> @panic
/// * Unhandled errors:
///     Cudaz doesn't support the full Cuda API (yet ?). Trying to use Cudaz
///     check on errors returned by unsupported part of the API will trigger a @panic
///     (this will only happen if you directly call cuda). Feel free to open PR
///     to support more parts of Cuda
///     -> @panic + dedicated error log
pub fn check(result: cu.CUresult) Error!void {
    if (result == cu.CUDA_SUCCESS) return;
    log_err_message(result);
    return silent_check(result);
}

pub fn silent_check(result: cu.CUresult) Error!void {
    var err: Error = switch (result) {
        cu.CUDA_SUCCESS => return,
        // Resource errors:
        cu.CUDA_ERROR_OUT_OF_MEMORY => error.OutOfMemory,
        // Device capability error:
        cu.CUDA_ERROR_STUB_LIBRARY,
        cu.CUDA_ERROR_NO_DEVICE,
        cu.CUDA_ERROR_INVALID_DEVICE,
        cu.CUDA_ERROR_DEVICE_NOT_LICENSED,
        cu.CUDA_ERROR_NOT_PERMITTED, // TODO: make a distinctions for permission ?
        cu.CUDA_ERROR_NOT_SUPPORTED,
        cu.CUDA_ERROR_SYSTEM_NOT_READY,
        cu.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
        cu.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE,
        cu.CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
        => error.NotSupported,
        // LAUNCH_OUT_OF_RESOURCES can indicate either that the too many threads
        // where requested wrt to the maximum supported by the GPU.
        // It can also be triggered by passing too many args to a kernel,
        // but this should be caught at compile time by Cudaz, so we will ignore this.
        cu.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => error.NotSupported,
        // API usage errors:
        cu.CUDA_ERROR_INVALID_VALUE => @panic("Received invalid parameters (typically device/host pointer mismatch"),
        cu.CUDA_ERROR_NOT_INITIALIZED,
        cu.CUDA_ERROR_DEINITIALIZED,
        cu.CUDA_ERROR_PROFILER_NOT_INITIALIZED,
        cu.CUDA_ERROR_PROFILER_ALREADY_STARTED,
        cu.CUDA_ERROR_PROFILER_ALREADY_STOPPED,
        cu.CUDA_ERROR_PROFILER_DISABLED,
        cu.CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
        cu.CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
        cu.CUDA_ERROR_INVALID_HANDLE,
        => @panic("Invalid API usage"),
        // Bug in Cudaz:
        cu.CUDA_ERROR_INVALID_IMAGE,
        cu.CUDA_ERROR_INVALID_CONTEXT,
        cu.CUDA_ERROR_NO_BINARY_FOR_GPU,
        cu.CUDA_ERROR_INVALID_PTX,
        cu.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
        cu.CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
        cu.CUDA_ERROR_INVALID_SOURCE,
        cu.CUDA_ERROR_FILE_NOT_FOUND,
        cu.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
        cu.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
        cu.CUDA_ERROR_NOT_FOUND,
        => @panic("Something looks wrong with compiled device code"),
        // Kernel execution error:
        cu.CUDA_ERROR_ECC_UNCORRECTABLE,
        cu.CUDA_ERROR_HARDWARE_STACK_ERROR,
        cu.CUDA_ERROR_ILLEGAL_INSTRUCTION,
        cu.CUDA_ERROR_MISALIGNED_ADDRESS,
        cu.CUDA_ERROR_ILLEGAL_ADDRESS,
        cu.CUDA_ERROR_INVALID_ADDRESS_SPACE,
        cu.CUDA_ERROR_INVALID_PC,
        cu.CUDA_ERROR_LAUNCH_FAILED,
        cu.CUDA_ERROR_EXTERNAL_DEVICE,
        => @panic("Unrecoverable error while running device code"),
        // Unhandled errors:
        // Map, Stream, Graph
        else => @panic("Part of the API not handled by Cudaz"),
    };
    return err;
}

pub fn error_name(result: cu.CUresult) []const u8 {
    return switch (result) {
        cu.CUDA_SUCCESS => "CUDA_SUCCESS",
        cu.CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE",
        cu.CUDA_ERROR_OUT_OF_MEMORY => "CUDA_ERROR_OUT_OF_MEMORY",
        cu.CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED",
        cu.CUDA_ERROR_DEINITIALIZED => "CUDA_ERROR_DEINITIALIZED",
        cu.CUDA_ERROR_PROFILER_DISABLED => "CUDA_ERROR_PROFILER_DISABLED",
        cu.CUDA_ERROR_PROFILER_NOT_INITIALIZED => "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
        cu.CUDA_ERROR_PROFILER_ALREADY_STARTED => "CUDA_ERROR_PROFILER_ALREADY_STARTED",
        cu.CUDA_ERROR_PROFILER_ALREADY_STOPPED => "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
        cu.CUDA_ERROR_STUB_LIBRARY => "CUDA_ERROR_STUB_LIBRARY",
        cu.CUDA_ERROR_NO_DEVICE => "CUDA_ERROR_NO_DEVICE",
        cu.CUDA_ERROR_INVALID_DEVICE => "CUDA_ERROR_INVALID_DEVICE",
        cu.CUDA_ERROR_DEVICE_NOT_LICENSED => "CUDA_ERROR_DEVICE_NOT_LICENSED",
        cu.CUDA_ERROR_INVALID_IMAGE => "CUDA_ERROR_INVALID_IMAGE",
        cu.CUDA_ERROR_INVALID_CONTEXT => "CUDA_ERROR_INVALID_CONTEXT",
        cu.CUDA_ERROR_CONTEXT_ALREADY_CURRENT => "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
        cu.CUDA_ERROR_MAP_FAILED => "CUDA_ERROR_MAP_FAILED",
        cu.CUDA_ERROR_UNMAP_FAILED => "CUDA_ERROR_UNMAP_FAILED",
        cu.CUDA_ERROR_ARRAY_IS_MAPPED => "CUDA_ERROR_ARRAY_IS_MAPPED",
        cu.CUDA_ERROR_ALREADY_MAPPED => "CUDA_ERROR_ALREADY_MAPPED",
        cu.CUDA_ERROR_NO_BINARY_FOR_GPU => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        cu.CUDA_ERROR_ALREADY_ACQUIRED => "CUDA_ERROR_ALREADY_ACQUIRED",
        cu.CUDA_ERROR_NOT_MAPPED => "CUDA_ERROR_NOT_MAPPED",
        cu.CUDA_ERROR_NOT_MAPPED_AS_ARRAY => "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
        cu.CUDA_ERROR_NOT_MAPPED_AS_POINTER => "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
        cu.CUDA_ERROR_ECC_UNCORRECTABLE => "CUDA_ERROR_ECC_UNCORRECTABLE",
        cu.CUDA_ERROR_UNSUPPORTED_LIMIT => "CUDA_ERROR_UNSUPPORTED_LIMIT",
        cu.CUDA_ERROR_CONTEXT_ALREADY_IN_USE => "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
        cu.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
        cu.CUDA_ERROR_INVALID_PTX => "CUDA_ERROR_INVALID_PTX",
        cu.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
        cu.CUDA_ERROR_NVLINK_UNCORRECTABLE => "CUDA_ERROR_NVLINK_UNCORRECTABLE",
        cu.CUDA_ERROR_JIT_COMPILER_NOT_FOUND => "CUDA_ERROR_JIT_COMPILER_NOT_FOUND",
        cu.CUDA_ERROR_UNSUPPORTED_PTX_VERSION => "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
        cu.CUDA_ERROR_JIT_COMPILATION_DISABLED => "CUDA_ERROR_JIT_COMPILATION_DISABLED",
        cu.CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY",
        cu.CUDA_ERROR_INVALID_SOURCE => "CUDA_ERROR_INVALID_SOURCE",
        cu.CUDA_ERROR_FILE_NOT_FOUND => "CUDA_ERROR_FILE_NOT_FOUND",
        cu.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
        cu.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
        cu.CUDA_ERROR_OPERATING_SYSTEM => "CUDA_ERROR_OPERATING_SYSTEM",
        cu.CUDA_ERROR_INVALID_HANDLE => "CUDA_ERROR_INVALID_HANDLE",
        cu.CUDA_ERROR_ILLEGAL_STATE => "CUDA_ERROR_ILLEGAL_STATE",
        cu.CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND",
        cu.CUDA_ERROR_NOT_READY => "CUDA_ERROR_NOT_READY",
        cu.CUDA_ERROR_ILLEGAL_ADDRESS => "CUDA_ERROR_ILLEGAL_ADDRESS",
        cu.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        cu.CUDA_ERROR_LAUNCH_TIMEOUT => "CUDA_ERROR_LAUNCH_TIMEOUT",
        cu.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
        cu.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
        cu.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
        cu.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
        cu.CUDA_ERROR_CONTEXT_IS_DESTROYED => "CUDA_ERROR_CONTEXT_IS_DESTROYED",
        cu.CUDA_ERROR_ASSERT => "CUDA_ERROR_ASSERT",
        cu.CUDA_ERROR_TOO_MANY_PEERS => "CUDA_ERROR_TOO_MANY_PEERS",
        cu.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
        cu.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
        cu.CUDA_ERROR_HARDWARE_STACK_ERROR => "CUDA_ERROR_HARDWARE_STACK_ERROR",
        cu.CUDA_ERROR_ILLEGAL_INSTRUCTION => "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        cu.CUDA_ERROR_MISALIGNED_ADDRESS => "CUDA_ERROR_MISALIGNED_ADDRESS",
        cu.CUDA_ERROR_INVALID_ADDRESS_SPACE => "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        cu.CUDA_ERROR_INVALID_PC => "CUDA_ERROR_INVALID_PC",
        cu.CUDA_ERROR_LAUNCH_FAILED => "CUDA_ERROR_LAUNCH_FAILED",
        cu.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE",
        cu.CUDA_ERROR_NOT_PERMITTED => "CUDA_ERROR_NOT_PERMITTED",
        cu.CUDA_ERROR_NOT_SUPPORTED => "CUDA_ERROR_NOT_SUPPORTED",
        cu.CUDA_ERROR_SYSTEM_NOT_READY => "CUDA_ERROR_SYSTEM_NOT_READY",
        cu.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
        cu.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE",
        cu.CUDA_ERROR_MPS_CONNECTION_FAILED => "CUDA_ERROR_MPS_CONNECTION_FAILED",
        cu.CUDA_ERROR_MPS_RPC_FAILURE => "CUDA_ERROR_MPS_RPC_FAILURE",
        cu.CUDA_ERROR_MPS_SERVER_NOT_READY => "CUDA_ERROR_MPS_SERVER_NOT_READY",
        cu.CUDA_ERROR_MPS_MAX_CLIENTS_REACHED => "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED",
        cu.CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED => "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED",
        cu.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED",
        cu.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED",
        cu.CUDA_ERROR_STREAM_CAPTURE_MERGE => "CUDA_ERROR_STREAM_CAPTURE_MERGE",
        cu.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED",
        cu.CUDA_ERROR_STREAM_CAPTURE_UNJOINED => "CUDA_ERROR_STREAM_CAPTURE_UNJOINED",
        cu.CUDA_ERROR_STREAM_CAPTURE_ISOLATION => "CUDA_ERROR_STREAM_CAPTURE_ISOLATION",
        cu.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT",
        cu.CUDA_ERROR_CAPTURED_EVENT => "CUDA_ERROR_CAPTURED_EVENT",
        cu.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD",
        cu.CUDA_ERROR_TIMEOUT => "CUDA_ERROR_TIMEOUT",
        cu.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE",
        cu.CUDA_ERROR_EXTERNAL_DEVICE => "CUDA_ERROR_EXTERNAL_DEVICE",
        cu.CUDA_ERROR_UNKNOWN => "CUDA_ERROR_UNKNOWN",
        else => "<Unexpected cuda error please open a bug to Cudaz>",
    };
}

pub fn log_err_message(result: cu.CUresult) void {
    const err_name = error_name(result);
    var err_message: [*c]const u8 = undefined;
    const error_string_res = cu.cuGetErrorString(result, &err_message);
    if (error_string_res != cu.CUDA_SUCCESS) {
        err_message = "(no error message)";
    }
    log.err("{s}({d}): {s}", .{ err_name, result, err_message });
}
