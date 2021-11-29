; ModuleID = 'src/lesson3.cu'
; clang++ src/lesson3.cu --cuda-gpu-arch=sm_35 -pthread -I ../cudaz/src/ --cuda-device-only -S -emit-llvm
source_filename = "src/lesson3.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_blockIdx_t = type { i8 }

@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@sdata = external dso_local addrspace(3) global [0 x float], align 4
; @buffe = internal unnamed_addr addrspace(3) global [16 x [16 x i32]] undef, align 4

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local void @global_reduce_kernel(float* %0, float* %1) #0 {
  %3 = alloca float*, align 8
  %4 = alloca float*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store float* %0, float** %3, align 8
  store float* %1, float** %4, align 8
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !9
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !10
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !11
  %11 = mul i32 %9, %10
  %12 = add i32 %8, %11
  store i32 %12, i32* %5, align 4
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !9
  store i32 %13, i32* %6, align 4
  %14 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !10
  %15 = udiv i32 %14, 2
  store i32 %15, i32* %7, align 4
  br label %16

16:                                               ; preds = %38, %2
  %17 = load i32, i32* %7, align 4
  %18 = icmp ugt i32 %17, 0
  br i1 %18, label %19, label %41

19:                                               ; preds = %16
  %20 = load i32, i32* %6, align 4
  %21 = load i32, i32* %7, align 4
  %22 = icmp ult i32 %20, %21
  br i1 %22, label %23, label %37

23:                                               ; preds = %19
  %24 = load float*, float** %4, align 8
  %25 = load i32, i32* %5, align 4
  %26 = load i32, i32* %7, align 4
  %27 = add i32 %25, %26
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds float, float* %24, i64 %28
  %30 = load float, float* %29, align 4
  %31 = load float*, float** %4, align 8
  %32 = load i32, i32* %5, align 4
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds float, float* %31, i64 %33
  %35 = load float, float* %34, align 4
  %36 = fadd contract float %35, %30
  store float %36, float* %34, align 4
  br label %37

37:                                               ; preds = %23, %19
  call void @llvm.nvvm.barrier0()
  br label %38

38:                                               ; preds = %37
  %39 = load i32, i32* %7, align 4
  %40 = lshr i32 %39, 1
  store i32 %40, i32* %7, align 4
  br label %16

41:                                               ; preds = %16
  %42 = load i32, i32* %6, align 4
  %43 = icmp eq i32 %42, 0
  br i1 %43, label %44, label %54

44:                                               ; preds = %41
  %45 = load float*, float** %4, align 8
  %46 = load i32, i32* %5, align 4
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds float, float* %45, i64 %47
  %49 = load float, float* %48, align 4
  %50 = load float*, float** %3, align 8
  %51 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !11
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds float, float* %50, i64 %52
  store float %49, float* %53, align 4
  br label %54

54:                                               ; preds = %44, %41
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local void @shmem_reduce_kernel(float* %0, float* %1) #0 {
  %3 = alloca float*, align 8
  %4 = alloca float*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store float* %0, float** %3, align 8
  store float* %1, float** %4, align 8
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !9
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !10
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !11
  %11 = mul i32 %9, %10
  %12 = add i32 %8, %11
  store i32 %12, i32* %5, align 4
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !9
  store i32 %13, i32* %6, align 4
  %14 = load float*, float** %4, align 8
  %15 = load i32, i32* %5, align 4
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds float, float* %14, i64 %16
  %18 = load float, float* %17, align 4
  %19 = load i32, i32* %6, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds [0 x float], [0 x float]* addrspacecast ([0 x float] addrspace(3)* @sdata to [0 x float]*), i64 0, i64 %20
  store float %18, float* %21, align 4
  call void @llvm.nvvm.barrier0()
  %22 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !10
  %23 = udiv i32 %22, 2
  store i32 %23, i32* %7, align 4
  br label %24

24:                                               ; preds = %44, %2
  %25 = load i32, i32* %7, align 4
  %26 = icmp ugt i32 %25, 0
  br i1 %26, label %27, label %47

27:                                               ; preds = %24
  %28 = load i32, i32* %6, align 4
  %29 = load i32, i32* %7, align 4
  %30 = icmp ult i32 %28, %29
  br i1 %30, label %31, label %43

31:                                               ; preds = %27
  %32 = load i32, i32* %6, align 4
  %33 = load i32, i32* %7, align 4
  %34 = add i32 %32, %33
  %35 = zext i32 %34 to i64
  %36 = getelementptr inbounds [0 x float], [0 x float]* addrspacecast ([0 x float] addrspace(3)* @sdata to [0 x float]*), i64 0, i64 %35
  %37 = load float, float* %36, align 4
  %38 = load i32, i32* %6, align 4
  %39 = sext i32 %38 to i64
  %40 = getelementptr inbounds [0 x float], [0 x float]* addrspacecast ([0 x float] addrspace(3)* @sdata to [0 x float]*), i64 0, i64 %39
  %41 = load float, float* %40, align 4
  %42 = fadd contract float %41, %37
  store float %42, float* %40, align 4
  br label %43

43:                                               ; preds = %31, %27
  call void @llvm.nvvm.barrier0()
  br label %44

44:                                               ; preds = %43
  %45 = load i32, i32* %7, align 4
  %46 = lshr i32 %45, 1
  store i32 %46, i32* %7, align 4
  br label %24

47:                                               ; preds = %24
  %48 = load i32, i32* %6, align 4
  %49 = icmp eq i32 %48, 0
  br i1 %49, label %50, label %56

50:                                               ; preds = %47
  %51 = load float, float* getelementptr inbounds ([0 x float], [0 x float]* addrspacecast ([0 x float] addrspace(3)* @sdata to [0 x float]*), i64 0, i64 0), align 4
  %52 = load float*, float** %3, align 8
  %53 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !11
  %54 = zext i32 %53 to i64
  %55 = getelementptr inbounds float, float* %52, i64 %54
  store float %51, float* %55, align 4
  br label %56

56:                                               ; preds = %50, %47
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local void @naive_histo(i32* %0, i32* %1, i32 %2) #0 {
  %4 = alloca i32*, align 8
  %5 = alloca i32*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store i32* %0, i32** %4, align 8
  store i32* %1, i32** %5, align 8
  store i32 %2, i32* %6, align 4
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !9
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !10
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !11
  %13 = mul i32 %11, %12
  %14 = add i32 %10, %13
  store i32 %14, i32* %7, align 4
  %15 = load i32*, i32** %5, align 8
  %16 = load i32, i32* %7, align 4
  %17 = sext i32 %16 to i64
  %18 = getelementptr inbounds i32, i32* %15, i64 %17
  %19 = load i32, i32* %18, align 4
  store i32 %19, i32* %8, align 4
  %20 = load i32, i32* %8, align 4
  %21 = load i32, i32* %6, align 4
  %22 = srem i32 %20, %21
  store i32 %22, i32* %9, align 4
  %23 = load i32*, i32** %4, align 8
  %24 = load i32, i32* %9, align 4
  %25 = sext i32 %24 to i64
  %26 = getelementptr inbounds i32, i32* %23, i64 %25
  %27 = load i32, i32* %26, align 4
  %28 = add nsw i32 %27, 1
  store i32 %28, i32* %26, align 4
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local void @simple_histo(i32* %0, i32* %1, i32 %2) #0 {
  %4 = alloca i32*, align 8
  %5 = alloca i32*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store i32* %0, i32** %4, align 8
  store i32* %1, i32** %5, align 8
  store i32 %2, i32* %6, align 4
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !9
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !10
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !11
  %13 = mul i32 %11, %12
  %14 = add i32 %10, %13
  store i32 %14, i32* %7, align 4
  %15 = load i32*, i32** %5, align 8
  %16 = load i32, i32* %7, align 4
  %17 = sext i32 %16 to i64
  %18 = getelementptr inbounds i32, i32* %15, i64 %17
  %19 = load i32, i32* %18, align 4
  store i32 %19, i32* %8, align 4
  %20 = load i32, i32* %8, align 4
  %21 = load i32, i32* %6, align 4
  %22 = srem i32 %20, %21
  store i32 %22, i32* %9, align 4
  %23 = load i32*, i32** %4, align 8
  %24 = load i32, i32* %9, align 4
  %25 = sext i32 %24 to i64
  %26 = getelementptr inbounds i32, i32* %23, i64 %25
  %27 = call i32 @_ZL9atomicAddPii(i32* %26, i32 1) #1
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define internal i32 @_ZL9atomicAddPii(i32* %0, i32 %1) #2 {
  %3 = alloca i32*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32*, align 8
  %6 = alloca i32, align 4
  store i32* %0, i32** %5, align 8
  store i32 %1, i32* %6, align 4
  %7 = load i32*, i32** %5, align 8
  %8 = load i32, i32* %6, align 4
  store i32* %7, i32** %3, align 8
  store i32 %8, i32* %4, align 4
  %9 = load i32*, i32** %3, align 8
  %10 = load i32, i32* %4, align 4
  %11 = atomicrmw add i32* %9, i32 %10 seq_cst
  ret i32 %11
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

attributes #0 = { convergent noinline norecurse nounwind optnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx70,+sm_35" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind }
attributes #2 = { convergent noinline nounwind optnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx70,+sm_35" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3, !4, !5, !6}
!llvm.ident = !{!7, !8}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (float*, float*)* @global_reduce_kernel, !"kernel", i32 1}
!4 = !{void (float*, float*)* @shmem_reduce_kernel, !"kernel", i32 1}
!5 = !{void (i32*, i32*, i32)* @naive_histo, !"kernel", i32 1}
!6 = !{void (i32*, i32*, i32)* @simple_histo, !"kernel", i32 1}
!7 = !{!"clang version 12.0.1 (Fedora 12.0.1-1.fc34)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!9 = !{i32 0, i32 1024}
!10 = !{i32 1, i32 1025}
!11 = !{i32 0, i32 2147483647}
