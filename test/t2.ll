; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@ID0000 = internal constant [3 x i8] c"%d\0A"

declare i32 @printf(ptr, ...)

define i32 @main() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %6, %4 ], [ 1, %0 ]
  %3 = icmp slt i64 %2, 10
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = call i32 (ptr, ...) @printf(ptr @ID0000, i64 %2)
  %6 = add i64 %2, 1
  br label %1

7:                                                ; preds = %1
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
