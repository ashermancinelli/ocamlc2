module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @ID0000("%d\0A") {addr_space = 0 : i32}
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(10 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
    %4 = llvm.icmp "slt" %3, %1 : i64
    llvm.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = llvm.mlir.addressof @ID0000 : !llvm.ptr
    %6 = llvm.call @printf(%5, %3) vararg(!llvm.func<i32 (ptr, i64, ...)>) : (!llvm.ptr, i64) -> i32
    %7 = llvm.add %3, %2 : i64
    llvm.br ^bb1(%7 : i64)
  ^bb3:  // pred: ^bb1
    %8 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %8 : i32
  }
}

