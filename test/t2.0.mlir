module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @ID0000("%d\0A") {addr_space = 0 : i32}
  func.func @main() -> i32 {
    %c1_i64 = arith.constant 1 : i64
    %c10_i64 = arith.constant 10 : i64
    %c1_i64_0 = arith.constant 1 : i64
    scf.for %arg0 = %c1_i64 to %c10_i64 step %c1_i64_0  : i64 {
      %0 = llvm.mlir.addressof @ID0000 : !llvm.ptr
      %1 = llvm.call @printf(%0, %arg0) vararg(!llvm.func<i32 (ptr, i64, ...)>) : (!llvm.ptr, i64) -> i32
    }
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
