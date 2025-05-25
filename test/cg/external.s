	.file	""
	.data
	.globl	_camlExternal$data_begin
_camlExternal$data_begin:
	.text
	.globl	_camlExternal$code_begin
_camlExternal$code_begin:
	nop
	.align	3
	.data
	.align	3
	.data
	.align	3
	.quad	1792
	.globl	_camlExternal
	.globl	_camlExternal
_camlExternal:
	.quad	1
	.data
	.align	3
	.globl	_camlExternal$gc_roots
	.globl	_camlExternal$gc_roots
_camlExternal$gc_roots:
	.quad	_camlExternal
	.quad	0
	.text
	.align	3
	.globl	_camlExternal$entry
_camlExternal$entry:
	.cfi_startproc
L101:
L102:
	str	x30, [sp, #-8]
	.cfi_offset 30, -8
	sub	sp, sp, #16
	.cfi_adjust_cfa_offset	16
	.ifne (. - L102) - 8
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L103:
L100:
	.ifne (. - L103) - 0
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L104:
	movz	x0, #21, lsl #0
	.ifne (. - L104) - 4
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L105:
	adrp	x8, _foo_bindc@GOTPAGE
	ldr	x8, [x8, _foo_bindc@GOTPAGEOFF]
	bl	_caml_c_call
L106:
	.ifne (. - L105) - 12
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L107:
	mov	x1, x0
	.ifne (. - L107) - 4
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L108:
	adrp	x0, _camlExternal@GOTPAGE
	ldr	x0, [x0, _camlExternal@GOTPAGEOFF]
	.ifne (. - L108) - 8
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L109:
	mov	x19, sp
	.cfi_remember_state
	.cfi_def_cfa_register 19
	ldr	x16, [x28, 64]
	mov	sp, x16
	bl	_caml_initialize
	mov	sp, x19
	.cfi_restore_state
	.ifne (. - L109) - 20
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L110:
	orr	x0, xzr, #1
	.ifne (. - L110) - 4
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L111:
	.ifne (. - L111) - 0
	.error "Emit.instr_size: instruction length mismatch"
	.endif
L112:
	add	sp, sp, #16
	.cfi_adjust_cfa_offset	-16
	ldr	x30, [sp, #-8]
	ret
	.cfi_adjust_cfa_offset	16
	.ifne (. - L112) - 12
	.error "Emit.instr_size: instruction length mismatch"
	.endif
	.ifne (. - L101) - 72
	.error "Emit.instr_size: instruction length mismatch"
	.endif
	.cfi_endproc
	.data
	.align	3
	.quad	_foo_bindc
	.text
	.globl	_camlExternal$code_end
_camlExternal$code_end:
	.data
	.quad	0
	.globl	_camlExternal$data_end
_camlExternal$data_end:
	.quad	0
	.align	3
	.globl	_camlExternal$frametable
_camlExternal$frametable:
	.quad	1
	.quad	L106
	.short	17
	.short	0
	.align	2
	.long	L113 - . + 0x0
	.align	3
	.align	2
L113:
	.long	L115 - . + 0x0
	.long	0x102880
L114:
	.asciz	"test/cg/external.ml"
	.align	2
L115:
	.long	L114 - . + 0x0
	.asciz	"External.bar"
	.align	3
