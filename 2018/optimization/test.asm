BITS 64

SECTION .TEXT

GLOBAL _start
_start:
	mov rdi, 0
	mov rax, 60
	syscall
