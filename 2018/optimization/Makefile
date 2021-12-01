all: asm c

asm:
	nasm -f elf64 -l test.lst test.asm
	ld test.o -o test

c:
	gcc ex06.c -O3 --static -o ex06
