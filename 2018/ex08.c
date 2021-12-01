#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>



void shiftleft(char *str, int index, int num) {
  int a = index;
  index+=num;
  while(str[index]) {
    str[a++] = str[index++];
  }
  str[a] = '\0';
}


char *rem(char *buf, int torem) {
  char *res = (char *)malloc((strlen(buf)+1)*sizeof(char));
  strcpy(res, buf);
  for(int i = 0; res[i]; i++) {
    if(tolower(res[i]) == torem + 'a') {
      shiftleft(res, i, 1);
      i--;
    }
  }
  return res;
}




char *react(char *buf) {
  int oldlen, newlen;
  char *res = (char *)malloc((strlen(buf)+1)*sizeof(char));
  strcpy(res, buf);

  newlen = strlen(res);
  do {
    oldlen = newlen;
    for(int i = 0; res[i+1]; i++) {
      if(res[i] - 'a' + 'A' == res[i+1] || res[i] - 'A' + 'a' == res[i+1]) {
        shiftleft(res, i, 2);
      }
    }
    newlen = strlen(res);
  } while(newlen != oldlen);

  return res;
}


int main() {
  char buf[0x10000] = { 0 };
  FILE *fd = fopen("inputs/input09.txt","r");
  int n = read(fileno(fd), buf, sizeof(buf) - 1);  
  buf[n-1] = '\0';
  printf("%lu\n", strlen(buf));

  //strcpy(buf,"dabAcCaCBAcCcaDA");

  size_t min = 1000000;
  for(int i = 0; i < 26; i++) {
    char *removed = rem(buf, i);
    char *test = react(removed);
    if(strlen(test) < min)
      min = strlen(test);
    printf("%c: %lu\n", i + 'a', strlen(test));
    //printf("Removed: %s\n", removed);
    free(test);
    free(removed);
  } 

  printf("%lu\n", min);

  return 0;
}

