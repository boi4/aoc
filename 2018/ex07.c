#include <stdio.h>
#include <stdlib.h>


typedef struct {
  int id;
  int sum;
  int minutes[60];
} guard;


int main() {
  int i;
  char c;
  char buf[0x1000];
  int next_starting;
  int first = 1;
  guard *current;
  guard *max;
  int max_sum = 0;
  int max_id;

  guard *guards = (guard *) calloc(sizeof(guard), 0x1000);
  FILE *fd = fopen("inputs/input07_sorted.txt", "r");


  for(;;) {

    for(i = 0; (buf[i] = (c = getc(fd))) != 'G' && c != EOF && i < sizeof(buf); i++);
    buf[--i] = '\0';

    printf("\n\n%s\n",buf);

    if(c == EOF) break;

    char *start = buf;

    // only set next_starting
    if(first) {
      while(*start++ != ':');
      int a = atoi(start-3);
      int b = atoi(start);
      printf("a:%d b:%d\n",a,b);
      if(a != 00) {
        next_starting = 0;
      } else {
        next_starting = b;
      }
      first = 0;
      continue;
    }


    // get current guard
    while(*start++ != '#');
    int num = atoi(start);
    if(num > 0x1000) {
      printf("num was %d\n", num);
      exit(-1);
    }
    printf("%d starting at %d\n", num, next_starting);

    current = &(guards[num]);
    current->id = num;
    
    while(*start++ != ':');
    int next_switch = atoi(start);
    printf("gonna switch at %d\n", next_switch);
    int state = 0;

    for(i = next_starting; i < 60;i++) {
      if(i == next_switch) {
        while(*start++ != ':' && *start != '\0');

        next_switch = atoi(start);
        if(*(start-3) == '2' || next_switch < i) {
          next_switch = -1;
        } else {
          printf("gonna switch at %d\n", next_switch);
        }
        state = 1 - state;
      }

      current->minutes[i] += state;
      current->sum+=state;
    }

    int a = atoi(start-3);
    int b = atoi(start);
    printf("a:%d b:%d\n",a,b);
    if(a != 00) {
      next_starting = 0;
    } else {
      next_starting = b;
    }
    if(current->sum > max_sum) {
      max_sum = current->sum;
      max = current;
      max_id = num;
    }

    printf("Guard stats: num: %d sum: %d\n", current->id, current->sum);
    for(int i = 0; i < 60; i++) {
      printf("%2d ", current->minutes[i]);
    }
    printf("\n");

  }
  fclose(fd);


  int max_min = 0;
  int max_val = 0;
  for(i = 0; i < 60; i++) {
    if(max->minutes[i] > max_val) {
      max_min =i;
      max_val = max->minutes[i];
    }
  }
  printf("\n\nMax-Guard stats: num: %d sum: %d\n", max->id, max->sum);
  for(int i = 0; i < 60; i++) {
    printf("%2d ", current->minutes[i]);
  }
  printf("\n");
  printf("\n\n\n%d %d %d %d\n", max_id, max_val, max_min, max_id * max_min);




  max_val = 0;
  max_id = 0;  
  max_min = 0;
  for(int j = 0; j < 60; j++) {
    for(i = 0; i < 0x1000; i++) {
      current = &guards[i];
      if(current->minutes[j] > max_val) {
        max_id = current->id;
        max_val = current->minutes[j];
        max_min = j;
      }
    } 
  }
  printf("%d %d %d %d\n", max_id, max_val, max_min, max_id*max_min);



  return 0;
}

