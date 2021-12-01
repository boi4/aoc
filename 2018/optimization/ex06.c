#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int id;
  int x;
  int y;
  int width;
  int height;
} claim;

void fill_claim(char *line, claim *a) {
//    printf("%s\n", line);
  line++;
  a->id = atoi(line);
  while(*line++ != '@');
  a->x = atoi(line);
  while(*line++ != ',');
  a->y = atoi(line);
  while(*line++ != ':');
  a->width = atoi(line);;
  while(*line++ != 'x');
  a->height = atoi(line);;
//    printf("#%d @ %d,%d : %dx%d\n", a->id, a->x, a->y, a->width, a->height);
}


int main() {
  int cloth[1001][1001] = { 0 };  
  int id_field[1001][1001];
  int ids[1227];

  char *line = NULL;
  size_t n = 0;
  claim cl;

  FILE *fd = fopen("../inputs/input05.txt", "r");

  while(getline(&line, &n, fd) != -1) {
    // get_claim
    fill_claim(line, &cl);
    // fill all squares
    for(int i = 0; i < cl.width; i++) {
      for(int j = 0; j < cl.height; j++) {
        if(cloth[cl.x+i][cl.y+j]++ == 0) {
          id_field[cl.x+i][cl.y+j] = cl.id;
        } else {
          ids[id_field[cl.x+i][cl.y+j]] = 1;
          ids[cl.id] = 1;
        }
      }
    }
    free(line);
    n = 0;
    line = NULL;
  }
  fclose(fd);

  // look for count == 1
  for(int i = 1; i < 1227; i++) {
    if(ids[i] == 0) {
      printf("%d\n", i);
      //break;
    }
  }
  return 0;
}

