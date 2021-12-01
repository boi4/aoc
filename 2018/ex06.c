#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int a;
  int b;
} pair;

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
  pair field[1001][1001];  
  char *line = NULL;
  size_t n = 0;
  claim cl;

  // set all counts to zero
  for(int i = 0; i < 1001; i++) {
    for(int j = 0; j < 1001; j++) {
      field[i][j].a = 0;
      field[i][j].b = -1;
    }
  }

  FILE *fd = fopen("inputs/input05.txt", "r");
  int *ids = (int *) calloc(sizeof(int), 1227);

  while(getline(&line, &n, fd) != -1) {
    // get_claim
    fill_claim(line, &cl);
    // fill all squares
    for(int i = 0; i < cl.width; i++) {
      for(int j = 0; j < cl.height; j++) {
        field[cl.x+i][cl.y+j].a++;
        if (field[cl.x+i][cl.y+j].b == -1) {
          field[cl.x+i][cl.y+j].b = cl.id;
        } else {
          ids[field[cl.x+i][cl.y+j].b] = 1;
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
  for(int i = 0; i < 1227; i++) {
    if(ids[i] == 0) {
      printf("%d\n", i);
    }
  }
  return 0;
}

