#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


typedef struct {
  int x;
  int y;
} point;


point *parseCoords(char *buf) {
  int len = strlen(buf);
  int count = 0;
  point *res = (point *) calloc(sizeof(point), len);
  for(int i = 0; buf[i]; ) {
    point *p = &res[count++];
    p->x = atoi(&buf[i]);
    while(buf[i++] != ',');
    p->y = atoi(&buf[i]);
    while(buf[i++] != '\n' && buf[i]);
  }
  return res;
}


int dist(point a, point b) {
  int dx = (a.x - b.x);
  if(dx < 0) dx = -1 * dx;
  int dy = (a.y - b.y);
  if(dy < 0) dy = -1 * dy;
  return dx + dy;
}

int computeClosest(point p, point *coords) {
  int closestdist = 100000;
  int closest;
  int tie = 0;
  for(int i = 0; coords[i].x; i++) {
    int d = dist(p, coords[i]);
    //printf("Testing: (%d,%d) - (%d,%d) = %d\n", p.x, p.y, coords[i].x, coords[i].y ,d);
    if(d == closestdist) {
      tie = 1;
    }
    if(d < closestdist) {
      closestdist = d;
      closest = i;
      tie = 0;
    }
  }
  if (tie) {
    printf("tie\n");
    return -1;
  }
  return closest;
}


int parta() {
  char buf[0x10000];
  FILE *fd = fopen("inputs/input06.txt","r");
  if (!fd) {
    perror("yo");
    exit(-1);
  }
  int n = read(fileno(fd),buf,sizeof(buf)-1);
  buf[n] = '\0';
  point *coords = parseCoords(buf);

  for(int i = 0; coords[i].x; i++) {
    printf("%d %d\n", coords[i].x, coords[i].y);
  }

  int xmin = 10000;
  int ymin = 10000;
  int xmax = -1;
  int ymax = -1;

  int i;
  for(i = 0; coords[i].x; i++) {
    if(coords[i].x < xmin) {
      xmin = coords[i].x;
    }
    if(coords[i].x > xmax) {
      xmax = coords[i].x;
    }
    if(coords[i].y < ymin) {
      ymin = coords[i].y;
    }
    if(coords[i].y > ymax) {
      ymax = coords[i].y;
    }
  }


  point tmp;
  // mark all infinite ones
  int *infinite = (int *) calloc(sizeof(int), i);
  int top = 0;
  // check all sides 
  tmp.x = xmin;
  for(int l = 0; l < 2; l++) {
    for(int y = ymin; y <= ymax; y++) {
      tmp.y = y;
      int closest = computeClosest(tmp, coords);
      int j;
      for(j = 0; j < top; j++) {
        if(infinite[j] == closest)
          break;
      }
      if(j == top) {
        infinite[top++] = closest;
      }
    }
    tmp.x = xmax;
  }
  tmp.y = ymin;
  for(int l = 0; l < 2; l++) {
    for(int x = xmin; x <= xmax; x++) {
      tmp.x = x;
      int closest = computeClosest(tmp, coords);
      int j;
      for(j = 0; j < top; j++) {
        if(infinite[j] == closest)
          break;
      }
      if(j == top) {
        infinite[top++] = closest;
      }
    }
    tmp.y = ymax;
  }

  printf("infinite: ");
  for(int l = 0; l < top; l++) {
    printf("%d ", infinite[l]); 
  }
  printf("\n");

  //printf("xmin: %d, xmax: %d, ymin: %d, ymax: %d\n", xmin, xmax, ymin, ymax);
  int *count = (int *)malloc(i * sizeof(int));
  for(int j = xmin; j <= xmax; j++) {
    for(int k = ymin; k <= ymax; k++) {
      tmp.x = j; 
      tmp.y = k;
      int c = computeClosest(tmp, coords);
      if (c != -1)
        count[c]++;
    }
  }

  int max = -1;
  for(int j = 0; j < i; j++) {
    if(count[j] > max) {
      printf("%d higher than current max\n", j);
      int l;
      for(l = 0; l < top; l++) {
        if(infinite[l] == j)
          break;
      }
      if (l == top) {
        max = count[j];
      } else {
        printf("Sadly, infinite\n");
      }
    }
   printf("%3d: %d\n", j, count[j]);
   //printf("%d\n", count[j]);
  }
  printf("highest count was: %d\n", max);

  return 0;
}


int partb() {
  char buf[0x10000];
  FILE *fd = fopen("inputs/input06.txt","r");
  if (!fd) {
    perror("yo");
    exit(-1);
  }
  int n = read(fileno(fd),buf,sizeof(buf)-1);
  buf[n] = '\0';
  point *coords = parseCoords(buf);

  for(int i = 0; coords[i].x; i++) {
    printf("%d %d\n", coords[i].x, coords[i].y);
  }

  int xmin = 10000;
  int ymin = 10000;
  int xmax = -1;
  int ymax = -1;

  int i;
  for(i = 0; coords[i].x; i++) {
    if(coords[i].x < xmin) {
      xmin = coords[i].x;
    }
    if(coords[i].x > xmax) {
      xmax = coords[i].x;
    }
    if(coords[i].y < ymin) {
      ymin = coords[i].y;
    }
    if(coords[i].y > ymax) {
      ymax = coords[i].y;
    }
  }
  int numcoords = i;

  int count = 0;
  point tmp;
  for(int j = xmin; j <= xmax; j++) {
    for(int k = ymin; k <= ymax; k++) {
      int s = 0;
      tmp.x = j;
      tmp.y = k;
      for(int i = 0; i < numcoords; i++) {
        s += dist(tmp, coords[i]);
      }
      if (s < 10000) count++;
    }
  }
  printf("%d\n", count);

  return 0;
}

int main() {
  partb();
}

