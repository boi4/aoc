/*
 * C file for day 23, task 2
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#define LEN 1000000
#define ROUNDS 10000000
#define COUNT_OF(x) \
  ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))

struct node {
  int value;
  struct node *next;
};


int init[] = { 5,2,3,7,6,4,8,1,9 };
//int init[] = {3,8,9,1,2,5,4,6,7};

void print_list(struct node *node, size_t num_elements) {
  for (size_t i = 0; i < num_elements; i++) {
    printf("%d ", node->value);
    node = node->next;
  }
  printf("\n");
}

struct node *setup_nodes() {

  struct node *nodes = calloc(sizeof(struct node), LEN);
  if (!nodes) {
    printf("Memory error!\n");
    exit(-1);
  }

  for (size_t i = 0; i < LEN; i++) {
    nodes[i].value = i + 1;
    nodes[i].next = &nodes[(i+1) % LEN];
  }


  nodes[LEN-1].next = &nodes[init[0]-1];
  for (size_t i = 0; i < COUNT_OF(init); i++) {
    if (i == COUNT_OF(init) - 1) {
      nodes[init[i]-1].next = &nodes[COUNT_OF(init) % LEN];
    } else {
      nodes[init[i]-1].next = &nodes[init[i+1]-1];
    }
  }
  return nodes;
}

bool next_three_contain(struct node *node, int value) {
  return node->next->value == value || node->next->next->value == value || node->next->next->next->value == value;
}

int play_round(struct node *nodes, int current_cup) {
  struct node *append_node, *current_node, *old_append_next, *new_current_next;
  int append_cup;

  // get cup that we will append the three to
  for (append_cup = ((LEN + current_cup - 1 - 1) % LEN) + 1;
      next_three_contain(&nodes[current_cup - 1], append_cup);
      append_cup = ((LEN + append_cup - 1 - 1) % LEN) + 1) { }

  assert(1 <= append_cup && append_cup <= 1000000);

  // append the three to the cup
  append_node = &nodes[append_cup - 1];
  current_node = &nodes[current_cup - 1];

  old_append_next = append_node->next;
  new_current_next = current_node->next->next->next->next;

  // insert piece after append node
  append_node->next = current_node->next;
  append_node->next->next->next->next = old_append_next;

  // shorten after current node
  current_node->next = new_current_next;

  return current_node->next->value;
}

int main() {
  int cur_cup = init[0];

  struct node *nodes = setup_nodes();
  print_list(&nodes[5-1], 20);
  print_list(&nodes[LEN-1], 20);

  for (int i = 0; i < ROUNDS; i++) {
  //for (int i = 0; i < 1; i++) {
    cur_cup = play_round(nodes, cur_cup);
  }
  print_list(&nodes[0], 10);

  printf("%lu\n", (unsigned long)(&nodes[0])->next->value * (&nodes[0])->next->next->value);
}
