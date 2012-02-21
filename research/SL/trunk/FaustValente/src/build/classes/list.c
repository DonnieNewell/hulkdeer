#include <stdio.h>
#include <string.h>

#include "list.h"
#include "macros.h"

void list_init(list *l,
	       int (*compare)(const void *key, const void *with),
	       void (*datum_delete)(void *))
{
  l->head = l->tail = NULL;
  l->length = 0;
  l->compare = compare;
  l->datum_delete = datum_delete;
}

void list_delete(list *l)
{
  list_item *li, *delete;

  for (li = l->head; li;) {
    delete = li;
    li = li->next;
    list_item_delete(delete, l->datum_delete);
  }
  l->head = l->tail = NULL;
  l->length = 0;
}

void list_insert_item_head(list *l, list_item *i)
{
  if (l->head) {
    i->next = l->head;
    l->head->pred = i;
    l->head = i;
    l->head->pred = NULL;
  } else {
    l->head = l->tail = i;
    i->next = i->pred = NULL;
  }
  l->length++;
}

void list_insert_item_tail(list *l, list_item *i)
{
  if (l->head) {
    l->tail->next = i;
    i->pred = l->tail;
    i->next = NULL;
    l->tail = i;
  } else {
    l->head = l->tail = i;
    i->next = i->pred = NULL;
  }
  l->length++;
}

void list_insert_item_before(list *l, list_item *pred, list_item *i)
{
  /* Assume pred is actually in the list! */
  /* If it's not, we may lose the list.   */
  if (l->head == pred) {
    i->next = pred;
    i->pred = NULL;
    l->head = i;
    pred->pred = i;
  } else {
    i->next = pred;
    i->pred = pred->pred;
    pred->pred->next = i;
    pred->pred = i;
  }
  l->length++;
}

void list_insert_item_after(list *l, list_item *next, list_item *i)
{
  /* Assume pred is actually in the list! */
  /* If it's not, we may lose the list.   */
  if (l->tail == next) {
    i->pred = next;
    i->next = NULL;
    l->tail = i;
    next->next = i;
  } else {
    i->pred = next;
    i->next = next->next;
    next->next->pred = i;
    next->next = i;
  }
  l->length++;
}

void list_insert_item_sorted(list *l, list_item *i)
{
  list_item *itr;

  if (l->head) {
    for (itr = l->head; itr && l->compare(list_item_get_datum(i),
					  list_item_get_datum(itr)) < 0;
	 itr = itr->next)
      ;
    if (itr) {
      i->next = itr;
      i->pred = itr->pred;
      itr->pred = i;
      i->pred->next = i;
    } else {
      l->tail->next = i;
      i->pred = l->tail;
      i->next = NULL;
      l->tail = i;
    }
  } else {
    l->head = l->tail = i;
    i->pred = i->next = NULL;
  }
  l->length++;
}

void list_insert_head(list *l, void *v)
{
  list_item *i;

  i = malloc(sizeof (list_item));
  list_item_init(i, v);

  if (l->head) {
    i->next = l->head;
    l->head->pred = i;
    l->head = i;
    l->head->pred = NULL;
  } else {
    l->head = l->tail = i;
    i->next = i->pred = NULL;
  }
  l->length++;
}

void list_insert_tail(list *l, void *v)
{
  list_item *i;

  i = malloc(sizeof (list_item));
  list_item_init(i, v);
  if (l->head) {
    l->tail->next = i;
    i->pred = l->tail;
    i->next = NULL;
    l->tail = i;
  } else {
    l->head = l->tail = i;
    i->next = i->pred = NULL;
  }
  l->length++;
}

void list_insert_before(list *l, list_item *pred, void *v)
{
  list_item *i;

  i = malloc(sizeof (list_item));
  list_item_init(i, v);

  /* Assume pred is actually in the list! */
  /* If it's not, we may lose the list.   */
  if (l->head == pred) {
    i->next = pred;
    i->pred = NULL;
    l->head = i;
    pred->pred = i;
  } else {
    i->next = pred;
    i->pred = pred->pred;
    pred->pred->next = i;
    pred->pred = i;
  }
  l->length++;
}

void list_insert_after(list *l, list_item *next, void *v)
{
  list_item *i;

  i = malloc(sizeof (list_item));
  list_item_init(i, v);

  /* Assume pred is actually in the list! */
  /* If it's not, we may lose the list.   */
  if (l->tail == next) {
    i->pred = next;
    i->next = NULL;
    l->tail = i;
    next->next = i;
  } else {
    i->pred = next;
    i->next = next->next;
    next->next->pred = i;
    next->next = i;
  }
  l->length++;
}

void list_insert_sorted(list *l, void *v)
{
  list_item *itr;
  list_item *i;

  i = malloc(sizeof (list_item));
  list_item_init(i, v);


  if (l->head) {
    for (itr = l->head; itr && l->compare(list_item_get_datum(i),
					  list_item_get_datum(itr)) < 0;
	 itr = itr->next)
      ;
    if (itr) {
      i->next = itr;
      i->pred = itr->pred;
      itr->pred = i;
      i->pred->next = i;
    } else {
      l->tail->next = i;
      i->pred = l->tail;
      i->next = NULL;
      l->tail = i;
    }
  } else {
    l->head = l->tail = i;
    i->pred = i->next = NULL;
  }
  l->length++;
}

void list_remove(list *l, list_item *i)
{
  if (i == l->head) {
    l->head = l->head->next;
    if (l->head)
      l->head->pred = NULL;
    else
      l->tail = NULL;
  } else if (i == l->tail) {
    l->tail = l->tail->pred;
    l->tail->next = NULL;
  } else {
    i->pred->next = i->next;
    i->next->pred = i->pred;
  }
  l->length--;
  list_item_delete(i, l->datum_delete);
}

void list_remove_head(list *l)
{
  list_remove(l, list_get_head(l));
}

void list_remove_tail(list *l)
{
  list_remove(l, list_get_tail(l));
}

list_item *list_find(list *l, void *datum)
{
  list_item *li;

  for (li = l->head; li && l->compare(datum, list_item_get_datum(li));
       li = li->next)
    ;

  return li;
}

list_item *list_get_head(list *l)
{
  return l->head;
}

list_item *list_get_tail(list *l)
{
  return l->tail;
}

unsigned list_get_length(list *l)
{
  return l->length;
}

int list_is_empty(list *l)
{
  return (l->length == 0);
}

int list_not_empty(list *l)
{
  return (l->length != 0);
}

void list_visit_items(list *l, void (*visitor)(void *v))
{
  list_item *li;

  for (li = l->head; li; li = li->next)
    visitor(list_item_get_datum(li));
}

void list_item_init(list_item *li, void *datum)
{
  li->pred = li->next = NULL;
  li->datum = datum;
}

void list_item_delete(list_item *li, void (*datum_delete)(void *datum))
{
  if (datum_delete) {
    datum_delete(li->datum);
  }
  free(li);
}

void *list_item_get_datum(list_item *li)
{
  return li->datum;
}

void list_iterator_init(list *l, list_iterator *li)
{
  *li = l ? l->head : NULL;
}

void list_iterator_delete(list_iterator *li)
{
  *li = NULL;
}

void list_iterator_next(list_iterator *li)
{
  if (*li)
    *li = (*li)->next;
}

void list_iterator_prev(list_iterator *li)
{
  if (*li)
    *li = (*li)->pred;
}

void *list_iterator_get_datum(list_iterator *li)
{
  return *li ? (*li)->datum : NULL;
}

int list_iterator_is_valid(list_iterator *li)
{
  return (*li != NULL);
}

void list_reverse_iterator_init(list *l, list_reverse_iterator *li)
{
  *li = l ? l->tail : NULL;
}

void list_reverse_iterator_delete(list_reverse_iterator *li)
{
  *li = NULL;
}

void list_reverse_iterator_next(list_reverse_iterator *li)
{
  if (*li)
    *li = (*li)->pred;
}

void list_reverse_iterator_prev(list_reverse_iterator *li)
{
  if (*li)
    *li = (*li)->next;
}

void *list_reverse_iterator_get_datum(list_reverse_iterator *li)
{
  return *li ? (*li)->datum : NULL;
}

int list_reverse_iterator_is_valid(list_reverse_iterator *li)
{
  return (li != NULL);
}

