#ifndef LIST_H
# define LIST_H

# ifdef __cplusplus
extern "C" {
# endif

# include <stdlib.h>

typedef struct list_item {
  struct list_item *pred, *next;
  void *datum;
} list_item;

typedef struct list {
  list_item *head, *tail;
  unsigned length;
  int (*compare)(const void *key, const void *with);
  void (*datum_delete)(void *);
} list;

typedef list_item * list_iterator;
typedef list_item * list_reverse_iterator;

void list_init(list *l,
	       int (*compare)(const void *key, const void *with),
	       void (*datum_delete)(void *datum));
void list_delete(list *l);
void list_insert_head(list *l, void *v);
void list_insert_tail(list *l, void *v);
void list_insert_before(list *l, list_item *pred, void *v);
void list_insert_after(list *l, list_item *next, void *v);
void list_insert_sorted(list *l, void *v);
void list_insert_item_head(list *l, list_item *i);
void list_insert_item_tail(list *l, list_item *i);
void list_insert_item_before(list *l, list_item *pred, list_item *i);
void list_insert_item_after(list *l, list_item *next, list_item *i);
void list_insert_item_sorted(list *l, list_item *i);
void list_remove(list *l, list_item *i);
void list_remove_head(list *l);
void list_remove_tail(list *l);
list_item *list_find(list *l, void *datum);
list_item *list_get_head(list *l);
list_item *list_get_tail(list *l);
unsigned list_get_length(list *l);
int list_is_empty(list *l);
int list_not_empty(list *l);
/* The second parameter of list_visit_items() returns nothing and takes */
/* a (void *) argument--a generic pointer.  It peeves me that if I pass */
/* an actual parameter of the form, say, void (*func)(int *i), the      */
/* compiler generates a warning.  The whole point of the (void *)       */
/* callback is to avoid the need for a cast, but nonetheless, I must    */
/* either cast the callback or match the prototype exactly and then     */
/* cast the argument internally to avoid the most heinious warning.     */
/* Why?!?                                                               */
void list_visit_items(list *l, void (*visitor)(void *v));

void list_item_init(list_item *li, void *datum);
void list_item_delete(list_item *li, void (*datum_delete)(void *datum));
void *list_item_get_datum(list_item *li);

void list_iterator_init(list *l, list_iterator *li);
void list_iterator_delete(list_iterator *li);
void list_iterator_next(list_iterator *li);
void list_iterator_prev(list_iterator *li);
void *list_iterator_get_datum(list_iterator *li);
int list_iterator_is_valid(list_iterator *li);
void list_reverse_iterator_init(list *l, list_iterator *li);
void list_reverse_iterator_delete(list_iterator *li);
void list_reverse_iterator_next(list_iterator *li);
void list_reverse_iterator_prev(list_iterator *li);
void *list_reverse_iterator_get_datum(list_iterator *li);
int list_reverse_iterator_is_valid(list_reverse_iterator *li);

# ifdef __cplusplus
}
# endif

#endif
