#ifndef SPUTIL
#define SPUTIL
#include "sparseval.h"

SPEval init_stats();
SPConf init_peval();
HPPtr load_hperc(char *hfile, char *readfile, char *readtok);
void config_peval(SPConf peval, char *readfile, char *readtok);
TreePtr load_tree(char *readfile, char *readtok, int *pos, TreePtr par, TreePtr lsib,SPConf peval,int gt, TreePtr intree, int ld, char TR);
AlPtr get_rawalign(FILE *fp, char *readfile, char *readtok);
AlPtr get_align(FILE *fp, char *readfile, char *readtok);
void show_stats(FILE *fp, int sent, SPEval eval, SPConf peval);
void get_strings(TreePtr gtree, TreePtr ttree, AlPtr align, char *tok, int ld);
void show_align(FILE *fp, AlPtr align, SPConf peval);
FILE *myfopen(char *file, char *mode);
FILE *getopen(char *file);
int show_htree(FILE *fp, TreePtr tree, int top);
int perc_heads(TreePtr tree, SPConf peval, int root);
int show_tree(FILE *fp, TreePtr tree, int top);
int eol(char c);
TreePtr remove_delemp(TreePtr tree, int del);
void eval_trees(FILE *ofp, TreePtr gtree, TreePtr ttree, SPEval loceval, SPEval globeval, SPConf peval, AlPtr oalign, char *tok);

#endif /* SPUTIL */
