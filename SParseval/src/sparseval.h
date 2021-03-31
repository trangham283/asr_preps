#ifndef SPEVHDR
#define SPEVHDR

#define MAXLABLEN 80920
#define THHASH 2015179
#define DELIM "=================================  ===================  ==================="

struct Tree;
typedef struct Tree *TreePtr;

struct Tree
{
  char *label;
  char *word;
  char *hlabel;
  int hopen;
  int hbpos;
  int bpos;
  int epos;
  int del;
  int empty;
  int edit;
  TreePtr hchild;
  TreePtr fchild;
  TreePtr parent;
  TreePtr lsibs;
  TreePtr rsibs;
};

struct LBList;
typedef struct LBList *LLPtr;

struct Alignment;
typedef struct Alignment *AlPtr;

struct Alignment
{
  char *gold;
  char *test;
  TreePtr gtree;
  TreePtr ttree;
  LLPtr glist;
  LLPtr tlist;
  int deletion;
  int subst;
  int idx;
  int sent;
  AlPtr next;
};

struct TreeHead;
typedef struct TreeHead *THPtr;

struct TreeHead
{
  TreePtr parent;
  TreePtr child;
  THPtr next;
};

struct Constituents;
typedef struct Constituents *CListPtr;

struct Constituents
{
  char *label;
  int bpos;
  int epos;
  CListPtr match;
  CListPtr next;
};

struct LBList
{
  char *label;
  char *mlabel;
  int left;
  LLPtr next;
};

struct EQCl;
typedef struct EQCl *HEQPtr;

struct EQCl
{
  char *label;
  HEQPtr next;
};

struct HeadPref;
typedef struct HeadPref *HXPtr;

struct HeadPref
{
  char *label;
  int left;
  HEQPtr eclass;
  HXPtr next;
};

struct SPevalConfig;
typedef struct SPevalConfig *SPConf;

struct SPevalConfig
{
  char *params;
  LLPtr delabel;
  LLPtr emlabel;
  LLPtr edlabel;
  LLPtr eqlabel;
  LLPtr fplabel;
  LLPtr eqword;
  LLPtr clclass;
  char *tfile;
  char *gfile;
  int hds;
  int gwds;
  int twds;
  HXPtr *hrules;
  int labeled;
  int verbose;
  int cside;
  int bag;
  int list;
  int runinfo;
};

struct SPevalStats;
typedef struct SPevalStats *SPEval;

struct SPevalStats
{
  int sents;
  int cmatch;
  int matched;
  int gold;
  int test;
  int cross;
  int nocross;
  int twocross;
  int hmatched;
  int hgold;
  int htest;
  int omatched;
  int ogold;
  int otest;
};

struct HeadPerc;
typedef struct HeadPerc *HPPtr;

struct HeadPerc
{
  char *label;
  char *child;
  int rank;
  HPPtr next;
};

#endif /* SPEVHDR */
