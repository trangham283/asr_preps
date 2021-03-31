#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include "sparseval.h"
#include "sputil.h"

void failproc(char *msg)
{
  fprintf(stderr,"%s\n",msg);
  exit(1);
}

FILE *myfopen(char *file, char *mode)
{
  FILE *fp=fopen(file,mode);
  if (fp==NULL) {
    fprintf(stderr,"ERROR: could not open file %s in mode \"%s\"\n",file,mode);
    exit(1);
  }
  return fp;
}

int ws(char c)
{
  if (c==' ' || c=='\t') return 1;
  return 0;
}

int eol(char c)
{
  if (c==0 || c=='\n' || c==EOF) return 1;
  return 0;
}

int functagdel(char c)
{
  if (c=='-' || c=='=') return 1;
  return 0;
}

FILE *getopen(char *file)
{
  int i=0;
  while (!eol(file[i])) i++;
  file[i]=0;
  return myfopen(file,"r");
}

int pfact(int idx)
{
  int i=idx%10;
  int primes[10]={7,11,13,17,19,23,29,31,37,41};
  return primes[i];
}

int hval(char c)
{
  int sc=0;
  if (c=='-') return sc;
  sc=c;
  if (sc > 64 && sc < 91) sc+=32;
  return sc;
}

long strhash(char *hstr)
{
  long h=0;
  int i, len=strlen(hstr);
  for (i=0; i < len; i++) h+=hval(hstr[i])*pfact(i);
  return h;
}

int baghash(TreePtr tree, TreePtr child, SPConf peval)
{
  long accum;
  int eqw1=equivlabel(tree->hlabel,peval),eqw2=equivlabel(child->hlabel,peval);
  char *w1=tree->hlabel, *w2=child->hlabel, *s1=tree->label, *s2=child->label;
  if (eqw1) w1=s1;
  if (eqw2) w2=s2;
  accum=strhash(w1);
  accum=accum%THHASH;
  accum+=strhash(w2);
  accum=accum%THHASH;
  accum+=strhash(s1);
  accum=accum%THHASH;
  accum+=strhash(s2);
  eqw1=accum%THHASH;
  return eqw1;
}

TreePtr def_tree(TreePtr par, TreePtr lsib)
{
  TreePtr tree=malloc(sizeof(struct Tree));
  tree->label=tree->word=NULL;
  tree->fchild=tree->rsibs=tree->hchild=NULL;
  tree->del=tree->edit=tree->empty=0;
  tree->hopen=1;
  tree->bpos=tree->epos=tree->hbpos=-1;
  tree->parent=par;
  tree->lsibs=lsib;
  return tree;
}

void show_vheader(SPConf peval, char *com, char *av1, char *av2, char *hprc, char *afile)
{
  int i=1;
  fprintf(stderr,"  command: %s\n",com);
  fprintf(stderr,"  arg1: %s\n",av1);
  fprintf(stderr,"  arg2: %s\n",av2);
  fprintf(stderr,"  params file:");
  if (peval->params != NULL) fprintf(stderr,"%s",peval->params);
  fprintf(stderr,"\n  switches:");
  if (!peval->labeled) fprintf(stderr," -u");
  if (peval->bag) fprintf(stderr," -b");
  if (peval->cside) fprintf(stderr," -c");
  if (peval->list) fprintf(stderr," -l");
  fprintf(stderr,"\n  head percolation: ");
  if (hprc != NULL) fprintf(stderr,"%s",hprc);
  fprintf(stderr,"\n  alignment: ");
  if (afile != NULL) fprintf(stderr,"%s",afile);
  fprintf(stderr,"\n");
}

SPConf init_peval()
{
  SPConf peval=malloc(sizeof(struct SPevalConfig));
  peval->params=NULL;
  peval->delabel=peval->eqlabel=NULL;
  peval->fplabel=peval->edlabel=NULL;
  peval->eqword=peval->clclass=peval->emlabel=NULL;
  peval->labeled=1;
  peval->verbose=peval->cside=peval->bag=peval->list=peval->runinfo=0;
  peval->hds=peval->gwds=peval->twds=0;
  peval->tfile=malloc(MAXLABLEN*sizeof(char));
  peval->gfile=malloc(MAXLABLEN*sizeof(char));
  peval->hrules=NULL;
  return peval;
}

void zero_stats(SPEval stats)
{
  stats->sents=stats->cmatch=0;
  stats->matched=stats->gold=stats->test=0;
  stats->cross=stats->nocross=stats->twocross=0;
  stats->hmatched=stats->hgold=stats->htest=0;
  stats->omatched=stats->ogold=stats->otest=0;
}

SPEval init_stats()
{
  SPEval stats=malloc(sizeof(struct SPevalStats));
  zero_stats(stats);
  return stats;
}

LLPtr default_LL()
{
  LLPtr ll=malloc(sizeof(struct LBList));
  ll->label=ll->mlabel=NULL;
  ll->left=0;
  ll->next=NULL;
  return ll;
}

LLPtr default_attachLL(int type,SPConf peval)
{
  LLPtr this=default_LL();
  switch (type) {
  case 0:
    this->next=peval->delabel;
    peval->delabel=this;
    break;
  case 1:
    this->next=peval->edlabel;
    peval->edlabel=this;
    break;
  case 2:
    this->next=peval->fplabel;
    peval->fplabel=this;
    break;
  case 3:
    this->next=peval->clclass;
    peval->clclass=this;
    break;
  case 4:
    this->next=peval->emlabel;
    peval->emlabel=this;
    break;
  case -2:
    this->next=peval->eqword;
    peval->eqword=this;
    break;
  default:
    this->next=peval->eqlabel;
    peval->eqlabel=this;
  }
  return this;
}

LLPtr add_label(SPConf peval, char *rdfile, char *rdtok, int type)
{
  int i=0, j=0, k, len=strlen(rdfile);
  LLPtr this=default_attachLL(type,peval);
  if (type >= 0) {
    this->label=malloc((len+1)*sizeof(char));
    strcpy(this->label,rdfile);
    return this;
  }
  while (i<len && !ws(rdfile[i])) i++;
  rdfile[i++]=0;
  k=i;
  while (i<len) rdtok[j++]=rdfile[i++];
  rdtok[j++]=0;
  this->label=malloc(k*sizeof(char));
  strcpy(this->label,rdfile);
  this->mlabel=malloc(j*sizeof(char));
  strcpy(this->mlabel,rdtok);
  return this;
}

void config_peval(SPConf peval, char *rdfile, char *rdtok)
{
  int i;
  FILE *fp;
  LLPtr this;
  fp=myfopen(peval->params,"r");
  while (fgets(rdfile,MAXLABLEN,fp) != NULL) {
    i=0;
    while (!eol(rdfile[i])) i++;
    rdfile[i]=0;
    if (strncmp(rdfile,"DELETE_LABEL ",13)==0)
      add_label(peval,&rdfile[13],rdtok,0);
    if (strncmp(rdfile,"EMPTY_NODE ",11)==0)
      add_label(peval,&rdfile[11],rdtok,4);
    if (strncmp(rdfile,"EQ_LABEL ",9)==0)
      add_label(peval,&rdfile[9],rdtok,-1);
    if (strncmp(rdfile,"EDIT_LABEL ",11)==0)
      add_label(peval,&rdfile[11],rdtok,1);
    if (strncmp(rdfile,"FILLED_PAUSE",12)==0) {
      this=add_label(peval,&rdfile[14],rdtok,2);
      this->left=rdfile[12];
    }
    if (strncmp(rdfile,"CLOSED_CLASS ",13)==0)
      add_label(peval,&rdfile[13],rdtok,3);
    if (strncmp(rdfile,"EQ_WORDS ",9)==0)
      add_label(peval,&rdfile[9],rdtok,-2);
  }
  fclose(fp);
}

void check_del(TreePtr tree, SPConf peval)
{
  LLPtr del=peval->delabel;
  while (del != NULL) {
    if (strcmp(del->label,tree->label)==0) {
      tree->del=1;
      return;
    }
    del=del->next;
  }
}

void check_empty(TreePtr tree, SPConf peval)
{
  LLPtr del=peval->emlabel;
  while (del != NULL) {
    if (strcmp(del->label,tree->label)==0) {
      tree->empty=1;
      return;
    }
    del=del->next;
  }
}

void check_clclass(TreePtr tree, SPConf peval)
{
  LLPtr cl=peval->clclass;
  while (cl != NULL) {
    if (strcmp(cl->label,tree->label)==0) {
      tree->hopen=0;
      return;
    }
    cl=cl->next;
  }
}

void check_ed(TreePtr tree, SPConf peval)
{
  LLPtr ed=peval->edlabel;
  while (ed != NULL) {
    if (strcmp(ed->label,tree->label)==0) {
      tree->edit=1;
      return;
    }
    ed=ed->next;
  }
}

int check_eq(char *rdtok, SPConf peval)
{
  int i, len;
  LLPtr eq=peval->eqlabel;
  len=strlen(rdtok);
  for (i=1; i < len-1; i++) {
    if (!functagdel(rdtok[i])) continue;
    rdtok[i]=0;
    break;
  }
  while (eq != NULL) {
    if (strcmp(eq->label,rdtok)==0) {
      strcpy(rdtok,eq->mlabel);
      break;
    }
    eq=eq->next;
  }
  len=strlen(rdtok);
  return len+1;
}

void relabel_top(TreePtr tree, SPConf peval)
{
  char *newlabel=malloc(4*sizeof(char));
  strcpy(newlabel,"TOP");
  free(tree->label);
  tree->label=newlabel;
}

void get_label(TreePtr tree,char *rdfile,char *rdtok,int *i,char delimit,int word,SPConf peval, int top, char TR)
{
  int j=0, k;
  char *lbl;
  while (rdfile[i[0]] != delimit && !eol(rdfile[i[0]]))
    rdtok[j++]=rdfile[i[0]++];
  if (j==0) {
    rdtok[j++]=TR;
    rdtok[j++]='U';
    rdtok[j++]='~';
    rdtok[j++]='l';
  }
  rdtok[j++]=0;
  if (!word) j=check_eq(rdtok,peval);
  lbl=malloc(j*sizeof(char));
  strcpy(lbl,rdtok);
  if (word) {
    tree->word=lbl;
    return;
  }
  tree->label=lbl;
  if (top) relabel_top(tree,peval);
  check_clclass(tree,peval);
  check_del(tree,peval);
  check_ed(tree,peval);
  check_empty(tree,peval);
}

void free_thistree(TreePtr tree)
{
  free(tree->label);
  if (tree->word!=NULL) free(tree->word);
  free(tree);
}

void free_tree(TreePtr tree)
{
  TreePtr child=tree->fchild, nchild;
  while (child != NULL) {
    nchild=child->rsibs;
    free_tree(child);
    child=nchild;
  }
  free_thistree(tree);
}

void misbracket(char *rdfile, TreePtr tree, SPConf peval)
{
  show_tree(stderr,tree,1);
  fprintf(stderr,"misbracketing:\n%sNode:%s\ngold file: %s\ntest file %s\n",
	  rdfile,tree->label,peval->gfile,peval->tfile);
  exit(1);
}

TreePtr load_tree_(char *rdfile, char *rdtok, int *pos, TreePtr par, TreePtr lsib,SPConf peval,int gt, TreePtr intree, int top, char TR)
{
  int initpos=pos[0];
  TreePtr tree, lastc=NULL, child;
  if (rdfile[pos[0]] != '(') failproc("oops, misparse");
  if (rdfile[pos[0]+1] != '(') pos[0]++;
  else rdfile[pos[0]]=' ';
  if (intree==NULL) {
    tree=def_tree(par,lsib);
    get_label(tree,rdfile,rdtok,pos,' ',0,peval,top,TR);
  }
  else {
    while (!ws(rdfile[pos[0]]) && !eol(rdfile[pos[0]])) pos[0]++;
    tree=intree;
    lastc=tree->fchild;
    while (lastc->rsibs != NULL) lastc=lastc->rsibs;
  }
  while (!eol(rdfile[pos[0]]) && rdfile[pos[0]] != ')') {
    while (ws(rdfile[pos[0]])) pos[0]++;
    if (eol(rdfile[pos[0]])) break;
    if (rdfile[pos[0]]!='(') {
      get_label(tree,rdfile,rdtok,pos,')',1,peval,0,TR);
      if (gt) peval->gwds++;
      else peval->twds++;
    }
    else {
      child=load_tree_(rdfile,rdtok,pos,tree,lastc,peval,gt,NULL,0,TR);
      if (child->empty) free_tree(child);
      else {
	if (lastc==NULL) tree->fchild=child;
	else lastc->rsibs=child;
	lastc=child;
      }
    }
  }
  if (tree->fchild==NULL && tree->word==NULL) tree->empty=1;  
  if (eol(rdfile[pos[0]])) misbracket(rdfile,tree,peval);
  pos[0]++;  
  while (ws(rdfile[pos[0]])) pos[0]++;
  if (initpos==0 && !eol(rdfile[pos[0]])) misbracket(rdfile,tree,peval);
  return tree;
}

TreePtr load_tree(char *rdfile, char *rdtok, int *pos, TreePtr par, TreePtr lsib,SPConf peval,int gt, TreePtr intree, int ld, char TR)
{
  TreePtr tree=load_tree_(rdfile,rdtok,pos,par,lsib,peval,gt,intree,1,TR);
  remove_delemp(tree,ld);
  return tree;
}

TreePtr remove_delemp(TreePtr tree, int del)
{
  TreePtr child=tree->fchild, ret=tree->rsibs;
  while (child != NULL) 
    child=remove_delemp(child, del);
  if (tree->parent==NULL) return ret;
  if (!tree->del && !tree->empty) return ret;
  if (!del && tree->del) return ret;
  child=tree->fchild;
  if (child==NULL) {
    if (tree->lsibs!=NULL) tree->lsibs->rsibs=tree->rsibs;
    else tree->parent->fchild=tree->rsibs;
    if (tree->rsibs!=NULL) tree->rsibs->lsibs=tree->lsibs;
  }
  else {
    if (tree->lsibs!=NULL) tree->lsibs->rsibs=child;
    else tree->parent->fchild=child;
    child->lsibs=tree->lsibs;
    while (child != NULL) {
      child->parent=tree->parent;
      if (child->rsibs==NULL) {
	child->rsibs=tree->rsibs;
	if (tree->rsibs!=NULL) tree->rsibs->lsibs=child;
	break;
      }
      child=child->rsibs;
    }
  }
  if (tree->parent->fchild==NULL) tree->parent->empty=1;
  free_thistree(tree);
  return ret;
}

void free_clist(CListPtr clist)
{
  CListPtr nlist;
  while (clist != NULL) {
    nlist=clist;
    clist=clist->next;
    free(nlist);
  }
}

CListPtr def_clist(char *label, int beg)
{
  CListPtr this=malloc(sizeof(struct Constituents));
  this->label=label;
  this->bpos=beg;
  this->epos=beg;
  this->next=this->match=NULL;
  return this;
}

CListPtr get_constituents(TreePtr tree, SPConf peval, int beg)
{
  int nbeg=beg;
  TreePtr child=tree->fchild, nchild;
  CListPtr this=NULL, next,last;
  if (tree->bpos < 0) tree->bpos=tree->epos=beg; 
  if (tree->word!=NULL) return this;
  last=this=def_clist(tree->label,beg);
  while (child != NULL) {
    last->next=get_constituents(child, peval, nbeg);
    nbeg=child->epos;
    while (last->next != NULL) last=last->next;
    child=child->rsibs;
  }
  tree->epos=this->epos=nbeg;
  if (tree->del || this->epos==this->bpos) {
    last=this;
    this=this->next;
    free(last);
  }
  return this;
}

void upd_gstats(SPEval globeval, SPEval loceval)
{
  globeval->sents++;
  if (loceval->matched==loceval->test && loceval->test==loceval->gold)
    globeval->cmatch++;
  globeval->matched+=loceval->matched;
  globeval->gold+=loceval->gold;
  globeval->test+=loceval->test;
  globeval->cross+=loceval->cross;
  if (loceval->cross==0) globeval->nocross++;
  if (loceval->cross<3) globeval->twocross++;
  globeval->hmatched+=loceval->hmatched;
  globeval->hgold+=loceval->hgold;
  globeval->htest+=loceval->htest;
  globeval->omatched+=loceval->omatched;
  globeval->ogold+=loceval->ogold;
  globeval->otest+=loceval->otest;
}

void free_echart(int *eds, int *subs)
{
  int i, j;
  for (i=0; i < 100; i++)
    for (j=0; j < 100; j++)
      eds[100*i+j]=subs[100*i+j]=0;
  for (i=0; i < 100; i++) eds[100*i]=eds[i]=i;
}

int fill_chart(LLPtr glist, LLPtr tlist, int j, int k, int *eds, int *subs)
{
  int a=100*j+k-1, c=100*(j-1)+k, b=c-1, sub=1, fin=100*j+k;
  if (tlist->label==NULL || glist->label==NULL) {
    if (tlist->label==glist->label && tlist->left==glist->left) sub=0;
    else sub=10000;
  }
  else if (strcmp(tlist->label,glist->label)==0) sub=0;
  eds[fin]=eds[b]+sub;  
  if (sub>0) sub=1;
  sub+=subs[b];
  if (eds[a]+1 < eds[fin]) {
    eds[fin]=eds[a]+1;
    sub=subs[a];
  }
  if (eds[c]+1 < eds[fin]) {
    eds[fin]=eds[c]+1;
    sub=subs[c];
  }
  subs[fin]=sub;
  return fin;
}

int bracket_cross(CListPtr tlist, CListPtr plist)
{
  if (tlist->bpos < plist->bpos && tlist->epos > plist->bpos &&
      tlist->epos < plist->epos) return 1;
  if (tlist->bpos > plist->bpos && tlist->bpos < plist->epos &&
      tlist->epos > plist->epos) return 1;
  return 0;
}

void calc_stats(CListPtr oglist, CListPtr oplist, SPEval globeval, SPEval loceval,SPConf peval,int *eds, int *subs)
{
  int i;
  float ol;
  CListPtr tlist, glist=oglist, plist=oplist, cblist=oglist;
  while (glist != NULL) {
    loceval->gold++;
    glist=glist->next;
  }
  glist=oglist;
  while (plist != NULL) {
    loceval->test++;
    while (glist != NULL && glist->bpos < plist->bpos) glist=glist->next;
    while (glist != NULL && glist->bpos == plist->bpos &&
	   glist->epos > plist->epos) glist=glist->next;
    if (glist!=NULL && glist->bpos==plist->bpos && glist->epos==plist->epos) {
      tlist=glist;
      while (tlist!=NULL && tlist->bpos==plist->bpos && 
	     tlist->epos==plist->epos) {
	if (tlist->match==NULL && 
	    (!peval->labeled || strcmp(tlist->label,plist->label)==0)) {
	  plist->match=tlist;
	  tlist->match=plist;
	  loceval->matched++;
	  break;
	}
	tlist=tlist->next;
      }
    }
    while (cblist != NULL && cblist->epos <= plist->bpos) cblist=cblist->next;
    tlist=cblist;
    while (tlist != NULL && tlist->bpos < plist->epos) {
      if (bracket_cross(tlist,plist)) {
	loceval->cross++;
	break;
      }
      tlist=tlist->next;
    }
    plist=plist->next;
  }
  upd_gstats(globeval,loceval);
}

void stats_hdr(FILE *fp, int sent, int verbose, int bracket, int labeled)
{
  if (sent == 1) {
    if (labeled) 
      fprintf(fp,"                  Labeled                Labeled             Labeled       \n");
    else fprintf(fp,"                 Unlabeled              Unlabeled           Unlabeled     \n");
    fprintf(fp," Sent. ---------- Brackets ------  ---- Head-depen ---  -- Open-class hh --\n");
    fprintf(fp,"  ID   match   gold   test  cross  match   gold   test  match   gold   test\n");
    fprintf(fp,"%s\n",DELIM);
  }
  if (sent < 0 && verbose) fprintf(fp,"%s\n",DELIM);
}

float get_fmeas(float *R, float *P, float gold, float test)
{
  float F;
  R[0]/=gold;
  P[0]/=test;
  if (P[0]==0 && R[0]==0) F=0;
  else {
    F=2*P[0]*R[0];
    F/=P[0]+R[0];
  }
  return F;
}

void show_setlabels(FILE *fp, char *hdr, LLPtr ll)
{
  while (ll != NULL) {
    fprintf(fp,"  %s %s\n",hdr,ll->label);
    ll=ll->next;
  }
}

void show_setup(FILE *fp, SPConf peval)
{
  LLPtr ll;
  fprintf(fp,"Evaluation parameterizations:\n");
  show_setlabels(fp,"DELETE_LABEL",peval->delabel);
  show_setlabels(fp,"EMPTY_NODE",peval->emlabel);
  ll=peval->eqlabel;
  while (ll!=NULL) {
    fprintf(fp,"  EQ_LABEL %s %s\n",ll->label,ll->mlabel);
    ll=ll->next;
  }
  show_setlabels(fp,"EDIT_LABEL",peval->edlabel);
  show_setlabels(fp,"CLOSED_CLASS",peval->clclass);
  fprintf(fp,"%s\n\n",DELIM);
}

void show_stats(FILE *fp, int sent, SPEval eval, SPConf peval)
{
  float R=100*eval->matched,P=R,F,C=eval->cross,HR=100*eval->hmatched,HP=HR,HF,
    Z=100*eval->nocross,TC=100*eval->twocross,OR=100*eval->omatched,OP=OR,OF;
  stats_hdr(fp,sent,peval->verbose,1-peval->bag,peval->labeled);
  if (sent >= 0 && !peval->verbose) return;
  if (sent >= 0) fprintf(fp,"%5d ",sent);
  else fprintf(fp,"Total:");
  fprintf(fp,"%6d %6d %6d %6d %6d %6d %6d %6d %6d %6d\n",
	  eval->matched,eval->gold,eval->test,eval->cross,
	  eval->hmatched,eval->hgold,eval->htest,
	  eval->omatched,eval->ogold,eval->otest);
  if (sent < 0) {
    F=get_fmeas(&R,&P,eval->gold,eval->test);
    HF=get_fmeas(&HR,&HP,eval->hgold,eval->htest);
    OF=get_fmeas(&OR,&OP,eval->ogold,eval->otest);
    C/=eval->sents;
    Z/=eval->sents;
    TC/=eval->sents;
    fprintf(fp,"\nSummary\n");
    fprintf(fp,"--------------------------------------\n");
    fprintf(fp,"Number of sentences:               %10d\n",eval->sents);
    if (!peval->bag) {
      if (peval->labeled) fprintf(fp,"Labeled ");
      else fprintf(fp,"Unlabeled ");
      fprintf(fp,"Bracketing Recall:                   %8.2f\n",R);
      if (peval->labeled) fprintf(fp,"Labeled ");
      else fprintf(fp,"Unlabeled ");
      fprintf(fp,"Bracketing Precision:                %8.2f\n",P);
      if (peval->labeled) fprintf(fp,"Labeled ");
      else fprintf(fp,"Unlabeled ");
      fprintf(fp,"Bracketing F-measure:                %8.2f\n",F);
      fprintf(fp,"Complete match:                    %10d\n",eval->cmatch);
      fprintf(fp,"--------------------------------------\n");
      fprintf(fp,"Average Crossing Brackets:           %8.2f\n",C);
      fprintf(fp,"No Crossing Brackets:                %8.2f\n",Z);
      fprintf(fp,"Two or less Crossing:                %8.2f\n",TC);
      fprintf(fp,"--------------------------------------\n");
    }
    if (peval->labeled) fprintf(fp,"Labeled ");
    else fprintf(fp,"Unlabeled ");
    fprintf(fp,"Head-dependency Recall:               %8.2f\n",HR);
    if (peval->labeled) fprintf(fp,"Labeled ");
    else fprintf(fp,"Unlabeled ");
    fprintf(fp,"Head-dependency Precision:            %8.2f\n",HP);
    if (peval->labeled) fprintf(fp,"Labeled ");
    else fprintf(fp,"Unlabeled ");
    fprintf(fp,"Head-dependency F-measure:            %8.2f\n",HF);
    fprintf(fp,"--------------------------------------\n");
    if (peval->labeled) fprintf(fp,"Labeled ");
    else fprintf(fp,"Unlabeled ");
    fprintf(fp,"Open-class head-dependency Recall:    %8.2f\n",OR);
    if (peval->labeled) fprintf(fp,"Labeled ");
    else fprintf(fp,"Unlabeled ");
    fprintf(fp,"Open-class head-dependency Precision: %8.2f\n",OP);
    if (peval->labeled) fprintf(fp,"Labeled ");
    else fprintf(fp,"Unlabeled ");
    fprintf(fp,"Open-class head-dependency F-measure: %8.2f\n",OF);
  }
}

AlPtr def_align(int del)
{
  AlPtr align=malloc(sizeof(struct Alignment));
  align->gold=align->test=NULL;
  align->gtree=align->ttree=NULL;
  align->deletion=del;
  align->idx=align->sent=-1;
  align->subst=0;
  align->next=NULL;
  return align;
}

AlPtr align_adv(AlPtr talign, char *msg0, char *msg1, int gold)
{
  int sent=-1, idx=-1;
  if (talign != NULL) {
    sent=talign->sent;
    idx=talign->idx;
  }
  if (gold) while (talign != NULL && talign->gold == NULL) talign=talign->next;
  else while (talign != NULL && talign->deletion) talign=talign->next;
  if (talign == NULL) {
    fprintf(stderr,"%s (%d,%d) %s\n",msg0,sent,idx,msg1);
    exit(1);
  }
  return talign;
}

AlPtr split_testalign(AlPtr talign, AlPtr xalign, TreePtr tree)
{
  int slen=strlen(talign->test)-strlen(tree->word);
  AlPtr zalign=xalign;
  if (zalign == NULL || !zalign->deletion) {
    zalign=def_align(talign->deletion);
    zalign->next=talign->next;
    talign->next=zalign;
  }
  else zalign->deletion=0;
  zalign->test=malloc((slen+1)*sizeof(char));
  strcpy(zalign->test,&(talign->test[strlen(tree->word)]));
  talign->test[strlen(tree->word)]=0;
  talign->ttree=tree;
  return talign->next;
}

int zalign_test(AlPtr zalign, char *match, char *tok, char *word, int gold)
{
  char *zmatch;
  if (zalign==NULL) return 0;
  if (gold) zmatch=zalign->gold;
  else {
    if (zalign->test==NULL) zmatch=zalign->gold;
    else zmatch=zalign->test;
  }
  sprintf(tok,"%s-%s",match,zmatch);
  if (strcmp(tok,word)==0) return 1;
  sprintf(tok,"%s%s",match,zmatch);
  if (strcmp(tok,word)==0) return 1;
  if (surface_equiv(tok,word)) return 1;
  return 0;
}

AlPtr test_align(AlPtr talign,TreePtr tree,char *tok)
{
  int slen, idx, sent;
  char *match=NULL;
  AlPtr zalign;
  talign=align_adv(talign,"warning: test mis-alignment:",tree->word,0);
  zalign=talign->next;
  if (talign->test==NULL) match=talign->gold;
  else match=talign->test;
  idx=talign->idx;
  sent=talign->sent;
  if (strcmp(tree->word,match)==0) {
    talign->ttree=tree;
    talign=talign->next;
  }
  else if (talign->test != NULL &&
	   strncmp(tree->word,talign->test,strlen(tree->word))==0)
    talign=split_testalign(talign,zalign,tree);
  else {
    talign->ttree=tree;
    talign=talign->next;
    if (zalign_test(zalign,match,tok,tree->word,0)) talign=talign->next;
    else fprintf(stderr,"warning, potential test mis-alignment (%d,%d): %s %s\n",
		 sent,idx+1,tree->word,match);
  }
  return talign;
}

AlPtr split_goldalign(AlPtr talign, TreePtr tree)
{
  AlPtr zalign;
  int del=talign->deletion, slen=strlen(talign->gold)-strlen(tree->word);
  if (talign->test!=NULL) del=1;
  zalign=def_align(del);
  zalign->next=talign->next;
  talign->next=zalign;
  zalign->gold=malloc((slen+1)*sizeof(char));
  strcpy(zalign->gold,&(talign->gold[strlen(tree->word)]));
  talign->gold[strlen(tree->word)]=0;
  if (talign->test!=NULL && strcmp(talign->test,talign->gold)==0) {
    free(talign->test);
    talign->test=NULL;
  }
  talign->gtree=tree;
  return talign->next;
}

AlPtr gold_align(AlPtr talign,TreePtr tree, char *tok)
{
  int slen, del;
  AlPtr zalign;
  talign=align_adv(talign,"warning: gold mis-alignment:",tree->word,1);
  zalign=talign->next;
  if (strcmp(tree->word,talign->gold)==0) {
    talign->gtree=tree;
    talign=talign->next;
    return talign;
  }
  else if (strncmp(tree->word,talign->gold,strlen(tree->word))==0)
    talign=split_goldalign(talign,tree);
  else {
    talign->gtree=tree;
    if (zalign_test(zalign,talign->gold,tok,tree->word,1)) talign=talign->next;
    else fprintf(stderr,"warning, potential gold mis-alignment (%d,%d): %s %s\n",
		 talign->sent,talign->idx+1,tree->word,talign->gold); 
    if (talign != NULL) talign=talign->next;
 }
  return talign;
}

AlPtr treealign(AlPtr align, TreePtr tree, int gold, char *tok, int ld)
{
  AlPtr talign=align;
  TreePtr child=tree->fchild;
  if (tree->word!=NULL && (!tree->del || ld)) {
    if (gold) talign=gold_align(talign,tree,tok);
    else talign=test_align(talign,tree,tok);
  }
  while (child != NULL) {
    talign=treealign(talign,child,gold,tok,ld);
    child=child->rsibs;
  }
  return talign;
}

int phrbound(TreePtr tree, int right)
{
  TreePtr par=tree->parent, rsib=par->rsibs, lsib=par->lsibs;
  if (right) {
    while (rsib != NULL && rsib->del) rsib=rsib->rsibs;      
    if (rsib==NULL || par->parent==NULL || par->parent->parent==NULL) return 1;
  }
  else {
    while (lsib != NULL && lsib->del) lsib=lsib->lsibs;
    if (lsib==NULL || par->parent==NULL || par->parent->parent==NULL) return 1;
  }
  return 0;
}

LLPtr add_leftbr(LLPtr olast,TreePtr par, int *lb)
{
  LLPtr last=olast,this;
  TreePtr lsib=par->lsibs;
  while (lsib != NULL && lsib->del) lsib=lsib->lsibs;	    
  if (lsib != NULL || par->parent == NULL || par->parent->parent == NULL) {
    this=default_LL();
    this->left=1;
    lb[0]=0;
    if (last!=NULL) last->next=this;
    last=this;
  }
  return last;
}

LLPtr add_leafcat(LLPtr olast, char *label)
{
  LLPtr last=olast, this=default_LL();
  if (last!=NULL) last->next=this;
  last=this;
  this->label=label;
  return last;
}

LLPtr add_ritebr(LLPtr olast, TreePtr par, int *rb)
{
  LLPtr last=olast, this;
  TreePtr rsib=par->rsibs;
  while (rsib != NULL && rsib->del) rsib=rsib->rsibs;      
  if (rsib != NULL || par->parent == NULL ||
      par->parent->parent == NULL) {
    this=default_LL();
    rb[0]=0;
    last->next=this;
    last=this;
  }
  return last;
}

void fill_strings(TreePtr tree, AlPtr align, int gold)
{
  int lb=phrbound(tree,0), rb=phrbound(tree,1);
  LLPtr this, first=NULL, last=NULL;
  TreePtr child=tree->fchild, par=tree->parent;
  while (par != NULL) {
    if (par->del) {
      par=par->parent;
      continue;
    }
    if (lb) {
      last=add_leftbr(last,par,&lb);
      if (first==NULL) first=last;
    }
    last=add_leafcat(last,par->label);
    if (first==NULL) first=last;
    if (rb) last=add_ritebr(last,par,&lb);
    par=par->parent;
  }
  if (gold) align->glist=first;
  else align->tlist=first;
}

int count_align(AlPtr align)
{
  int i=0;
  while (align != NULL) {
    i++;
    align=align->next;
  }
  return i;
}

void get_strings(TreePtr gtree, TreePtr ttree, AlPtr align, char *tok, int ld)
{
  int i, idx=0, b, e;
  AlPtr talign=align;
  if (align==NULL) return;
  treealign(align,gtree,1,tok,ld);
  treealign(align,ttree,0,tok,ld);
  while (talign!=NULL) {
    b=idx++;
    e=idx;
    if (talign->gtree!=NULL) {
      talign->gtree->bpos=b;
      talign->gtree->epos=e;
      fill_strings(talign->gtree,talign,1);
    }
    if (talign->ttree!=NULL) {
      talign->ttree->bpos=b;
      talign->ttree->epos=e;
      if (talign->gtree!=NULL && talign->gtree->del)
	talign->ttree->del=1;
      fill_strings(talign->ttree,talign,0);
    }
    talign=talign->next;
  }
}

void free_align(AlPtr align)
{
  AlPtr this;
  while (align != NULL) {
    this=align;
    align=align->next;
    if (this->gold != NULL) free(this->gold);
    if (this->test != NULL) free(this->test);
    free(this);
  }
}

int check_match(THPtr this, TreePtr tree, TreePtr child, SPConf peval)
{
  if (peval->labeled) {
    if (strcmp(this->parent->label,tree->label)!=0) return 0;
    if (strcmp(this->child->label,child->label)!=0) return 0;
  }
  if (!equiv_word(this->parent->hlabel,tree->hlabel,peval)) return 0;
  if (!equiv_word(this->child->hlabel,child->hlabel,peval)) return 0;
  return 1;
}

THPtr adv_hstart(THPtr last, THPtr gh, THPtr oogh)
{
  THPtr ogh=oogh;
  if (last==NULL) ogh=gh->next;
  else last->next=gh->next;
  free(gh);
  return ogh;
}

THPtr free_hadv(THPtr th)
{
  THPtr next=th->next;
  free(th);
  return next;
}

void match_heads(THPtr gheads, THPtr theads, SPEval loceval, SPConf peval)
{
  int i=0;
  THPtr th=theads, ogh=gheads, gh=gheads, last;
  while (gh != NULL) {
    loceval->hgold++;
    if (gh->parent->hopen && gh->child->hopen) loceval->ogold++;
    gh=gh->next;
  }
  while (th != NULL) {
    gh=ogh;
    last=NULL;
    while (gh != NULL) {
      if (gh->child->hbpos < th->child->hbpos) {
	gh=ogh=adv_hstart(last,gh,ogh);
	continue;
      }
      if (gh->child->hbpos > th->child->hbpos) break;
      if (check_match(gh,th->parent,th->child,peval)) {
	loceval->hmatched++;
	if (gh->parent->hopen && gh->child->hopen &&
	    th->parent->hopen && th->child->hopen) loceval->omatched++;
	ogh=adv_hstart(last,gh,ogh);
	break;
      }
      last=gh;
      gh=gh->next;
    }
    loceval->htest++;
    if (th->parent->hopen && th->child->hopen) loceval->otest++;
    th=free_hadv(th);
  }
  while (ogh != NULL) ogh=free_hadv(ogh);
}

TreePtr first_child(TreePtr tree, int left)
{
  TreePtr child=tree->fchild;
  while (!left && child !=NULL && child->rsibs != NULL) child=child->rsibs;
  return child;
}

TreePtr adv_child(TreePtr tree, int left)
{
  if (tree==NULL) return tree;
  if (left) return tree->rsibs;
  return tree->lsibs;
}

int eq_match(TreePtr child, HEQPtr eclass)
{
  if (eclass==NULL) return 1;
  while (eclass != NULL) {
    if (strcmp(eclass->label,child->label)==0) return 1;
    eclass=eclass->next;
  }
  return 0;
}

TreePtr find_child(TreePtr tree, HXPtr hrule)
{
  TreePtr child=first_child(tree,hrule->left), dchild=NULL;
  while (child != NULL) {
    if (eq_match(child,hrule->eclass)) {
      if (child->del || child->edit || child->empty) {
        if (dchild==NULL) dchild=child;
      }
      else return child;
    }
    child=adv_child(child,hrule->left);
  }
  if (child==NULL && dchild !=NULL) return dchild;
  return child;
}

TreePtr best_child(TreePtr tree, SPConf peval)
{
  int cat;
  HXPtr hrule=NULL;
  TreePtr bchild;
  for (cat=1; cat < peval->hds; cat++)
    if (strcmp(peval->hrules[cat]->label,tree->label)==0) break;
  if (cat < peval->hds) hrule=peval->hrules[cat];
  while (hrule != NULL) {
    bchild=find_child(tree,hrule);
    if (bchild != NULL) return bchild;
    hrule=hrule->next;
  }
  if (peval->hds > 0) hrule=peval->hrules[0];
  while (hrule != NULL) {
    bchild=find_child(tree,hrule);
    if (bchild != NULL) return bchild;
    hrule=hrule->next;
  }
  fprintf(stderr,"Head rules with insufficient default conditions");
  exit(1);
}

TreePtr find_bchix(TreePtr tree, SPConf peval, int root)
{
  TreePtr bchild=NULL;
  if (root) {
    tree->hlabel=tree->label;
    tree->hbpos=0;
  }
  else {
    bchild=best_child(tree,peval);
    if (bchild->bpos < bchild->epos) tree->hlabel=bchild->hlabel;
    else tree->hlabel=NULL;
    tree->hopen=bchild->hopen;
    tree->hbpos=bchild->hbpos;
  }
  return bchild;
}

THPtr make_hpoint(TreePtr tree, TreePtr child)
{
  THPtr next=malloc(sizeof(struct TreeHead));
  next->parent=tree;
  next->child=child;
  next->next=NULL;
  return next;
}

THPtr headlist(TreePtr tree, TreePtr bchild, THPtr ohpoint)
{
  THPtr hpoint=ohpoint,last=NULL,ipoint=hpoint, next;
  TreePtr child=tree->fchild;
  if (tree->hlabel==NULL) return hpoint;
  while (child != NULL) {
    if (child != bchild && !child->del && !child->edit && 
	child->bpos < child->epos && child->hlabel != NULL) {
      next=make_hpoint(tree,child);
      while (ipoint != NULL && ipoint->child->hbpos < next->child->hbpos) {
	last=ipoint;
	ipoint=ipoint->next;
      }
      if (last==NULL) {
	next->next=hpoint;
	hpoint=next;
      }
      else {
	next->next=last->next;
	last->next=next;
      }
      ipoint=next;
    }
    child=child->rsibs;
  }
  return hpoint;
}

THPtr find_heads(TreePtr tree, SPConf peval, int root)
{
  THPtr hpoint=NULL, next, last=NULL, ipoint;
  TreePtr child=tree->fchild, bchild;
  if (child == NULL || tree->edit) {
    tree->hlabel=tree->word;
    if (child == NULL && tree->bpos<tree->epos) tree->hbpos=tree->bpos;
    return hpoint;
  }
  while (child != NULL) {
    next=find_heads(child,peval,0);
    if (next != NULL) {
      if (hpoint==NULL) last=hpoint=next;
      else last->next=next;
      while (last->next != NULL) last=last->next;
    }
    child=child->rsibs;
  }
  bchild=find_bchix(tree,peval,root);
  return headlist(tree,bchild,hpoint);
}

HXPtr make_hrule(char *label)
{
  int sz;
  HXPtr hrule=malloc(sizeof(struct HeadPref));
  if (label != NULL) {
    sz=sizeof(label)+1;
    hrule->label=malloc(sz*sizeof(char));
    strcpy(hrule->label,label);
  }
  else hrule->label=NULL;
  hrule->left=0;
  hrule->eclass=NULL;
  hrule->next=NULL;
  return hrule;
}

HEQPtr make_eclass(char *label)
{
  int sz=sizeof(label)+1;
  HEQPtr eclass=malloc(sizeof(struct EQCl));
  eclass->next=NULL;
  eclass->label=malloc(sz*sizeof(char));
  strcpy(eclass->label,label);
  return eclass;
}

void read_hequiv(char *rdfile, char *rdtok, int *i, HXPtr hrule)
{
  int j;
  HEQPtr eclass;
  while (!eol(rdfile[i[0]]) && rdfile[i[0]]!=')') {
    j=0;
    while (ws(rdfile[i[0]])) i[0]++;
    while (!ws(rdfile[i[0]]) && rdfile[i[0]]!=')' && !eol(rdfile[i[0]])) 
      rdtok[j++]=rdfile[i[0]++];
    rdtok[j++]=0;
    eclass=make_eclass(rdtok);
    eclass->next=hrule->eclass;
    hrule->eclass=eclass;
    while (ws(rdfile[i[0]])) i[0]++;
  }
  if (rdfile[i[0]]!=')') failproc("headfile misparse");
  i[0]++;
}

void load_hperx(char *hfile, SPConf peval, char *rdfile, char *rdtok)
{
  int i,j,cat;
  FILE *fp=myfopen(hfile,"r");
  HXPtr *hrules, hrule;
  while (fgets(rdfile,MAXLABLEN,fp) != NULL) peval->hds++;
  fclose(fp);
  cat=0;
  hrules=peval->hrules=malloc(peval->hds*sizeof(HXPtr));
  for (i=0; i < peval->hds; i++) hrules[i]=NULL;
  fp=myfopen(hfile,"r");
  while (fgets(rdfile,MAXLABLEN,fp) != NULL) {  
    i=j=0;
    while (ws(rdfile[i])) i++;
    while (!ws(rdfile[i]) && !eol(rdfile[i])) rdtok[j++]=rdfile[i++];
    rdtok[j++]=0;
    if (cat==0 && strcmp(rdtok,"*default*")!=0) cat++;
    if (strcmp(rdtok,"*default*")==0 && hrules[0]==NULL) cat=0;
    hrule=hrules[cat]=make_hrule(rdtok);
    while (ws(rdfile[i])) i++;
    while (!eol(rdfile[i])) {
      if (rdfile[i]!='(') failproc("misparse head file");
      i++;
      while (ws(rdfile[i])) i++;
      if (rdfile[i++]=='l') hrule->left=1;
      while (ws(rdfile[i])) i++;
      read_hequiv(rdfile,rdtok,&i,hrule);
      while (ws(rdfile[i])) i++;
      if (!eol(rdfile[i])) {
	hrule->next=make_hrule(NULL);
	hrule=hrule->next;
      }
    }
    while (cat < peval->hds && hrules[cat]!=NULL) cat++;
  }
  fclose(fp);
}

AlPtr perfect_align(TreePtr tree)
{
  int del=0, len;
  AlPtr first=NULL, align=NULL, last=NULL;
  TreePtr child=tree->fchild;
  if (tree->word != NULL) {
    align=def_align(del);
    len=strlen(tree->word);
    align->gold=malloc((len+1)*sizeof(char));
    strcpy(align->gold,tree->word);
    return align;
  }
  while (child != NULL) {
    align=perfect_align(child);
    if (align != NULL) {
      if (first==NULL) first=last=align;
      else last->next=align;
      while (last->next != NULL) last=last->next;
    }
    child=child->rsibs;
  }
  return first;
}

void flatten_tree(TreePtr tree)
{
  TreePtr child=tree->fchild, nchild, xchild, last=NULL;
  while (child != NULL) {
    nchild=child;
    child=child->rsibs;
    nchild->rsibs=NULL;
    if (nchild->fchild == NULL) xchild=nchild;
    else {
      flatten_tree(nchild);
      xchild=nchild->fchild;
      free(nchild->label);
      free(nchild);
    }    
    if (last==NULL) tree->fchild=xchild;
    else last->rsibs=xchild;
    xchild->lsibs=last;
    while (xchild != NULL) {
      last=xchild;
      xchild->parent=tree;
      xchild=xchild->rsibs;
    }
  }
}

void edit_flat(TreePtr tree)
{
  TreePtr child=tree->fchild;
  if (tree->edit) {
    flatten_tree(tree);
    return;
  }
  while (child != NULL) {
    edit_flat(child);
    child=child->rsibs;
  }
}

TreePtr move_lechildup(TreePtr tree, TreePtr child, TreePtr echild)
{
  TreePtr nchild=echild->rsibs;
  if (nchild != NULL) nchild->lsibs=NULL;
  child->fchild=nchild;
  echild->lsibs=child->lsibs;
  if (echild->lsibs != NULL) echild->lsibs->rsibs=echild;
  else tree->fchild=echild;
  echild->parent=tree;
  child->lsibs=echild;
  echild->rsibs=child;
  return nchild;
}

TreePtr move_rechildup(TreePtr tree, TreePtr child, TreePtr echild)
{
  TreePtr nchild=echild->lsibs;
  if (nchild != NULL) nchild->rsibs=NULL;
  echild->rsibs=child->rsibs;
  if (echild->rsibs != NULL) echild->rsibs->lsibs=echild;
  echild->parent=tree;
  child->rsibs=echild;
  echild->lsibs=child;
  return nchild;
}

void move_edit(TreePtr tree)
{
  TreePtr child=tree->fchild, nchild, echild;
  while (child != NULL) {
    if (!child->edit && child->fchild != NULL) {
      move_edit(child);
      nchild=child->fchild;
      while (nchild != NULL && nchild->edit)
	nchild=move_lechildup(tree,child,nchild);
      nchild=child->fchild;
      if (nchild==NULL) {
	nchild=child->rsibs;
	if (child->lsibs!=NULL) child->lsibs->rsibs=child->rsibs;
	if (child->rsibs!=NULL) child->rsibs->lsibs=child->lsibs;
	free(child->label);
	free(child);
	child=nchild;
	continue;
      }
      else {
	while (nchild != NULL && nchild->rsibs != NULL) 
	  nchild=nchild->rsibs;
	while (nchild != NULL && nchild->edit) 
	  nchild=move_rechildup(tree,child,nchild);
      }
    }
    child=child->rsibs;
  }
}

void merge_edit(TreePtr tree)
{
  TreePtr child=tree->fchild, xchild, lchild;
  while (child != NULL) {
    if (child->edit) {
      if (child->rsibs != NULL && child->rsibs->edit) {
	lchild=child->fchild;
	while (lchild->rsibs != NULL) lchild=lchild->rsibs;
	if (child->rsibs!=NULL) xchild=child->rsibs->fchild;
	xchild->lsibs=lchild;
	lchild->rsibs=xchild;
	while (xchild != NULL) {
	  xchild->parent=child;
	  xchild=xchild->rsibs;
	}
	xchild=child->rsibs;
	child->rsibs=xchild->rsibs;
	if (child->rsibs != NULL) child->rsibs->lsibs=child;
	free(xchild->label);
	free(xchild);	
	continue;
      }
    }
    else merge_edit(child);
    child=child->rsibs;
  }
}

void edit_norm(TreePtr tree)
{
  edit_flat(tree);
  move_edit(tree);
  merge_edit(tree);
}

THPtr *def_baghead()
{
  int i;
  THPtr *bag=malloc(THHASH*sizeof(THPtr));
  for (i=0; i < THHASH; i++) bag[i]=NULL;
  return bag;
}

int caphypdiff(char *w1, char *w2)
{
  int i=0, j=0, len1=strlen(w1), len2=strlen(w2);
  while (i < len1 && j < len2) {
    if (w1[i]=='-') {
      i++;
      continue;
    }
    if (w2[j]=='-') {
      j++;
      continue;
    }
    if (w1[i]!=w2[j] && w1[i]+32!=w2[j] && w1[i]!=w2[j]+32) return 0;
    i++;
    j++;
  }
  while (i < len1 && w1[i]=='-') i++;
  while (j < len2 && w2[j]=='-') j++;
  if (i < len1 || j < len2) return 0;
  return 1;
}

int surface_equiv(char *w1, char *w2)
{
  if (w1==NULL || w2==NULL) return 0;
  if (strcmp(w1,w2)==0) return 1;
  if (caphypdiff(w1,w2)) return 1;
  return 0;
}

int filledpause(char *word, SPConf peval)
{
  LLPtr fp=peval->fplabel;
  while (fp != NULL) {
    if (surface_equiv(word,fp->label)) return fp->left;
    fp=fp->next;
  }
  return 0;
}

int eqwordlist(char *w1, char *w2, SPConf peval)
{
  LLPtr fp=peval->eqword;
  while (fp != NULL) {
    if (surface_equiv(w1,fp->label) && surface_equiv(w2,fp->mlabel)) return 1;
    if (surface_equiv(w2,fp->label) && surface_equiv(w1,fp->mlabel)) return 1;
    fp=fp->next;
  }
  return 0;
}

int equiv_word(char *w1, char *w2, SPConf peval)
{
  int fp1, fp2;
  if (surface_equiv(w1,w2)) return 1;
  fp1=filledpause(w1,peval);
  fp2=filledpause(w2,peval);
  if (fp1 > 0 && fp1==fp2) return 1;
  if (eqwordlist(w1,w2,peval)) return 1;
  return 0;
}

int equivlabel(char *label, SPConf peval)
{
  LLPtr fp=peval->eqword;
  if (label == NULL) return 0;
  if (filledpause(label,peval) > 0) return 1;
  while (fp != NULL) {
    if (surface_equiv(label,fp->label) || 
	surface_equiv(label,fp->mlabel)) return 1;
    fp=fp->next;
  }
  return 0;
}

int perc_heads(TreePtr tree, SPConf peval, int root)
{
  int gh=0;
  TreePtr child=tree->fchild, bchild;
  if (child == NULL || tree->edit) {
    tree->hlabel=tree->word;
    if (child==NULL && !tree->del) gh++;
    return gh;
  }
  while (child != NULL) {
    gh+=perc_heads(child,peval,0);
    child=child->rsibs;
  }
  if (!root) {
    tree->hchild=best_child(tree,peval);
    tree->hlabel=tree->hchild->hlabel;
    tree->hopen=tree->hchild->hopen;
  }
  else {
    tree->hchild=tree;
    tree->hlabel=tree->label;
  }
  return gh;
}

THPtr add_hh(TreePtr tree, SPConf peval, THPtr inhh, SPEval loceval, int root, int gold)
{
  int hash;
  THPtr outhh=inhh, this;
  TreePtr child=tree->fchild, ctree=tree, par=tree->parent;
  if (tree->edit) return outhh;
  if (child==NULL && !tree->del) {
    while (ctree==par->hchild) {
      ctree=par;
      par=ctree->parent;
    }
    if (gold) {
      loceval->hgold++;
      if (par->hopen && ctree->hopen) loceval->ogold++;
    }
    else {
      loceval->htest++;
      if (par->hopen && ctree->hopen) loceval->otest++;
    }
    this=make_hpoint(par,ctree);
    this->next=outhh;
    return this;
  }
  while (child != NULL) {
    outhh=add_hh(child,peval,outhh,loceval,0,gold);
    child=child->rsibs;
  }
  return outhh;
}

void free_thptr(THPtr ttest)
{
  THPtr hold;
  while (ttest != NULL) {
    hold=ttest;
    ttest=ttest->next;
    free(hold);
  }
}

THPtr fr_thbuf(THPtr ttest,int *j, int *st)
{
  THPtr otest=ttest->next;
  free(ttest);
  st[0]++;
  j[0]++;
  return otest;
}

THPtr adv_thbuf(THPtr test,int i, int maxdiff, int *j, int *st, int *this, int *last, int *othis, int *olast)
{
  THPtr otest=test, ttest=test;
  while (ttest != NULL && j[0] < i-maxdiff) {
    this[st[0]]=last[st[0]];
    othis[st[0]]=olast[st[0]];
    otest=fr_thbuf(ttest,j,st);
    ttest=otest;
  }
  return otest;
}

int upd_maxpdiff(int this,int maxp,int i,int j,int *maxdiff, int range)
{
  if (i > j) maxdiff[0]=i-j;
  else maxdiff[0]=j-i;
  maxdiff[0]+=range;
  return this;
}

int check_cmp(int *this, int *last, int *othis, int *olast, int j)
{
  this[j]=this[j-1];
  othis[j]=othis[j-1];
  if (last[j]>this[j]) {
    this[j]=last[j];
    othis[j]=olast[j];
  }
  if (last[j-1]+1>this[j]) {
    this[j]=last[j-1];
    othis[j]=olast[j-1];
    return 1;
  }
  return 0;
}

void free_hhptr(THPtr otest, int *this, int *last, int *othis, int *olast)
{
  free_thptr(otest);
  free(this);
  free(last);
  free(othis);
  free(olast);
}

int upd_match(int *this, int *othis, int i, int j, int maxp, THPtr tgold, THPtr ttest, int *maxdiff, int range,SPConf peval)
{
  int ret=maxp, match;
  match=check_match(tgold,ttest->parent,ttest->child,peval);
  if (!match) return ret;
  this[j]++;
  if (tgold->parent->hopen && tgold->child->hopen &&
      ttest->parent->hopen && ttest->child->hopen) othis[j]++;
  if (this[j]>maxp) ret=upd_maxpdiff(this[j],maxp,i,j,maxdiff,range);
  return ret;
}

THPtr move_hhptr(int **this, int **last, int **othis, int **olast, THPtr tgold)
{
  THPtr next=tgold->next;
  int *ihold=last[0];
  last[0]=this[0];
  this[0]=ihold;
  ihold=olast[0];
  olast[0]=othis[0];
  othis[0]=ihold;
  free(tgold);
  return next;
}

int get_bestp(int *this, int i, int j, SPEval loceval)
{
  int bestp=this[j]+1;
  bestp+=loceval->hgold-i;
  return bestp;
}

void upd_resthis(int *this, int j, SPEval loceval)
{
  while (j < loceval->htest) {
    this[j]=this[j-1];
    j++; 
  }
}

int bpmpcomp(int *this, int i, int j, SPEval loceval, int maxp)
{
  int zc=0, bestp=get_bestp(this,i,j,loceval);
  if (bestp <= maxp) zc=1;
  return zc;
}

void hh_edits(THPtr gold, THPtr test, SPEval loceval, SPConf peval)
{
  int hta=loceval->htest+1, *this=malloc((hta)*sizeof(int)),match, zc=0, i=1,j,
    *last=malloc((hta)*sizeof(int)), *othis=malloc((hta)*sizeof(int)), bestp,
    *olast=malloc((hta)*sizeof(int)), maxp=0, st=1, range=2000, maxd=range;
  THPtr tgold=gold, ttest, otest=test;
  for (j=0; j <= loceval->htest; j++) last[j]=olast[j]=0;
  while (tgold != NULL) {
    j=st;
    this[st-1]=othis[st-1]=0;
    ttest=otest=adv_thbuf(otest,i,maxd,&j,&st,this,last,othis,olast);
    while (ttest != NULL && j < maxd+i) {
      if (check_cmp(this,last,othis,olast,j)) {
	bestp=get_bestp(this,i,j,loceval);
	if (bestp > maxp)
	    maxp=upd_match(this,othis,i,j,maxp,tgold,ttest,&maxd,range,peval);
	else if (ttest==otest) zc=1;
      }
      else if (ttest==otest) zc=bpmpcomp(this,i,j,loceval,maxp);
      if (zc) {
	zc=0;
	ttest=otest=fr_thbuf(otest,&j,&st);
	continue;
      }
      j++;
      ttest=ttest->next;
    }
    upd_resthis(this,j,loceval);
    tgold=move_hhptr(&this,&last,&othis,&olast,tgold);
    i++;
  }
  loceval->hmatched=last[loceval->htest];
  loceval->omatched=olast[loceval->htest];
  free_hhptr(otest,this,last,othis,olast);
}

void headbagger(TreePtr gtree, TreePtr ttree, SPConf peval, SPEval loceval, THPtr *bagheads)
{
  int gh;
  THPtr gold=NULL,test=NULL;
  gh=perc_heads(gtree,peval,1);
  perc_heads(ttree,peval,1);  
  gold=add_hh(gtree,peval,gold,loceval,1,1);
  test=add_hh(ttree,peval,test,loceval,1,0);
  hh_edits(gold,test,loceval,peval);
}

void free_baghead(THPtr *bagheads)
{
  int i;
  THPtr this, next;
  for (i=0; i < THHASH; i++) {
    this=bagheads[i];
    while (this != NULL) {
      next=this->next;
      free(this);
      this=next;
    }
  }
  free(bagheads);
}

void index_align(AlPtr align, int sent)
{
  int idx=1;
  while (align != NULL) {
    align->idx=idx++;
    align->sent=sent;
    align=align->next;
  }
}

void eval_trees(FILE *ofp, TreePtr gtree, TreePtr ttree, SPEval loceval, SPEval globeval, SPConf peval, AlPtr oalign, char *tok)
{
  int eds[10000], subs[10000], ld=0;
  AlPtr align=oalign,tal;
  THPtr *bagheads;
  CListPtr glist=NULL, plist=NULL;
  THPtr gheads=NULL, theads=NULL;
  edit_norm(gtree);
  edit_norm(ttree);
  zero_stats(loceval);
  if (!peval->bag && align==NULL) {
    ld=1;
    tal=align=perfect_align(gtree);
  }
  if (align!=NULL) {
    index_align(align,globeval->sents);
    get_strings(gtree,ttree,align,tok,ld);
    remove_delemp(gtree,1);
    remove_delemp(ttree,1);
    glist=get_constituents(gtree,peval,0);
    plist=get_constituents(ttree,peval,0);
    if (peval->hrules!=NULL) {
      gheads=find_heads(gtree,peval,1);
      theads=find_heads(ttree,peval,1);
      match_heads(gheads,theads,loceval,peval);
    }
    calc_stats(glist,plist,globeval,loceval,peval,&eds[0],&subs[0]);
    free_align(align);
    free_clist(glist);
    free_clist(plist);
  }
  else {
    remove_delemp(gtree,1);
    remove_delemp(ttree,1);
    if (peval->hrules!=NULL) {
      bagheads=def_baghead();
      headbagger(gtree,ttree,peval,loceval,bagheads);
      free_baghead(bagheads);
    }
    upd_gstats(globeval,loceval);
  }
  show_stats(ofp,globeval->sents,loceval,peval);
  fflush(ofp);
  free_tree(gtree);
  free_tree(ttree);
}

int show_tree(FILE *fp, TreePtr tree, int top)
{
  int prt=1;
  TreePtr child=tree->fchild;
  if (!top) fprintf(fp," ");
  fprintf(fp,"(%s",tree->label);
  if (child == NULL) {
    fprintf(fp," %s)",tree->word);
    return 1;
  }
  while (child != NULL) {
    prt=show_tree(fp,child,0);
    child=child->rsibs;
    if (top && prt && child != NULL) {
      fprintf(fp,")\n(%s",tree->label);
    }
  }
  fprintf(fp,")");
  if (top) fprintf(fp,"\n");
  return 1;
}

int show_htree(FILE *fp, TreePtr tree, int top)
{
  int prt=1;
  TreePtr child=tree->fchild;
  if (!top) fprintf(fp," ");
  fprintf(fp,"(%s",tree->label);
  if (child == NULL) {
    fprintf(fp," %s)",tree->word);
    return 1;
  }
  if (!top && tree->hlabel != NULL) fprintf(fp,"^%s",tree->hlabel);
  while (child != NULL) {
    prt=show_htree(fp,child,0);
    child=child->rsibs;
    if (top && prt && child != NULL) {
      fprintf(fp,")\n(%s",tree->label);
    }
  }
  fprintf(fp,")");
  if (top) fprintf(fp,"\n");
  return 1;
}

int get_stoken(char *rdfile, char *rdtok, int i)
{
  int j=0;
  while (ws(rdfile[i])) i++;
  while (!ws(rdfile[i]) && !eol(rdfile[i])) rdtok[j++]=rdfile[i++];
  rdtok[j]=0;
  return j+1;
}

AlPtr get_rawalign(FILE *fp, char *rdfile, char *rdtok)
{
  int i, j, start=0, ins, del;
  AlPtr first=NULL, align=NULL, last=NULL;
  while (fgets(rdfile,MAXLABLEN,fp) != NULL) {  
    if (!start) {
      if (strncmp(rdfile," ref del ins sub ",17) == 0) start=1;
      continue;
    }
    i=del=ins=0;
    if (eol(rdfile[0])) break;
    if (rdfile[3]!='1') ins=1;
    if (rdfile[7]=='1' || rdfile[7]=='0') del=1;
    align=def_align(del);
    if (!ins) {
      j=get_stoken(rdfile,rdtok,16);
      align->gold=malloc(j*sizeof(char));
      strcpy(align->gold,rdtok);
    }
    if (!del) {
      j=get_stoken(rdfile,rdtok,62);
      if (ins || strcmp(rdtok,align->gold) != 0) {
	align->test=malloc(j*sizeof(char));
	strcpy(align->test,rdtok);
      }
    }
    if (first==NULL) first=align;
    else last->next=align;
    last=align;
  }
  if (first==NULL) failproc("no alignment data in file!\n");
  return first;
}

AlPtr get_align(FILE *fp, char *rdfile, char *rdtok)
{
  int i, j, ins;
  AlPtr first=NULL, align=NULL, last=NULL;
  while (fgets(rdfile,MAXLABLEN,fp) != NULL) {  
    align=def_align(0);
    i=j=ins=0;
    while (!ws(rdfile[i]) && !eol(rdfile[i])) rdtok[j++]=rdfile[i++];
    if (j > 0) {
      rdtok[j++]=0;
      align->gold=malloc(j*sizeof(char));
      strcpy(align->gold,rdtok);
    }
    else ins=1;
    while (ws(rdfile[i])) i++;
    j=0;
    while (!ws(rdfile[i]) && !eol(rdfile[i])) rdtok[j++]=rdfile[i++];
    rdtok[j++]=0;
    if (!eol(rdfile[i])) {
      while (ws(rdfile[i])) i++;
      if (rdfile[i+1]=='0' && ins) {
	fprintf(stderr,"%s\n",rdfile);
	failproc("disagreement in alignment a");
      }
      if (rdfile[i+1]=='1' && !ins) failproc("disagreement in alignment b");
      if (rdfile[i+2]=='1') align->subst=1;
      if (align->gold==NULL || strcmp(align->gold,rdtok)!=0) {
	align->test=malloc(j*sizeof(char));
	strcpy(align->test,rdtok);
      }
    }
    else {
      if (strcmp(rdtok,"100")!=0) failproc("disagreement in alignment c");
      align->deletion=1;
    }
    if (first==NULL) first=align;
    else last->next=align;
    last=align;
  }
  if (first==NULL) failproc("no alignment data in file!\n");
  return first;
}

void show_align(FILE *fp, AlPtr align, SPConf peval)
{
  int del, ins, subst;
  while (align != NULL) {
    del=ins=subst=0;
    if (align->gtree==NULL && align->ttree==NULL) {
      align=align->next;
      continue;
    }
    if (align->gtree==NULL) {
      fprintf(fp,"\t\t");
      ins=1;
    }
    else {
      fprintf(fp,"%s\t",align->gtree->word);
      if (strlen(align->gtree->word) < 8) fprintf(fp,"\t");
    }
    if (align->ttree==NULL) {
      fprintf(fp,"\t\t");
      del=1;
    }
    else {
      fprintf(fp,"%s\t",align->ttree->word);
      if (strlen(align->ttree->word) < 8) fprintf(fp,"\t");
      if (!ins && !equiv_word(align->gtree->word,align->ttree->word,peval)) 
	subst=1;
    }
    fprintf(fp,"%d%d%d\n",del,ins,subst);
    align=align->next;
  }
}
