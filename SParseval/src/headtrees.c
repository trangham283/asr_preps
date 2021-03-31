#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include "sparseval.h"
#include "sputil.h"

#define USAGE "Usage: %s [-opts] parsefile                                 \n\
                                                                           \n\
Options:                                                                   \n\
 -p file         evaluation parameter file                                 \n\
 -h file         head percolation file                                     \n\
 -F file         output file                                               \n\
 -l              goldfile and parsefile are lists of files to evaluate     \n\
 -v              verbose                                                   \n\
 -?              info/options                                              \n"

int main(int ac, char *av[])
{ 
  int i, j, k=-1, c, err=0, cside=0, oclose=0;
  char *ifile=NULL, rdtok[MAXLABLEN], rdfile[MAXLABLEN], *hprc=NULL,
    *gfile, *pfile, *afile=NULL;
  FILE *fp=stdin,*ifp,*afp=NULL,*ofp=stdout,*lgfp=NULL,*lpfp=NULL,*lafp=NULL;
  TreePtr gtree=NULL, ttree=NULL;
  AlPtr align=NULL;
  THPtr gheads=NULL, theads=NULL;
  SPConf peval=init_peval();
  SPEval globeval=init_stats(), loceval=init_stats();
  extern char *optarg;
  extern int optind;
  
  while ((c = getopt(ac, av, "p:h:F:lv?")) != -1)
    switch (c) {
    case 'v':
      peval->verbose = 1;
      break;
    case 'p':
      peval->params=optarg;
      break;
    case 'l':
      peval->list=1;
      break;
    case 'F':
      ofp=myfopen(optarg,"w");
      oclose=1;
      break;
    case 'h':
      hprc=optarg;
      break;
    case '?':
    default:
      err++;
    }
  
  if (err || ac != optind+1) {
    fprintf(stderr, USAGE, av[0]);
    exit(1);
  }
  
  if (hprc != NULL) load_hperx(hprc,peval,&rdfile[0],&rdtok[0]);

  gfile=av[optind];
  if (peval->params != NULL) config_peval(peval,&rdfile[0],&rdtok[0]);
  if (peval->list) lgfp=myfopen(gfile,"r");
  c=0;
  while (!peval->list || fgets(rdfile,MAXLABLEN,lgfp) != NULL) {
    if (peval->list) {
      i=0;
      while (rdfile[i]!='\n') i++;
      rdfile[i]=0;
      fp=myfopen(rdfile,"r");
      strcpy(peval->gfile,rdfile);
    }
    else {
      fp=myfopen(gfile,"r");
      strcpy(peval->gfile,gfile);
    }
    
    while (fgets(rdfile,MAXLABLEN,fp) != NULL) {  
      i=peval->gwds=peval->twds=0;
      while (ws(rdfile[i])) i++;
      if (rdfile[i]=='\n') continue;
      gtree=load_tree(&rdfile[0],&rdtok[0],&i,NULL,NULL,peval,1,gtree,1,'g');
      remove_delemp(gtree,1);
      perc_heads(gtree,peval,1);
      show_htree(ofp,gtree,1);
      gtree=ttree=NULL;
    }
    fclose(fp);
    if (!peval->list) break;
  }
  if (oclose) fclose(ofp);
  if (peval->list) fclose(lgfp);
}
