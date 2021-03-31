#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include "sparseval.h"
#include "sputil.h"

#define USAGE "Usage: %s [-opts] goldfile parsefile alignfile              \n\
                                                                           \n\
Options:                                                                   \n\
 -p file         evaluation parameter file                                 \n\
 -F file         output file                                               \n\
 -v              verbose                                                   \n\
 -?              info/options                                              \n"

int main(int ac, char *av[])
{ 
  int i, j, k=-1, c, err=0, oclose=0;
  char *ifile=NULL, rdtok[MAXLABLEN], rdfile[MAXLABLEN];
  FILE *fp, *ofp=stdout;
  TreePtr gtree=NULL, ttree=NULL;
  AlPtr align=NULL, zalign;
  SPConf peval=init_peval();
  extern char *optarg;
  extern int optind;
  
  while ((c = getopt(ac, av, "p:a:F:v?")) != -1)
    switch (c) {
    case 'v':
      peval->verbose = 1;
      break;
    case 'p':
      peval->params=optarg;
      break;
    case 'F':
      ofp=myfopen(optarg,"w");
      oclose=1;
      break;
    case '?':
    default:
      err++;
    }
  
  if (err || ac != optind+3) {
    fprintf(stderr, USAGE, av[0]);
    exit(1);
  }
  
  if (peval->params != NULL) config_peval(peval,&rdfile[0],&rdtok[0]);
  strcpy(peval->gfile,av[optind]);
  strcpy(peval->tfile,av[optind+1]);
  fp=myfopen(av[optind],"r");
  while (fgets(rdfile,MAXLABLEN,fp) != NULL) {
    i=0;
    while (ws(rdfile[i])) i++;
    if (rdfile[i]=='\n') continue;
    gtree=load_tree(&rdfile[0],&rdtok[0],&i,NULL,NULL,peval,1,gtree,0,'g');
  }
  fclose(fp);
  i=0;
  fp=myfopen(av[optind+1],"r");
  while (fgets(rdfile,MAXLABLEN,fp) != NULL) {
    i=0;
    while (ws(rdfile[i])) i++;
    if (rdfile[i]=='\n') continue;
    ttree=load_tree(&rdfile[0],&rdtok[0],&i,NULL,NULL,peval,0,ttree,0,'t');
  }
  fclose(fp);
  fp=myfopen(av[optind+2],"r");
  align=get_rawalign(fp,&rdfile[0],&rdtok[0]);
  fclose(fp);
  get_strings(gtree,ttree,align, &rdtok[0],0);
  show_align(ofp,align,peval);
  if (oclose) fclose(ofp);
}
