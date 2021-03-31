#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include "sparseval.h"
#include "sputil.h"

#define USAGE "Usage: %s [-opts] goldfile parsefile                        \n\
                                                                           \n\
Options:                                                                   \n\
 -p file         evaluation parameter file                                 \n\
 -h file         head percolation file                                     \n\
 -a file         string alignment file                                     \n\
 -F file         output file                                               \n\
 -l              goldfile and parsefile are lists of files to evaluate     \n\
 -b              no alignment (bag of head dependencies)                   \n\
 -c              conversation side                                         \n\
 -u              unlabeled evaluation                                      \n\
 -v              verbose                                                   \n\
 -z              show info                                                 \n\
 -?              info/options                                              \n"

int main(int ac, char *av[])
{ 
  int i, j, k=-1, c, err=0, oclose=0, ld=1, foundt;
  char rdtok[MAXLABLEN], rdfile[MAXLABLEN], *hprc=NULL, *gfile, *pfile, *afile=NULL;
  FILE *fp=stdin,*ifp,*afp=NULL,*ofp=stdout,*lgfp=NULL,*lpfp=NULL,*lafp=NULL;
  TreePtr gtree, ttree;
  AlPtr align=NULL;
  THPtr gheads=NULL, theads=NULL;
  SPConf peval=init_peval();
  SPEval globeval=init_stats(), loceval=init_stats();
  extern char *optarg;
  extern int optind;
  
  while ((c = getopt(ac, av, "p:a:h:F:lcbvzu?")) != -1)
    switch (c) {
    case 'z':
      peval->runinfo=1;
      break;
    case 'v':
      peval->verbose = 1;
      break;
    case 'u':
      peval->labeled = 0;
      break;
    case 'c':
      peval->cside = 1;
      break;
    case 'b':
      peval->bag = 1;
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
    case 'a':
      afile=optarg;
      break;
    case '?':
    default:
      err++;
    }
  
  if (peval->verbose) {
    if (optind >= ac) show_vheader(peval,av[0],"NOT PROVIDED","NOT PROVIDED",hprc,afile);
    else if (optind+1 >= ac) show_vheader(peval,av[0],av[optind],"NOT PROVIDED",hprc,afile);
    else show_vheader(peval,av[0],av[optind],av[optind+1],hprc,afile);
  }
 
  if (err || ac != optind+2) {
    fprintf(stderr, USAGE, av[0]);
    exit(1);
  }

  if (!peval->bag && afile==NULL) ld=0;

  if (hprc != NULL) load_hperx(hprc,peval,&rdfile[0],&rdtok[0]);

  gfile=av[optind];
  pfile=av[optind+1];
  if (peval->params != NULL) config_peval(peval,&rdfile[0],&rdtok[0]);
  if (peval->list) {
    lgfp=myfopen(gfile,"r");
    lpfp=myfopen(pfile,"r");
    if (afile!=NULL) lafp=myfopen(afile,"r");
  }
  c=0;
  if (peval->runinfo) show_setup(ofp,peval);
  while (!peval->list || fgets(rdfile,MAXLABLEN,lgfp) != NULL) {
    if (peval->list) {
      fp=getopen(&rdfile[0]);
      strcpy(peval->gfile,rdfile);
      if (fgets(rdfile,MAXLABLEN,lpfp) == NULL) {
	fprintf(stderr,"file length mismatch: %s %s\n",gfile,pfile);
	exit(1);
      }
      ifp=getopen(&rdfile[0]);
      strcpy(peval->tfile,rdfile);
    }
    else {
      fp=myfopen(gfile,"r");
      ifp=myfopen(pfile,"r");
      strcpy(peval->gfile,gfile);
      strcpy(peval->tfile,pfile);
    }
    if (afile!=NULL) {
      peval->cside=1;
      if (peval->list) {
	if (fgets(rdfile,MAXLABLEN,lafp) == NULL) {
	  fprintf(stderr,"file length mismatch (align): %s %s\n",gfile,afile);
	  exit(1);
	}
	afp=getopen(&rdfile[0]);
      }
      else afp=myfopen(afile,"r");
      align=get_align(afp,&rdfile[0],&rdtok[0]);
      fclose(afp);      
    }
    while (fgets(rdfile,MAXLABLEN,fp) != NULL) {  
      gtree=ttree=NULL;
      i=peval->gwds=peval->twds=0;
      while (ws(rdfile[i])) i++;
      if (eol(rdfile[i])) continue;
      gtree=load_tree(&rdfile[0],&rdtok[0],&i,NULL,NULL,peval,1,gtree,ld,'g');
      if (peval->cside) 
	while (fgets(rdfile,MAXLABLEN,fp) != NULL) {
	  i=0;
	  while (ws(rdfile[i])) i++;
	  if (eol(rdfile[i])) continue;
	  gtree=load_tree(&rdfile[0],&rdtok[0],&i,NULL,NULL,peval,1,gtree,ld,'g');
	}
      
      foundt=i=0;
      while (!foundt) {
	if (fgets(rdfile,MAXLABLEN,ifp) == NULL) {
	  fprintf(stderr,"file length mismatch (files): %s %s\n",peval->gfile,peval->tfile);
	  exit(1);
	}
	while (ws(rdfile[i])) i++;
	if (!eol(rdfile[i])) foundt=1;
      }
      ttree=load_tree(&rdfile[0],&rdtok[0],&i,NULL,NULL,peval,0,ttree,ld,'t');
      if (peval->cside) 
	while (fgets(rdfile,MAXLABLEN,ifp) != NULL) {
	  i=0;
	  while (ws(rdfile[i])) i++;
	  if (eol(rdfile[i])) continue;
	  ttree=load_tree(&rdfile[0],&rdtok[0],&i,NULL,NULL,peval,0,ttree,ld,'t');
	}
      eval_trees(ofp,gtree,ttree,loceval,globeval,peval,align,&rdtok[0]);
    }
    fclose(fp);
    fclose(ifp);
    align=NULL;
    if (!peval->list) break;
  }
  show_stats(ofp,-1,globeval,peval);
  if (oclose) fclose(ofp);
  if (peval->list) {
    fclose(lgfp);
    fclose(lpfp);
    if (afile!=NULL) fclose(lafp);
  }
}
