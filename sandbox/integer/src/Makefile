# This is the top-level Makefile for Kaldi.
# Also see kaldi.mk which supplies options and some rules
# used by the Makefiles in the subdirectories. 

SHELL := /bin/bash

SUBDIRS = base matrix util feat tree gmm tied transform sgmm \
          thread fstext hmm lm decoder lat cudamatrix nnet nnet_cpu \
          bin fstbin gmmbin fgmmbin tiedbin sgmmbin featbin \
          nnetbin nnetbin_cpu latbin sgmm2 sgmm2bin kwsbin 

# Optional subdirectories
EXT_SUBDIRS = online onlinebin

all: test_install $(SUBDIRS) 
	echo Done

# Compile optional stuff 
ext: test_install check_portaudio $(SUBDIRS) $(EXT_SUBDIRS)

# Note: if you have put the OpenFst installation somewhere else, you can just
# delete or comment out the lines below.
OPENFST_VER = $(shell grep 'PACKAGE_VERSION' ../tools/openfst/Makefile | sed -e 's:.*= ::')
test_install:
	@[ $(OPENFST_VER) == '1.2.10' ] || { echo "You now need openfst-1.2.10.  Do: cd ../tools; svn update; ./install.sh; cd ../src; make depend; make"; exit 1; };

check_portaudio:
	@[ -d ../tools/portaudio ] || ( cd ../tools;  ./install_portaudio.sh )

clean: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x clean; done

# the part after the ":" expands to "base/test matrix/test util/test ...."
test: $(addsuffix /test, $(SUBDIRS))

# Define an implicit rule, expands to e.g.:
#  base/test: base
#     $(MAKE) -C base test 
%/test: %
	$(MAKE) -C $< test


valgrind:
	-for x in $(SUBDIRS); do $(MAKE) -C $$x valgrind || { echo "valgrind on $$x failed"; exit 1; }; done

depend: 
	-for x in $(SUBDIRS); do $(MAKE) -C $$x depend; done


.PHONY: $(SUBDIRS)
$(SUBDIRS) :
	$(MAKE) -C $@ 

.PHONY: $(EXT_SUBDIRS)
$(EXT_SUBDIRS) :
	$(MAKE) -C $@


### Dependency list ###
# this is necessary for correct parallel compilation
#1)The tools depend on all the libraries
bin fstbin gmmbin fgmmbin tiedbin sgmmbin sgmm2bin featbin nnetbin nnetbin_cpu latbin: base matrix util feat tree optimization gmm tied transform sgmm sgmm2 fstext hmm lm decoder lat cudamatrix nnet nnet_cpu
#2)The libraries have inter-dependencies
base:
matrix : base thread
util: base matrix 
thread: util 
feat: base matrix thread util gmm transform
tree: base util matrix thread
optimization: base matrix thread
gmm: base util matrix thread tree
tied: base util matrix thread gmm tree transform
transform: base util matrix thread gmm tree
sgmm: base util matrix gmm tree transform thread
sgmm2: base util matrix gmm tree transform thread
fstext: base util matrix thread tree
hmm: base tree matrix thread
lm: base util
decoder: base util matrix thread gmm sgmm hmm tree transform
lat: base util
cudamatrix: base util matrix thread	
nnet: base util matrix thread cudamatrix
nnet_cpu: base util matrix thread 
#3)Dependencies for optional parts of Kaldi
onlinebin: base matrix util feat tree optimization gmm tied transform sgmm sgmm2 fstext hmm lm decoder lat cudamatrix nnet nnet_cpu online
online: decoder
kwsbin: fstext lat base util
