
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = align-equal align-equal-compiled acc-tree-stats \
        show-alignments compile-questions cluster-phones \
        compute-wer make-h-transducer \
        add-self-loops convert-ali compile-train-graphs \
        make-pdf-to-tid-transducer make-ilabel-transducer show-transitions \
        ali-to-phones ali-to-post weight-silence-post acc-lda est-lda \
        ali-to-pdf est-mllt build-tree decode-faster decode-faster-mapped \
        scale-vecs init-tree-special


OBJFILES = 

all:  $(BINFILES)


TESTFILES =

$(BINFILES): ../decoder/kaldi-decoder.a  ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 

# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o *.a $(TESTFILES) $(BINFILES) 

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >&/dev/null || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

