
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = \
  tied-diag-gmm-init-mono \
  tied-diag-gmm-acc-stats-ali \
  tied-diag-gmm-est \
  tied-diag-gmm-align-compiled \
  tied-diag-gmm-decode-simple \
  tied-diag-gmm-decode-faster \
  tied-diag-gmm-init-model \
  smooth-stats-diag \
  tied-full-gmm-init-mono \
  tied-full-gmm-acc-stats-ali \
  tied-full-gmm-est \
  tied-full-gmm-align-compiled \
  tied-full-gmm-decode-simple \
  tied-full-gmm-decode-faster \
  tied-full-gmm-init-model \
  smooth-stats-full \
  init-tied-codebooks \
  tied-lbg \
  full-to-diag

OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feature.a ../transform/kaldi-transform.a ../tied/kaldi-tied-gmm.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../thread/kaldi-thread.a ../util/kaldi-util.a ../base/kaldi-base.a 



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) $(BINFILES)

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

