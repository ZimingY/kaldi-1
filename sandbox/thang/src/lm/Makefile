EXTRA_CXXFLAGS = -Wno-sign-compare

all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
MATHLIB = NONE

# Uncomment following line to use IRSTLM toolkit installed in ../lmtoolkit
#include ./irstlm.mk

include ../kaldi.mk

TESTFILES = lm-lib-test

OBJFILES = kaldi-lmtable.o kaldi-lm.o

TESTOUTPUTS = composed.fst output.fst output1.fst output2.fst

LIBFILE = kaldi-lm.a

all:  $(LIBFILE)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../util/kaldi-util.a ../base/kaldi-base.a


# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) $(TESTOUTPUTS) $(CMDFILES) tmp* *.tmp

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

