

all:

include ../kaldi.mk

TESTFILES = nnet-component-test nnet-precondition-test

OBJFILES = nnet-component.o nnet-nnet.o nnet-update.o train-nnet.o \
     nnet-randomize.o nnet-compute.o am-nnet.o nnet-functions.o  \
     nnet-precondition.o shrink-nnet.o combine-nnet.o combine-nnet-a.o \
     mixup-nnet.o nnet-lbfgs.o nnet-update-parallel.o combine-nnet-fast.o \
     nnet-fix.o nnet-stats.o rescale-nnet.o nnet-limit-rank.o

LIBFILE = kaldi-nnet-cpu.a 

all:  $(LIBFILE)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 

# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) tmp* *.tmp

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

