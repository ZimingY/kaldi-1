
all:

include ../kaldi.mk

# note, build-tree-utils-test also tests build-tree-questions.cc

TESTFILES = event-map-test context-dep-test build-tree-utils-test cluster-utils-test build-tree-test

OBJFILES = event-map.o context-dep.o clusterable-classes.o cluster-utils.o build-tree-utils.o build-tree.o build-tree-questions.o 

LIBFILE = kaldi-tree.a

all:  $(LIBFILE) $(TESTFILES)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../base/kaldi-base.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a


# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o *.a $(TESTFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >&/dev/null || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

