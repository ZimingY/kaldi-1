
all:

include ../kaldi.mk

TESTFILES = const-integer-set-test stl-utils-test text-utils-test \
    edit-distance-test hash-list-test timer-test kaldi-io-test parse-options-test \
    kaldi-table-test

OBJFILES = text-utils.o kaldi-io.o \
          kaldi-table.o parse-options.o simple-io-funcs.o

LIBFILE = kaldi-util.a

all:  $(LIBFILE) $(TESTFILES)


$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../base/kaldi-base.a  ../matrix/kaldi-matrix.a


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

