all:

EXTRA_CXXFLAGS += -Wno-sign-compare -I../../tools/portaudio/install/include
include ../kaldi.mk

# The PA_RingBuffer interface is internal and is not exported in the .so libray
# so we have to link against the static one
EXTRA_LDLIBS = ../../tools/portaudio/install/lib/libportaudio.a

UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
  ifneq ($(wildcard ../../tools/portaudio/install/include/pa_linux_alsa.h),)
    EXTRA_LDLIBS += -lasound
  else
    EXTRA_LDLIBS += -lrt
  endif
endif

ifeq ($(UNAME), Darwin)
    EXTRA_LDLIBS += -framework CoreAudio -framework AudioToolbox -framework AudioUnit -framework CoreServices
endif


BINFILES = online-net-client online-server-gmm-decode-faster online-gmm-decode-faster \
           online-wav-gmm-decode-faster online-audio-server-decode-faster \
           online-audio-client

OBJFILES =


TESTFILES =


ADDLIBS = ../online/kaldi-online.a ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  \
          ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
