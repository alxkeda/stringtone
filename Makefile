TARGET = STRINGTONE
CPP_SOURCES = main.cpp touchscreen/ili9341.cpp touchscreen/ft6206.cpp
C_SOURCES = touchscreen/fonts.c

C_INCLUDES += -Itouchscreen
CPP_INCLUDEES += -Itouchscreen

LIBDAISY_DIR = ../../DaisySeed/DaisyExamples/libDaisy
DAISYSP_DIR  = ../../DaisySeed/DaisyExamples/DaisySP

SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile