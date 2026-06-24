TARGET = ILI9341
CPP_SOURCES = main.cpp adafruit_tft/ili9341.cpp
C_SOURCES = adafruit_tft/fonts.c

C_INCLUDES += -Iadafruit_tft
CPP_INCLUDEES += -Iadafruit_tft

LIBDAISY_DIR = ../../DaisySeed/DaisyExamples/libDaisy
DAISYSP_DIR  = ../../DaisySeed/DaisyExamples/DaisySP

SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile