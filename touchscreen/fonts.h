#ifndef __FONTS_H__
#define __FONTS_H__

#include <stdint.h>

typedef struct {
    const uint8_t width;
    uint8_t height;
    const uint16_t *data;
} Adafruit_TFT_FontDef;


extern Adafruit_TFT_FontDef Adafruit_TFT_Font_7x10;
extern Adafruit_TFT_FontDef Adafruit_TFT_Font_11x18;
extern Adafruit_TFT_FontDef Adafruit_TFT_Font_16x26;

#endif // __FONTS_H__