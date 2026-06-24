#ifndef __ILI9341_TOUCH_HPP__
#define __ILI9341_TOUCH_HPP__

#include "daisy_seed.h"
#include "per/i2c.h"
#include "stm32h7xx_hal.h"

#define FT6206_TIMEOUT HAL_MAX_DELAY

using namespace daisy;

// NOTE: I2C SCL, SDA pins are defined by the I2CHandle::Config type passed into the initialization of the I2C device
extern GPIO gpio_irq;

#define FT6206_ADDR 0x38 << 1
extern I2CHandle i2c_handle;

// change depending on screen orientation
#define FT6206_SCALE_X 240
#define FT6206_SCALE_Y 320

// to calibrate uncomment UART_Printf line in ft6206.c
#define FT6206_MIN_RAW_X 1500
#define FT6206_MAX_RAW_X 31000
#define FT6206_MIN_RAW_Y 3276
#define FT6206_MAX_RAW_Y 30110

bool FT6206_Is_Pressed();
bool FT6206_TouchGetCoordinates(uint16_t* x, uint16_t* y);

#endif // __ILI9341_TOUCH_HPP__