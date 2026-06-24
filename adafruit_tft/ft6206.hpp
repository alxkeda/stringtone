// MODIFIED 6/22 BY ALEX IKEDA FOR CAPACITIVE I2C TOUCHSCREEN CONTROLLER

/* vim: set ai et ts=4 sw=4: */
#ifndef __ILI9341_TOUCH_H__
#define __ILI9341_TOUCH_H__

#include <stdbool.h>

#include "per/i2c.h"

#define ILI9341_TOUCH_I2C_PORT hi2c1
extern daisy::I2CHandle i2c_handle;

#define ILI9341_TOUCH_ADDR 0x38 << 1

// Note:    pin 12; D11; I2C1 SCL; I2C1_SCL; PB8
//          pin 13; D12; I2C1 SDA; I2C1_SDA; PB9

#define FT6206_SDA_Pin       GPIO_PIN_9      // pin 13; D12; I2C1 SDA; I2C1_SDA; PB9
#define FT6206_SDA_GPIO_Port GPIOB
#define FT6206_SCL_Pin       GPIO_PIN_8      // pin 12; D11; I2C1 SCL; I2C1_SCL; PB8
#define FT6206_SCL_GPIO_Port GPIOB
#define FT6206_IRQ_Pin       GPIO_PIN_6      // pin 14; D13; IRQ; PB6
#define FT6206_IRQ_GPIO_Port GPIOB

// change depending on screen orientation
#define FT6206_SCALE_X 240
#define FT6206_SCALE_Y 320

// to calibrate uncomment UART_Printf line in ft6206.c
#define FT6206_MIN_RAW_X 1500
#define FT6206_MAX_RAW_X 31000
#define FT6206_MIN_RAW_Y 3276
#define FT6206_MAX_RAW_Y 30110

bool FT6206_TouchPressed();
bool FT6206_TouchGetCoordinates(uint16_t* x, uint16_t* y);

#endif // __ILI9341_TOUCH_H__