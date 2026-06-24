// TODO:    Replace all memory addresses with macros describing the memory address
//          Change nsamples to a macro in the header 

#include "ft6206.hpp"

#define READ_X 0xD0
#define READ_Y 0x90

bool FT6206_Is_Pressed() {
    return gpio_irq.Read() == false;
}

bool FT6206_TouchGetCoordinates(uint16_t* x, uint16_t* y) {
    static const uint8_t cmd_read_x[] = { READ_X };
    static const uint8_t cmd_read_y[] = { READ_Y };
    static const uint8_t zeroes_tx[] = { 0x00, 0x00 };

    uint32_t avg_x = 0;
    uint32_t avg_y = 0;
    uint8_t nsamples = 0;
    for(uint8_t i = 0; i < 16; i++) {
        if(!FT6206_Is_Pressed())
            break;

        nsamples++;

        i2c_handle.TransmitBlocking(FT6206_ADDR, (uint8_t*)cmd_read_y, sizeof(cmd_read_y), FT6206_TIMEOUT);
        uint8_t y_raw[2];
        i2c_handle.ReceiveBlocking(FT6206_ADDR, y_raw, sizeof(y_raw), FT6206_TIMEOUT);

        i2c_handle.TransmitBlocking(FT6206_ADDR, (uint8_t*)cmd_read_x, sizeof(cmd_read_x), FT6206_TIMEOUT);
        uint8_t x_raw[2];
        i2c_handle.ReceiveBlocking(FT6206_ADDR, x_raw, sizeof(x_raw), FT6206_TIMEOUT);

        avg_x += (((uint16_t)x_raw[0]) << 8) | ((uint16_t)x_raw[1]);
        avg_y += (((uint16_t)y_raw[0]) << 8) | ((uint16_t)y_raw[1]);
    }

    if(nsamples < 16)
        return false;

    uint32_t raw_x = (avg_x / 16);
    if(raw_x < FT6206_MIN_RAW_X) raw_x = FT6206_MIN_RAW_X;
    if(raw_x > FT6206_MAX_RAW_X) raw_x = FT6206_MAX_RAW_X;

    uint32_t raw_y = (avg_y / 16);
    if(raw_y < FT6206_MIN_RAW_Y) raw_y = FT6206_MIN_RAW_Y;
    if(raw_y > FT6206_MAX_RAW_Y) raw_y = FT6206_MAX_RAW_Y;

    // Uncomment this line to calibrate touchscreen:
    // UART_Printf("raw_x = %d, raw_y = %d\r\n", x, y);

    *x = (raw_x - FT6206_MIN_RAW_X) * FT6206_SCALE_X / (FT6206_MAX_RAW_X - FT6206_MIN_RAW_X);
    *y = (raw_y - FT6206_MIN_RAW_Y) * FT6206_SCALE_Y / (FT6206_MAX_RAW_Y - FT6206_MIN_RAW_Y);

    return true;
}