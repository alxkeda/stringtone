/* ----- UNCOMMENT DEBUG TO ENABLE DEBUGGING IN ONE GO!             ----- */
// #define DEBUG

/* ----- Standard library includes                                  ----- */
#include "math.h"

/* ----- TFT screen library includes                                ----- */
#include "touchscreen/ili9341.hpp"
#include "touchscreen/fonts.h"
#include "touchscreen/ft6206.hpp"

/* ----- libDaisy includes                                          ----- */
#include "daisy_seed.h"
#include "per/spi.h"
#include "per/i2c.h"
// #include "per/qspi.h"

using namespace daisy;          // For hardware device handles; SPI, I2C, DaisySeed, etc.
using namespace daisy::seed;    // For GPIO pins; D1, D2, D3, etc.

DaisySeed hw;

/* ----- GPIO for Adafruit TFT                                      ----- */
GPIO gpio_dc;   // SPI
GPIO gpio_res;  // SPI
GPIO gpio_cs;   // SPI
GPIO gpio_irq;  // I2C

void GPIO_Init(void) {
    gpio_dc.Init(D5, GPIO::Mode::OUTPUT);           // pin 6  D5    PORTD GPIO_PIN_2
    gpio_res.Init(D6, GPIO::Mode::OUTPUT);          // pin 7  D6    PORTC GPIO_PIN_12
    gpio_cs.Init(D7, GPIO::Mode::OUTPUT);           // pin 8  D7    PORTG GPIO_PIN_10
    gpio_irq.Init(D13, GPIO::Mode::INPUT);          // pin 14 D13   PORTB GPIO_PIN_6
}

/* ----- SPI for ILI9341 screen; I2C for FT6206 touch controller    ----- */
SpiHandle            spi_handle;
SpiHandle::Config    spi_conf;
I2CHandle            i2c_handle;
I2CHandle::Config    i2c_conf;

void SPI_Init(void) {
    spi_conf.periph          = SpiHandle::Config::Peripheral::SPI_1;
    spi_conf.mode            = SpiHandle::Config::Mode::MASTER;
    spi_conf.direction       = SpiHandle::Config::Direction::TWO_LINES;
    spi_conf.clock_polarity  = SpiHandle::Config::ClockPolarity::LOW;
    spi_conf.clock_phase     = SpiHandle::Config::ClockPhase::ONE_EDGE;
    spi_conf.nss             = SpiHandle::Config::NSS::SOFT;
    spi_conf.baud_prescaler  = SpiHandle::Config::BaudPrescaler::PS_2;
    spi_conf.datasize        = 8;

    spi_conf.pin_config.sclk = {DSY_GPIOG, 11};     // pin 9  D8    PORTG GPIO_PIN_11
    spi_conf.pin_config.miso = {DSY_GPIOB,  4};     // pin 10 D9    PORTB GPIO_PIN_4
    spi_conf.pin_config.mosi = {DSY_GPIOB,  5};     // pin 11 D10   PORTB GPIO_PIN_5

    spi_handle.Init(spi_conf);
}

void I2C_Init(void) {
    i2c_conf.periph          = I2CHandle::Config::Peripheral::I2C_1;
    i2c_conf.speed           = I2CHandle::Config::Speed::I2C_400KHZ;
    i2c_conf.mode            = I2CHandle::Config::Mode::I2C_MASTER;

    i2c_conf.pin_config.scl  = {DSY_GPIOB, 8};      // pin 12 D11   PORTB GPIO_PIN_8   
    i2c_conf.pin_config.sda  = {DSY_GPIOB, 9};      // pin 13 D12   PORTB GPIO_PIN_9

    i2c_handle.Init(i2c_conf);
}

int main(void) {

    hw.Init();

    // QSPIHandle::Config qspi_config = {
    //     .device = QSPIHandle::Config::Device::IS25LP064A,
    //     .mode = QSPIHandle::Config::Mode::MEMORY_MAPPED
    // };
    // hw.qspi.Init(qspi_config);

    #ifdef DEBUG
    hw.StartLog(true);
    hw.PrintLine("=== boot complete ===");
    #endif

    SPI_Init();
    I2C_Init();

    #ifdef DEBUG
    hw.PrintLine("=== spi/i2c init complete ===");
    #endif

    GPIO_Init();

    #ifdef DEBUG
    hw.PrintLine("=== gpio init complete ===");
    #endif

    ILI9341_Unselect();
    ILI9341_Init();

    #ifdef DEBUG
    hw.PrintLine("=== ili9341 init complete ===");
    #endif

    ILI9341_Select();
    ILI9341_FillScreen_Raw(ILI9341_WHITE);
    ILI9341_Unselect();

    #ifdef DEBUG
    hw.PrintLine("=== ili9341 fill complete ===");
    #endif

    ILI9341_Select();   // ← assert CS ONCE for the entire frame

    bool STATE_PRESSED = false;

    while(1) {
        if(FT6206_Is_Pressed() && !STATE_PRESSED) {
            ILI9341_Select();
            ILI9341_FillScreen_Raw(ILI9341_BLACK);
            ILI9341_Unselect();
            STATE_PRESSED = true;
        } else if (FT6206_Is_Pressed() && STATE_PRESSED) {
            ILI9341_Select();
            ILI9341_FillScreen_Raw(ILI9341_WHITE);
            ILI9341_Unselect();
            STATE_PRESSED = false;
        }
        hw.DelayMs(10);

    }
    ILI9341_Unselect();  // ← release CS ONCE after the entire frame

}