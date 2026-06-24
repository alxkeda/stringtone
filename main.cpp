#include "adafruit_tft/ili9341.hpp"
#include "adafruit_tft/fonts.h"

#include "daisy_seed.h"
#include "per/spi.h"
#include "per/i2c.h"

using namespace daisy;
using namespace daisy::seed;

DaisySeed hw;

daisy::SpiHandle            spi_handle;
daisy::SpiHandle::Config    spi_conf;
daisy::I2CHandle            i2c_handle;
daisy::I2CHandle::Config    i2c_conf;

GPIO gpio_dc;
GPIO gpio_res;
GPIO gpio_cs;
GPIO gpio_touch_irq;

void GPIO_Init(void) {
    gpio_dc.Init(D5, GPIO::Mode::OUTPUT);           // pin 6  D5    PORTD GPIO_PIN_2
    gpio_res.Init(D6, GPIO::Mode::OUTPUT);          // pin 7  D6    PORTC GPIO_PIN_12
    gpio_cs.Init(D7, GPIO::Mode::OUTPUT);           // pin 8  D7    PORTG GPIO_PIN_10
    gpio_touch_irq.Init(D13, GPIO::Mode::INPUT);    // pin 14 D13   PORTB GPIO_PIN_6
}

void SPI_Init(void) {
    spi_conf.periph          = daisy::SpiHandle::Config::Peripheral::SPI_1;
    spi_conf.mode            = daisy::SpiHandle::Config::Mode::MASTER;
    spi_conf.direction       = daisy::SpiHandle::Config::Direction::TWO_LINES;
    spi_conf.clock_polarity  = daisy::SpiHandle::Config::ClockPolarity::HIGH;
    spi_conf.clock_phase     = daisy::SpiHandle::Config::ClockPhase::ONE_EDGE;
    spi_conf.nss             = daisy::SpiHandle::Config::NSS::SOFT;
    spi_conf.baud_prescaler  = daisy::SpiHandle::Config::BaudPrescaler::PS_32;
    spi_conf.datasize        = 8;

    spi_conf.pin_config.sclk = {DSY_GPIOG, 11};  // pin 9  D8
    spi_conf.pin_config.miso = {DSY_GPIOB,  4};  // pin 10 D9
    spi_conf.pin_config.mosi = {DSY_GPIOB,  5};  // pin 11 D10

    spi_handle.Init(spi_conf);
}

void I2C_Init(void) {
    i2c_conf.periph          = daisy::I2CHandle::Config::Peripheral::I2C_1;
    i2c_conf.speed           = daisy::I2CHandle::Config::Speed::I2C_100KHZ;

    i2c_conf.pin_config.scl  = {DSY_GPIOB, 8};  // pin 12 D11
    i2c_conf.pin_config.sda  = {DSY_GPIOB, 9};  // pin 13 D12

    i2c_handle.Init(i2c_conf);
}

int main(void) {

    hw.Init();
    hw.StartLog(true);
    hw.PrintLine("=== boot complete ===");

    SPI_Init();

    hw.PrintLine("=== spi init complete ===");

    GPIO_Init();

    hw.PrintLine("=== gpio init complete ===");

    ILI9341_Unselect();
    ILI9341_Init();

    hw.PrintLine("=== ili9341 init complete ===");

    ILI9341_FillScreen(ILI9341_BLACK);
    ILI9341_WriteString(10, 10, "Stringtone", Adafruit_TFT_Font_11x18, ILI9341_WHITE, ILI9341_BLACK);

    hw.PrintLine("=== ili9341 fill complete ===");

    while (1) {
    }
}