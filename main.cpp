// #define DEBUG

#include "math.h"

#include "adafruit_tft/ili9341.hpp"
#include "adafruit_tft/fonts.h"

#include "daisy_seed.h"
#include "per/spi.h"
#include "per/i2c.h"
#include "per/qspi.h"

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
    spi_conf.clock_polarity  = daisy::SpiHandle::Config::ClockPolarity::LOW;
    spi_conf.clock_phase     = daisy::SpiHandle::Config::ClockPhase::ONE_EDGE;
    spi_conf.nss             = daisy::SpiHandle::Config::NSS::SOFT;
    spi_conf.baud_prescaler  = daisy::SpiHandle::Config::BaudPrescaler::PS_2;
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

    #ifdef DEBUG
    hw.PrintLine("=== spi init complete ===");
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
    hw.DelayMs(1000);
    ILI9341_Select();
    ILI9341_FillScreen_Raw(ILI9341_BLACK);
    ILI9341_Unselect();

    #ifdef DEBUG
    hw.PrintLine("=== ili9341 fill complete ===");
    #endif

    hw.DelayMs(1000);

    // ── Standing Wave Setup ──────────────────────────────────────────────────
    static const int   WAVE_CY    = ILI9341_HEIGHT / 2;            // 160
    static const int   WAVE_AMP   = 55;                             // px amplitude
    static const float WAVE_K     = (2.0f * 3.14159265f * 2.5f)   // 2.5 cycles
                                    / (float)ILI9341_WIDTH;
    static const float WAVE_OMEGA = 0.07f;                          // temporal speed
    static const int   WAVE_THICK = 2;                              // half-thickness → 5px band

    uint16_t prev_y[ILI9341_WIDTH];
    for(int i = 0; i < ILI9341_WIDTH; i++)
        prev_y[i] = (uint16_t)WAVE_CY;

    float wave_t = 0.0f;

    ILI9341_Select();   // ← assert CS ONCE for the entire frame
    while(1) {

        for(int x = 0; x < ILI9341_WIDTH; x++) {

            int ny = WAVE_CY + (int)(WAVE_AMP
                     * sinf(WAVE_K * (float)x)
                     * cosf(WAVE_OMEGA * wave_t));

            // clamp so the glow band stays on-screen
            if(ny < WAVE_THICK)                        ny = WAVE_THICK;
            if(ny > ILI9341_HEIGHT - WAVE_THICK - 1)  ny = ILI9341_HEIGHT - WAVE_THICK - 1;

            uint16_t new_y = (uint16_t)ny;

            if(new_y != prev_y[x]) {

                // erase old band — single SetAddressWindow + burst fill
                ILI9341_FillRectangle_Raw(
                    (uint16_t)x,
                    (uint16_t)(prev_y[x] - WAVE_THICK),
                    1,
                    (uint16_t)(2 * WAVE_THICK + 1),
                    ILI9341_BLACK
                );

                // draw new band — bright centre, softer glow edges
                ILI9341_DrawPixel_Raw((uint16_t)x, new_y - WAVE_THICK, ILI9341_BLUE);
                ILI9341_DrawPixel_Raw((uint16_t)x, new_y - 1,          ILI9341_CYAN);
                ILI9341_DrawPixel_Raw((uint16_t)x, new_y,              ILI9341_WHITE);
                ILI9341_DrawPixel_Raw((uint16_t)x, new_y + 1,          ILI9341_CYAN);
                ILI9341_DrawPixel_Raw((uint16_t)x, new_y + WAVE_THICK, ILI9341_BLUE);

                prev_y[x] = new_y;
            }
        }

        wave_t += 1.0f;
    }

    ILI9341_Unselect();  // ← release CS ONCE after the entire frame

}