/**
 * stringtone — Hall-Effect Sensor Instrument
 * ─────────────────────────────────────────────────────────────────────────────
 * Target : Daisy Seed (STM32H750) + libDaisy
 * Author : Alex Ikeda
 *
 * ── Hardware connections ─────────────────────────────────────────────────────
 *
 *  Selector mux  (controls which channel mux is enabled)
 *    S0 → D17 | S1 → D16 | S2 → D15
 *
 *  Channel muxes (8×, all share the same address lines)
 *    S0 → D20 | S1 → D19 | S2 → D18
 *
 *  Analog input  — active sensor output → A8
 *  Audio output  — waveform → OUT_L (audio out 1)
 *
 * ── Grid addressing (col 0..7, row 0..7) ─────────────────────────────────────
 *
 *  col_pair     = col / 2
 *  channel_base = (col % 2 == 0) ? 4 : 0      even cols use ch 4–7, odd use 0–3
 *  selector     = col_pair * 2 + (row >= 4)    top half → even sel, bottom → odd
 *  ch_address   = channel_base + (row % 4)
 *
 *  Example:  col=0 row=0 → sel=0 ch=4  (top of col 1, sensor a4, selector a0)
 *            col=0 row=6 → sel=1 ch=6  (bottom of col 1, sensor a6, selector a1)
 *            col=1 row=0 → sel=0 ch=0  (top of col 2, sensor a0, selector a0)
 *
 * ── Waveform semantics ───────────────────────────────────────────────────────
 *
 *  Per-column positions feed 8 control points: amp0..amp7
 *  ZERO_ROW (3.5) is the zero-amplitude reference — the midpoint between rows 3
 *  and 4.  The mapping is symmetric: p=0 → −1.0, p=3.5 → 0.0, p=7 → +1.0.
 *  At idle (no magnet), UpdatePosition holds each column at ZERO_ROW, so all
 *  control points remain at 0 and the output is silence.
 *  Interpolated via Catmull-Rom cubic Hermite → smooth but saw/triangle capable.
 *  Wavetable swaps use a short linear crossfade (CROSSFADE_LEN samples) so
 *  clicks are eliminated regardless of waveform shape at the swap point.
 *
 * ── Noise filtering ──────────────────────────────────────────────────────────
 *
 *  1. Deviation threshold (kNoiseFloor):       sensor differences (compared to baseline) 
 *                                              below this are ignored.
 *  2. Centroid weighting:                      position is the deviation-weighted
 *                                              centroid of all active sensors,
 *                                              giving sub-sensor precision for free.
 *  3. Hysteresis dead-zone (kPosDead):         new centroid must differ from current
 *                                              position by > kPosDead to register.
 *  4. Smoothing (kSmooth):                     moving average on position
 *                                              eliminates residual scan-to-scan jitter.
 */


// Install libDaisy here: https://daisy.audio/tutorials/cpp-dev-env/
#include <daisy_seed.h>
#include <dev/oled_ssd130x.h>
#include <per/i2c.h>

#include <algorithm>   // std::clamp
#include <cmath>       // std::abs, std::fabsf
#include <cstring>     // std::memset

using namespace daisy;

// ═════════════════════════════════════════════════════════════════════════════
//  Constants  (do not change at runtime)
// ═════════════════════════════════════════════════════════════════════════════

static constexpr int   STARTUP_DELAY_MS = 1000;
static constexpr int   BASELINE_FACTOR = 5; // Delay multiplier for baseline reads, to ensure the sensors have stabilised after changing MUX address
static constexpr int   NUM_COLS       = 8;
static constexpr int   NUM_ROWS       = 8;
static constexpr int   WAVETABLE_SIZE = 2048;   // samples per waveform period, linear interpolation between 
                                                // samples depending on sample rate
static constexpr int   CTRL_PTS       = 8;      // one control point per column, for Catmull-Rom interpolation
static constexpr float ZERO_ROW       = 3.5f;   // midpoint between rows 3 and 4 → 0.0 amplitude
                                                // p=0 → −1.0, p=3.5 → 0.0, p=7.0 → +1.0
static constexpr float SUPPLY_VOLTS   = 3.3f;   // ADC reference voltage

// Crossfade length in samples.  At 48 kHz this is ≈ 2.7 ms — long enough to
// suppress any discontinuity click, short enough to be imperceptible as lag.
// Must be shorter than one full scan cycle (≈ 9.6 ms at kScanDelayUs = 150).
static constexpr int   CROSSFADE_LEN  = 500;

// ═════════════════════════════════════════════════════════════════════════════
//  User-adjustable parameters
// ═════════════════════════════════════════════════════════════════════════════

static float baseline[NUM_COLS][NUM_ROWS];

static float    kNoiseFloor     = 0.040f;   // |deviation| below which a sensor is ignored (V)
static float    kPosDead        = 0.01f;    // Dead-zone for centroid hysteresis (row units)
static float    kSmooth         = 1.0f;     // Smoothing coefficient  [0=frozen, 1=instant]
static float    kSplineTension  = 0.0f;     // Tensiopn parameter for Catmull-Rom spline (0 = standard, 1 = linear)

static uint32_t kScanDelayUs = 100;    // µs to wait after changing MUX address

static bool     kDebugStep      = false;
static bool     kDebugDisplay   = true;
static bool     kDebugParams    = false;
static uint32_t kDebugDelayMs   = 100;

static float    kPlayFrequency  = 80.0f; // Hz — A3 by default

// ═════════════════════════════════════════════════════════════════════════════
//  Hardware objects
// ═════════════════════════════════════════════════════════════════════════════

static DaisySeed hw;

static SSD130xDriver<128, 64, SSD130xI2CTransport>         display;
static SSD130xDriver<128, 64, SSD130xI2CTransport>::Config display_config;

static GPIO sel_gpio[3];   // Selector mux: S0→D17, S1→D16, S2→D15
static GPIO ch_gpio[3];    // Channel muxes: S0→D20, S1→D19, S2→D18

static GPIO noise_floor_gpio;
static GPIO pos_dead_gpio;
static GPIO smooth_gpio;
static GPIO scan_delay_gpio;

// ═════════════════════════════════════════════════════════════════════════════
//  Sensor & position state
// ═════════════════════════════════════════════════════════════════════════════

static float sensor_v[NUM_COLS][NUM_ROWS]; // ADC readings in volts, updated each scan
static float col_pos[NUM_COLS];            // hysteresis-gated centroid positions (0..7)
static float col_pos_smooth[NUM_COLS];     // IIR-smoothed positions fed to waveform

// ═════════════════════════════════════════════════════════════════════════════
//  Wavetable — double-buffered for click-free updates
// ═════════════════════════════════════════════════════════════════════════════

static float            wavetable[2][WAVETABLE_SIZE];
static volatile int     active_buf  = 0;
static volatile bool    buf_pending = false;

static float phase = 0.0f; // normalised playback phase [0, 1)

// ═════════════════════════════════════════════════════════════════════════════
//  MUX addressing helpers
// ═════════════════════════════════════════════════════════════════════════════

static inline void SetSelectorAddr(uint8_t addr)
{
    sel_gpio[0].Write((addr >> 0) & 1u);
    sel_gpio[1].Write((addr >> 1) & 1u);
    sel_gpio[2].Write((addr >> 2) & 1u);
}

static inline void SetChannelAddr(uint8_t addr)
{
    ch_gpio[0].Write((addr >> 0) & 1u);
    ch_gpio[1].Write((addr >> 1) & 1u);
    ch_gpio[2].Write((addr >> 2) & 1u);
}

static void GetMuxAddrs(int col, int row, uint8_t &sel, uint8_t &ch)
{
    const int col_pair = col / 2;
    const int ch_base  = (col % 2 == 0) ? 4 : 0;
    sel = static_cast<uint8_t>(col_pair * 2 + (row >= 4 ? 1 : 0));
    ch  = static_cast<uint8_t>(ch_base  + (row % 4));
}

// ═════════════════════════════════════════════════════════════════════════════
//  Position estimation
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Compute a deviation-weighted centroid position for one column's sensor array.
 *
 * At idle (no magnet, all sensors near baseline), w_total < 1e-9 and
 * current_pos is returned unchanged.  Because current_pos is initialised to
 * ZERO_ROW (3.5), PosToAmplitude returns 0.0 → silence with no input.
 *
 * When a magnet is present the centroid naturally interpolates between adjacent
 * sensors, giving sub-sensor resolution for free.  Dead-zone hysteresis
 * prevents small noise perturbations from registering as position changes.
 *
 * @param voltages     NUM_ROWS sensor voltages for this column (volts).
 * @param baselines    NUM_ROWS per-sensor calibration baselines (volts).
 * @param current_pos  Previous position estimate [0.0, 7.0].
 * @return             Updated (or held) position in [0.0, 7.0].
 */
static float UpdatePosition(const float *voltages, const float *baselines, float current_pos)
{
    float w_sum   = 0.0f;
    float w_total = 0.0f;

    for (int r = 0; r < NUM_ROWS; r++) {
        float dev = fabsf(voltages[r] - baselines[r]);
        if (dev > kNoiseFloor) {
            w_sum   += static_cast<float>(r) * dev;
            w_total += dev;
        }
    }

    if (w_total < 1e-9f)
        return current_pos; // hold at ZERO_ROW when no sensor is active

    const float new_pos = w_sum / w_total;
    return (fabsf(new_pos - current_pos) > kPosDead) ? new_pos : current_pos;
}

/**
 * Map a continuous row position to audio amplitude.
 *
 * The mapping is a single symmetric linear expression:
 *
 *   A(p) = (p − ZERO_ROW) / ZERO_ROW
 *
 * With ZERO_ROW = 3.5 and p ∈ [0, 7]:
 *   p = 0.0  → A = −1.0
 *   p = 3.5  → A =  0.0   ← idle / neutral
 *   p = 7.0  → A = +1.0
 *
 * Because UpdatePosition initialises and holds at ZERO_ROW, all control
 * points evaluate to exactly 0.0 when no magnet is present.
 */
static float PosToAmplitude(float p)
{
    return (p - ZERO_ROW) / ZERO_ROW;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Wavetable builder — Catmull-Rom cubic Hermite interpolation
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Fill `table` (WAVETABLE_SIZE samples) by Catmull-Rom cubic Hermite
 * interpolation over CTRL_PTS uniformly-spaced control points.
 *
 * Tangents (local parameter space, one unit per segment):
 *   m[0]       = pts[1] − pts[0]                 forward difference
 *   m[k]       = (pts[k+1] − pts[k-1]) / 2       central difference (interior)
 *   m[CTRL-1]  = pts[CTRL-1] − pts[CTRL-2]       backward difference
 *
 * Cubic Hermite basis on segment [k, k+1], t ∈ [0,1]:
 *   p(t) = h00·p₀ + h10·m₀ + h01·p₁ + h11·m₁
 *   h00 =  2t³ − 3t² + 1
 *   h10 =   t³ − 2t² + t
 *   h01 = −2t³ + 3t²
 *   h11 =   t³ −  t²
 *
 * Output is clamped to [−1, 1] to guard against minor interpolation overshoot.
 */
static void BuildWavetable(float *table, const float *pts)
{
    const float s = (1.0f - kSplineTension);
    float m[CTRL_PTS];
    m[0]          = s * (pts[1] - pts[0]);
    m[CTRL_PTS-1] = s * (pts[CTRL_PTS-1] - pts[CTRL_PTS-2]);
    for (int k = 1; k < CTRL_PTS - 1; k++)
        m[k] = s * (pts[k+1] - pts[k-1]) * 0.5f;

    for (int i = 0; i < WAVETABLE_SIZE; i++) {
        const float tau = static_cast<float>(i) / static_cast<float>(WAVETABLE_SIZE);
        const float ts  = tau * static_cast<float>(CTRL_PTS - 1);

        int         seg = static_cast<int>(ts);
        if (seg >= CTRL_PTS - 1) seg = CTRL_PTS - 2;
        const float t   = ts - static_cast<float>(seg);
        const float t2  = t * t;
        const float t3  = t2 * t;

        const float h00 =  2.f*t3 - 3.f*t2 + 1.f;
        const float h10 =      t3 - 2.f*t2 + t;
        const float h01 = -2.f*t3 + 3.f*t2;
        const float h11 =      t3 -     t2;

        float v = h00 * pts[seg]
                + h10 * m[seg]
                + h01 * pts[seg + 1]
                + h11 * m[seg + 1];

        table[i] = fmaxf(-1.0f, fminf(v, 1.0f));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Audio callback  (runs in interrupt context at audio sample rate)
// ═════════════════════════════════════════════════════════════════════════════

static void AudioCallback(AudioHandle::InputBuffer  /*in*/,
                          AudioHandle::OutputBuffer out,
                          size_t                    size)
{
    const float p_inc = kPlayFrequency / hw.AudioSampleRate();

    static int xfade_ctr = 0; // Preserve the crossfade across buffer fills
    static int fade_from = 0;

    for (size_t i = 0; i < size; i++) {
        const float  f_idx = phase * static_cast<float>(WAVETABLE_SIZE);
        const int    i_idx = static_cast<int>(f_idx);
        const int    i_nxt = (i_idx + 1) % WAVETABLE_SIZE;
        const float  frac  = f_idx - static_cast<float>(i_idx);

        auto lerp_buf = [&](int buf) -> float {
            const float *tbl = wavetable[buf];
            return tbl[i_idx] + frac * (tbl[i_nxt] - tbl[i_idx]);
        };

        float sample;

        if (xfade_ctr > 0) {
            const float alpha = static_cast<float>(xfade_ctr)
                              / static_cast<float>(CROSSFADE_LEN);
            sample = alpha * lerp_buf(fade_from)
                   + (1.0f - alpha) * lerp_buf(active_buf);
            --xfade_ctr;
        } else {
            sample = lerp_buf(active_buf);

            if (buf_pending) {
                fade_from   = active_buf;
                active_buf ^= 1;
                xfade_ctr   = CROSSFADE_LEN;
                buf_pending = false;
            }
        }

        out[0][i] = sample;
        out[1][i] = 0.0f;

        phase += p_inc;
        if (phase >= 1.0f)
            phase -= 1.0f;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Main
// ═════════════════════════════════════════════════════════════════════════════

int main()
{
    hw.Init();
    hw.SetAudioBlockSize(64);
    hw.StartLog(false);

    System::Delay(STARTUP_DELAY_MS); // Startup delay to allow the power supply to stabilise for the OLED

    // ── Display — I2C1, D11 (SCL), D22 (SDA), address 0x3C ──────────
    //  periph and i2c_address must both be set; zero-initialised defaults
    //  leave the peripheral unselected, causing Init to silently fail and
    //  every subsequent Update() to block on a hardware timeout.
    display_config.transport_config.i2c_config = {
        .pin_config = {
            .scl = seed::D11, // I2C1 device SCL pin
            .sda = seed::D12, // I2C1 device SDA pin
        },
        .speed = I2CHandle::Config::Speed::I2C_400KHZ, // Should support up to 400kHz
    };
    display_config.transport_config.i2c_address = 0x3C; // 0x3C and 0x3D are the most common addresses for the SSD1306
                                                        // Could probe the pin to check?

    // Display initialization
    display.Init(display_config);
    // Probe the display with a fill+update to confirm the bus is alive.
    // If this hangs (ie, the 27 second startup delay), the peripheral, address, or pull-up resistors are wrong.
    display.Fill(false);
    display.Update();
    hw.PrintLine("Display initialized. This is a false message if audio took more than 20 seconds to start up.");

    size_t DISPLAY_WIDTH = display.Width(); // Get display width in case a different dim display is connected
    size_t DISPLAY_HEIGHT = display.Height(); // Get display height in case a different dim display is connected

    // ── GPIO: selector mux — S0→D17, S1→D16, S2→D15 ─────────────────────────
    static const Pin kSelPins[3] = { seed::D17, seed::D16, seed::D15 };
    for (int i = 0; i < 3; i++)
        sel_gpio[i].Init(kSelPins[i], GPIO::Mode::OUTPUT);

    // ── GPIO: channel muxes — S0→D20, S1→D19, S2→D18 ────────────────────────
    static const Pin kChPins[3] = { seed::D20, seed::D19, seed::D18 };
    for (int i = 0; i < 3; i++)
        ch_gpio[i].Init(kChPins[i], GPIO::Mode::OUTPUT);

    // ── GPIO: debug parameter pins (D24..D31) ─────────────────────────────────
    if (kDebugParams) { // RESERVED FOR FUTURE USE TO TUNE PARAMETERS WITH POTS
        static const Pin kNoiseFloorPin = seed::A9;
        static const Pin kPosDeadPin    = seed::A10;
        static const Pin kSmoothPin     = seed::A11;
        static const Pin kScanDelayPin  = seed::A12;

        noise_floor_gpio.Init(kNoiseFloorPin, GPIO::Mode::INPUT);
        pos_dead_gpio.Init(kPosDeadPin,       GPIO::Mode::INPUT);
        smooth_gpio.Init(kSmoothPin,          GPIO::Mode::INPUT);
        scan_delay_gpio.Init(kScanDelayPin,   GPIO::Mode::INPUT);
    }

    // ── ADC: single channel on A8 ─────────────────────────────────────────────
    AdcChannelConfig adc_cfg;
    adc_cfg.InitSingle(seed::A8);
    hw.adc.Init(&adc_cfg, 1);
    hw.adc.Start();

    // ── Initialise sofware state — all columns start at ZERO_ROW → amplitude = 0
    for (int c = 0; c < NUM_COLS; c++) {
        col_pos[c]        = ZERO_ROW;
        col_pos_smooth[c] = ZERO_ROW;
        for (int r = 0; r < NUM_ROWS; r++)
            sensor_v[c][r] = 0.0f;
    }
    std::memset(wavetable, 0, sizeof(wavetable));

    System::Delay(STARTUP_DELAY_MS); // Another delay to ensure the sensors are stabilised at their baseline before calibration

    // ── Per-sensor baseline calibration ──────────────────────────────────────
    //  Averages CAL_SAMPLES reads per sensor at power-on.
    //  Ensure no magnets are near the array during this window.
    //  In the future, add a button to retare the instrument wihtout needing to reboot it
    hw.PrintLine("Calibrating...");
    static constexpr int CAL_SAMPLES = 16;
    for (int c = 0; c < NUM_COLS; c++) {
        for (int r = 0; r < NUM_ROWS; r++) {
            uint8_t sel_addr, ch_addr;
            GetMuxAddrs(c, r, sel_addr, ch_addr);
            SetSelectorAddr(sel_addr);
            SetChannelAddr(ch_addr);
            System::DelayUs(BASELINE_FACTOR * kScanDelayUs);

            float acc = 0.0f;
            for (int s = 0; s < CAL_SAMPLES; s++) {
                acc += hw.adc.GetFloat(0) * SUPPLY_VOLTS;
                System::DelayUs(BASELINE_FACTOR * kScanDelayUs);
            }
            baseline[c][r] = acc / static_cast<float>(CAL_SAMPLES);

        }
    }
    hw.PrintLine("Calibration done.");

    hw.StartAudio(AudioCallback);
    hw.PrintLine("Stringtone ready. kDebugStep=%d  kPlayFrequency=%d.%01d Hz",
        (int)kDebugStep,
        (int)kPlayFrequency,
        (int)(kPlayFrequency * 10) % 10);

    // ═════════════════════════════════════════════════════════════════════════
    //  Main scan loop
    //  Each iteration: scan 64 sensors → compute positions → build wavetable
    // ═════════════════════════════════════════════════════════════════════════

    float normalized = 0;

    while (true) {

        // ── Step 1: scan all 64 sensors ───────────────────────────────────────

        for (int c = 0; c < NUM_COLS; c++) {
            for (int r = 0; r < NUM_ROWS; r++) {

                uint8_t sel_addr, ch_addr;
                GetMuxAddrs(c, r, sel_addr, ch_addr);

                SetSelectorAddr(sel_addr);
                SetChannelAddr(ch_addr);
                System::DelayUs(kScanDelayUs);

                normalized = hw.adc.GetFloat(0);
                sensor_v[c][r] = normalized * SUPPLY_VOLTS;

                if (kDebugStep) { // This prints while the sensor scan occurs
                    hw.PrintLine( // Should fix output since float print is not enabled on the Daisy Seed by default
                        "[C%d R%d] sel=%u ch=%u  %d.%04d (norm)  %d.%04d V  dev=%c%d.%04d V",
                        c, r,
                        static_cast<unsigned>(sel_addr),
                        static_cast<unsigned>(ch_addr),
                        (int)normalized,
                        (int)(normalized * 10000) % 10000,
                        (int)sensor_v[c][r],
                        (int)(fabsf(sensor_v[c][r]) * 10000) % 10000,
                        (sensor_v[c][r] - baseline[c][r] >= 0) ? '+' : '-',
                        (int)fabsf(sensor_v[c][r] - baseline[c][r]),
                        (int)(fabsf(sensor_v[c][r] - baseline[c][r]) * 10000) % 10000);
                    System::Delay(kDebugDelayMs);
                }
            }
        }

        // ── Step 2: update per-column positions ───────────────────────────────

        for (int c = 0; c < NUM_COLS; c++) {
            col_pos[c] = UpdatePosition(sensor_v[c], baseline[c], col_pos[c]);
            col_pos_smooth[c] += kSmooth * (col_pos[c] - col_pos_smooth[c]);
        }

        // ── Step 3: build waveform control points ─────────────────────────────

        float ctrl[CTRL_PTS];
        for (int c = 0; c < NUM_COLS; c++)
            ctrl[c] = PosToAmplitude(col_pos_smooth[c]);

        // ── Step 4: fill inactive buffer, signal pending swap ─────────────────

        const int next_buf = active_buf ^ 1;
        BuildWavetable(wavetable[next_buf], ctrl);
        buf_pending = true;

        // Logging positions if kDebugStep is enabled
        if (kDebugStep) {
            hw.Print("Positions: ");
            for (int c = 0; c < NUM_COLS; c++)
                hw.Print("%c%d.%02d ",
                    col_pos_smooth[c] < 0 ? '-' : ' ',
                    (int)fabsf(col_pos_smooth[c]),
                    (int)(fabsf(col_pos_smooth[c]) * 100) % 100);
            hw.PrintLine("");
            System::Delay(kDebugDelayMs);
        }

        // Displaying positions if kDebugDisplay is enabled and the display initialized successfully
        if (kDebugDisplay) {
            display.Fill(false); // Clear display
            int w_inc = WAVETABLE_SIZE / static_cast<int>(DISPLAY_WIDTH);
            float DISPLAY_HEIGHT_ADJ = static_cast<float>(DISPLAY_HEIGHT) - 1.0f; // Adjust for zero-based indexing
            for (size_t x = 0; x < DISPLAY_WIDTH; x++) { // Draw pixels scaled to the display resolution (width)
                float sample = wavetable[next_buf][x * static_cast<size_t>(w_inc)];
                size_t y = static_cast<size_t>(
                    std::clamp((DISPLAY_HEIGHT_ADJ / 2.0f) - sample * (DISPLAY_HEIGHT_ADJ / 2.0f), 0.0f, DISPLAY_HEIGHT_ADJ)
                ); // Rescale audio amplitude to screen height
                display.DrawPixel(x, y, true);
            }
            display.Update(); // Send buffer to display
        }
    }
}