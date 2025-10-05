#pragma once
#include <vector>
#include <string>
#include <complex>

class GWData
{
public:
    // Load LIGO data from file
    bool loadFromFile(const std::string &filename);

    // Get strain at a particular time
    double getStrain(double time) const;

    // Get the full time series
    const std::vector<double> &getTimeSeries() const { return times_; }
    const std::vector<double> &getStrainData() const { return strains_; }

    // Generate a simple GW waveform for testing
    void generateTestWaveform(double duration, double sample_rate,
                              double frequency, double amplitude);

private:
    std::vector<double> times_;
    std::vector<double> strains_;
};