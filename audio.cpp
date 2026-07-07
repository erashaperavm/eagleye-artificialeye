#include "audio.h"
#include <portaudio.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <cmath>
#include <iostream>

struct AudioCapture::Impl {
    std::atomic<bool> running{false};
    std::thread th;
    std::mutex mtx;
    std::deque<float> waveform; // circular buffer of recent samples (mono)
    float direction = NAN; // -1..1
    bool vad = false;
    float level = 0.0f;
};

AudioCapture::AudioCapture() { impl = new Impl(); }
AudioCapture::~AudioCapture() { stop(); delete impl; }

bool AudioCapture::start() {
    if (impl->running) return true;
    PaError err = Pa_Initialize();
    if (err != paNoError) { std::cerr << "PortAudio init failed\n"; return false; }
    PaStreamParameters inputParams;
    inputParams.device = Pa_GetDefaultInputDevice();
    if (inputParams.device == paNoDevice) { std::cerr << "No default input device\n"; Pa_Terminate(); return false; }
    const PaDeviceInfo* info = Pa_GetDeviceInfo(inputParams.device);
    int chCount = std::min(2, info->maxInputChannels); // prefer stereo
    inputParams.channelCount = chCount;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = info->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    double sampleRate = 16000.0;
    PaStream* stream;
    err = Pa_OpenStream(&stream, &inputParams, nullptr, sampleRate, 512, paNoFlag, nullptr, nullptr);
    if (err != paNoError) { std::cerr << "PortAudio open stream failed: "<< Pa_GetErrorText(err) << std::endl; Pa_Terminate(); return false; }
    err = Pa_StartStream(stream);
    if (err != paNoError) { std::cerr << "PortAudio start stream failed\n"; Pa_CloseStream(stream); Pa_Terminate(); return false; }

    impl->running = true;
    impl->th = std::thread([this, stream, chCount]() {
        const int frameSize = 512;
        std::vector<float> buffer(frameSize * chCount);
        while (impl->running) {
            PaError r = Pa_ReadStream(stream, buffer.data(), frameSize);
            if (r != paNoError) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
            // compute mono mix and RMS
            float sumsqL = 0.0f, sumsqR = 0.0f;
            std::vector<float> mono(frameSize);
            for (int i = 0; i < frameSize; ++i) {
                float l = buffer[i*chCount + 0];
                float rch = (chCount>1) ? buffer[i*chCount + 1] : buffer[i*chCount + 0];
                mono[i] = 0.5f*(l+rch);
                sumsqL += l*l; sumsqR += rch*rch;
            }
                    float rmsL = std::sqrt(sumsqL/frameSize);
            float rmsR = std::sqrt(sumsqR/frameSize);
            // simple direction estimate
            if (chCount > 1) {
                impl->direction = (rmsR - rmsL) / (rmsR + rmsL + 1e-6f); // -1..1
            } else impl->direction = NAN;
            // simple VAD energy threshold
            float rms = 0.0f; for (auto v: mono) rms += v*v; rms = std::sqrt(rms/frameSize);
            impl->vad = (rms > 0.01f);
            impl->level = rms;

            // store samples (thread-safe)
            {
                std::lock_guard<std::mutex> lk(impl->mtx);
                for (float s: mono) {
                    impl->waveform.push_back(s);
                    if (impl->waveform.size() > 16000) impl->waveform.pop_front();
                }
            }
        }
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
    });
    return true;
}

void AudioCapture::stop() {
    if (!impl->running) return;
    impl->running = false;
    if (impl->th.joinable()) impl->th.join();
}

void AudioCapture::getWaveform(std::vector<float>& out) {
    std::lock_guard<std::mutex> lk(impl->mtx);
    out.assign(impl->waveform.begin(), impl->waveform.end());
}

float AudioCapture::getDirection() { return impl->direction; }
bool AudioCapture::getVAD() { return impl->vad; }
float AudioCapture::getLevel() { return impl->level; }
