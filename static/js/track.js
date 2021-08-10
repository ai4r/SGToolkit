class Track {
    constructor(setting) {
        this.wavesurfer = WaveSurfer.create(setting);
    }

    getWaveSurfer() {
        return this.wavesurfer;
    }

    loadAudio(filename) {
        this.wavesurfer.load("media/" + filename + "/temp");
    }

    getCurrentTime() {
        return this.wavesurfer.getCurrentTime();
    }

    play() {
        this.wavesurfer.play();
    }

    pause() {
        this.wavesurfer.pause();
    }

    getDuration() {
        return this.wavesurfer.getDuration();
    }

    getPosition() {
        return this.getCurrentTime() / this.getDuration();
    }

    seekTo(position) {
        this.wavesurfer.seekTo(position);
    }

    timeToProgress(time) {
        return time / this.wavesurfer.getDuration();
    }

    isPlaying() {
        return this.wavesurfer.isPlaying();
    }

}
