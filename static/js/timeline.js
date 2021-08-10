class Timeline {
    constructor(avatar, stylePanel, audio_div_id, wordCanvasId, motionControlContainerId, styleControlContainerId) {
        this.avatar = avatar;
        this.stylePanel = stylePanel;
        this.audio_div_id = audio_div_id;
        this.audioTrack = createAudioTrack(audio_div_id);
        this.wordCanvasId = wordCanvasId;
        this.motionControlContainerId = motionControlContainerId;
        this.styleControlContainerId = styleControlContainerId;
        this.historyManager = new HistoryManager();

        this.syncTracksOnPlayback();
        this.syncTracksOnSeek();
        this.controlDataModificationInvalidWarningId = null;
    }

    setCursorToStart() {
        this.audioTrack.seekTo(0);
    }

    load(filename, words, keypoints, styleNames) {
        var that = this;
        this.audioTrack.loadAudio(filename);
        this.audioTrack.wavesurfer.once('ready', function () {
            that.createControlTracks(keypoints, styleNames);

            function showWarning() {
                $(that.controlDataModificationInvalidWarningId).modal('show');
            }

            that.motionTrack.setCellGroupCreationInvalidCallback(showWarning);
            that.styleTrack.setCellGroupCreationInvalidCallback(showWarning);

            that.displayWords(words);
        });
    }

    displayWords(words) {
        var canvas = document.getElementById(this.wordCanvasId);
        canvas.width = canvas.clientWidth * 2;
        canvas.height = canvas.clientHeight * 2;
        var ctx = canvas.getContext('2d');
        // ctx.fillStyle = "gray";
        // ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = 'black';
        ctx.textAlign = 'center';

        if (canvas.clientWidth < 480) {
            ctx.font = '14px sans-serif';
        } else if (canvas.clientWidth < 768) {
            ctx.font = '18px sans-serif';
        } else {
            ctx.font = '25px sans-serif';
        }

        let audioDuration = this.audioTrack.getDuration();
        var wordPosition = 0;  // in px
        var wordWidth = 0;
        var y = 40;
        for (var word of words) {
            wordWidth = ctx.measureText(word[0]).width;
            wordPosition = (word[1] + word[2]) * 0.5 / audioDuration * canvas.width;

            ctx.save();
            ctx.translate(wordPosition, y)
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(word[0], 0, y / 2);
            ctx.restore();
        }
    }

    createControlTracks(keypoints, styleNames) {
        this.createMotionTrack(keypoints);
        this.createStyleTrack(keypoints.length, styleNames);

        this.syncAvatarWithCurrentCursor();
        this.setCellSelectedCallbacks();
    }

    syncAvatarWithCurrentCursor() {
        this.motionTrack.syncAvatarWithCursor(this.avatar);
    }

    syncStylePanelWithCurrentCursor() {
        this.styleTrack.syncStylePanelWithCursor(this.stylePanel);
    }

    createMotionTrack(keypoints) {
        var setting = {
            container: this.motionControlContainerId,
            numFrames: keypoints.length,
            duration: this.audioTrack.getDuration(),
        };

        if (this.motionTrack != null) {
            this.motionTrack.destruct();
        }
        this.motionTrack = new MotionCellTrack(setting, keypoints);
        this.motionTrack.seekTo(0);
    }

    createStyleTrack(nframes, styleNames) {
        var setting = {
            container: this.styleControlContainerId,
            numFrames: nframes,
            duration: this.audioTrack.getDuration(),
        };

        if (this.styleTrack != null) {
            this.styleTrack.destruct();
        }

        this.styleTrack = new StyleCellTrack(setting, styleNames);
    }

    updateMotionKeypoitns(keypoints) {
        this.motionTrack.updateBaseKeypoints(keypoints);
    }

    syncTracksOnPlayback() {
        this.audioTrack.wavesurfer.on('audioprocess', function (currentTime) {
            this.motionTrack.seekToIfOneCellSelected(currentTime);
            this.styleTrack.seekToIfOneCellSelected(currentTime);
            this.syncAvatarWithCurrentCursor();
        }.bind(this));
    }

    addCallbackOn(action, func) {
        this.audioTrack.wavesurfer.on(action, func);
    }

    getCurrentTime() {
        return this.audioTrack.getCurrentTime();
    }

    getDuration() {
        return this.audioTrack.getDuration();
    }

    syncTracksOnSeek() {
        this.audioTrack.wavesurfer.on('seek', function (position) {
            var time = position * this.audioTrack.getDuration();
            this.motionTrack.seekToIfOneCellSelected(time);
            this.styleTrack.seekToIfOneCellSelected(time);
            this.syncAvatarWithCurrentCursor();
            this.syncStylePanelWithCurrentCursor();
        }.bind(this));
    }

    play() {
        this.styleTrack.deselectAll();
        this.motionTrack.seekTo(0);
        this.avatar.restPose();
        this.audioTrack.play();
    }

    pause() {
        this.audioTrack.pause();
    }

    isPlaying() {
        return this.audioTrack.isPlaying();
    }

    setCellSelectedCallbacks() {

        function seekTrackToCellMiddle(track, cell) {
            var st = cell.getData('start-time');
            var et = cell.getData('end-time');
            var progress = track.timeToProgress((st + et) / 2);
            track.seekTo(progress);
        };

        this.motionTrack.addSelectedCallback(function (selectedCell) {
            if (selectedCell != null) {
                this.styleTrack.deselectAll();
                seekTrackToCellMiddle(this.audioTrack, selectedCell);
            }
        }.bind(this));

        this.styleTrack.addSelectedCallback(function (selectedCell) {
            if (selectedCell != null) {
                this.motionTrack.deselectAll();
                seekTrackToCellMiddle(this.audioTrack, selectedCell);
            }
        }.bind(this));
    }

    getMotionConstraints() {
        return this.motionTrack.getKeypointsConstraint();
    }

    getStyleConstraints() {
        return this.styleTrack.getStyleConstraints();
    }

    deleteSelectedCellControlData() {
        this.motionTrack.deleteSelectedCellControlData();
        this.styleTrack.deleteSelectedCellControlData();
        this.syncAvatarWithCurrentCursor();
        this.syncStylePanelWithCurrentCursor();
    }

    toggleMotionTrackControlDataBypass(isBypass) {
        this.motionTrack.isBypassControlData = isBypass;
        this.syncAvatarWithCurrentCursor();
    }

    updateMotionTrackControlData(keypoints) {
        this.motionTrack.updateMultipleKeypointsAsGroup(keypoints);
        this.syncAvatarWithCurrentCursor();
    }

    updateStyleTrackControlData(data) {
        if (this.styleTrack == null) {
            return;
        }
        this.styleTrack.updateStyleControlForSelectedCells(data);
    }

    setPlayCallback(func) {
        this.audioTrack.wavesurfer.on('play', func);
    }

    setPauseCallback(func) {
        this.audioTrack.wavesurfer.on('pause', func);
    }

    isAnyCellModified() {
        if (this.motionTrack == null || this.styleTrack == null) {
            return false;
        }
        return this.motionTrack.isAnyCellModified() || this.styleTrack.isAnyCellModified();
    }

    fillMotionControl() {
        this.motionTrack.interpolateTwoPoses();
    }

    copySelectedCellGroup() {
        if (this.motionTrack != null && this.motionTrack.isAnyCellSelected()) {
            this.motionTrack.copyGroup();
        } else if (this.styleTrack != null && this.styleTrack.isAnyCellSelected()) {
            this.styleTrack.copyGroup();
        }
    }

    pasteSelectedCellGroup() {
        if (this.motionTrack != null && this.motionTrack.isAnyCellSelected()) {
            this.motionTrack.pasteGroup();
        } else if (this.styleTrack != null && this.styleTrack.isAnyCellSelected()) {
            this.styleTrack.pasteGroup();
        }
    }

    saveTracksToHistory() {
        this.historyManager.addSnapshots(this.motionTrack, this.styleTrack);
    }

    clearHistory() {
        this.historyManager.clearHistory();
    }

    undoTracks() {
        this.historyManager.undo(this.motionTrack, this.styleTrack);
    }

    redoTracks() {
        this.historyManager.redo(this.motionTrack, this.styleTrack);
    }

    canUndo() {
        return this.historyManager.canUndo();
    }

    canRedo() {
        return this.historyManager.canRedo();
    }

}

function createAudioTrack(div_id) {
    var setting = {
        container: div_id,
        waveColor: 'tomato',
        progressColor: 'red',
        cursorColor: 'red',
        cursorWidth: 2,
        height: 128,
        barWidth: 2,
        barHeight: 1.5,
        barMinHeight: 0.1,
        barGap: null,
        responsive: true,
        hideScrollbar: true
    }
    return new Track(setting);
}
