class HistoryManager {
    constructor() {
        this.motionTrackSnapshots = [];
        this.styleTrackSnapshots = [];
        this.cursor = -1;
    }

    addSnapshots(motionTrack, styleTrack) {

        if (this.canRedo()) {
            // cursor not at the end.
            this.resetFutureFromNow();
        }

        this.motionTrackSnapshots.push(motionTrack.takeSnapshot());
        this.styleTrackSnapshots.push(styleTrack.takeSnapshot());
        this.cursor += 1;
    }

    clearHistory() {
        this.motionTrackSnapshots = [];
        this.styleTrackSnapshots = [];
        this.cursor = -1;
    }

    canUndo() {
        return this.cursor >= 0;
    }

    canRedo() {
        return this.cursor + 1 < this.motionTrackSnapshots.length;
    }

    undo(motionTrack, styleTrack) {
        if (!this.canUndo()) {
            return;
        }
        this.cursor -= 1;
        console.log("undo", this.cursor);
        this.loadCurrentCursor(motionTrack, styleTrack);
    }

    redo(motionTrack, styleTrack) {
        if (!this.canRedo()) {
            return;
        }
        this.cursor += 1;
        console.log("redo", this.cursor);
        this.loadCurrentCursor(motionTrack, styleTrack);
    }

    loadCurrentCursor(motionTrack, styleTrack) {
        if (this.cursor < 0) {
            this.tracksToInitialState(motionTrack, styleTrack);
            return;
        }
        var ms = this.motionTrackSnapshots[this.cursor];
        var ss = this.styleTrackSnapshots[this.cursor];

        motionTrack.loadSnapshot(ms);
        styleTrack.loadSnapshot(ss);
    }

    tracksToInitialState(motionTrack, styleTrack) {
        // initial state
        motionTrack.clearGroups();
        styleTrack.clearGroups();
    }

    resetFutureFromNow() {
        this.motionTrackSnapshots = this.motionTrackSnapshots.slice(0, this.cursor + 1);
        this.styleTrackSnapshots = this.styleTrackSnapshots.slice(0, this.cursor + 1);
    }
}
