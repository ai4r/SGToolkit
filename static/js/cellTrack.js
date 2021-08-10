class CellTrack {
    constructor(setting) {
        this.container = setting['container'];
        this.numFrames = setting['numFrames'];
        this.duration = setting['duration'];

        this.pointedCell = null;

        //this.addDeselectAllCallback();
        this.numSelected = 0;

        this.selectedCallbacks = [];
        this.cellGroupCreationInvalidCallback = null;

        this.isBypassControlData = false;
        this.copiedGroup = null;
    }

    destruct() {
        for (var cell of this.cells) {
            $(cell.div).remove();
        }

        this.removeSelectionCallbacks();
    }

    removeSelectionCallbacks() {
        $(this.container).ready(function () {
        });
    }

    setSelectionCallbacks() {
        // this should be called after adding all the cells.
        var that = this;
        var isShifting = false;
        var isDown = false;
        var downFid = -1;
        var prevFid = -1;
        var clickedGroup = null;
        this.jquerySelectAllChildDiv().mousedown(function (evt) {
            if (isDown) {
                // previent multi selection at once.
                return;
            }
            var cell = that.getCellByDiv(evt.target);
            that.deselectAll();

            isShifting = cell.isModified();

            cell.select();
            isDown = true;
            that.numSelected = 1;

            prevFid = cell.fid;
            downFid = cell.fid;
            clickedGroup = cell.group;
            evt.preventDefault();

        }).mouseenter(function (evt) {
            var cell = that.getCellByDiv(evt.target);

            if (isShifting) {
                if (cell.isInGroup() && cell.group != clickedGroup) {
                    return;
                }

                that.deselectAll();

                var dist = cell.fid - prevFid;
                that.shiftAsGroupDist(prevFid, dist);

            } else if (isDown) {

                that.selectOnlyBetween(downFid, cell.fid);
                that.numSelected = that.getSelectedCells().length;
            }

            prevFid = cell.fid;
        }).mouseleave(function (evt) {
            // 'leave' fires before 'enter'

            //var cell = that.getCellByDiv(evt.target);

        });
        $(document).mouseup(function (evt) {

            var selectedCells = that.getSelectedCells();
            var cell = null;
            if (selectedCells.length > 0) {
                cell = selectedCells[0];
            } else if (isShifting) {
                var cell = clickedGroup.getMiddleCell();
                cell.select();
            }

            if (isDown) {
                that.callCellSelectedCallbacks(cell);
            }

            isShifting = false;
            isDown = false;
            prevFid = -1;
            clickedGroup = null;
            return;

        });

    }

    callCellSelectedCallbacks(cell) {
        for (var callback of this.selectedCallbacks) {
            callback(cell);
        }
    }

    jquerySelectAllChildDiv() {
        return $(this.container).children(".cell-track > div");
    }

    addDeselectAllCallback() {
        //when empty space clicked
        $('body').click(function (e) {
            if (e.target.classList.length == 0) {
                this.deselectAll();
            }
        }.bind(this));
    }

    deselectAll() {
        var selected = this.getSelectedCells();
        for (const cell of selected) {
            cell.deselect();
        }
        this.numSelected = 0;
        //this.selection.clearSelection();
    }

    seekTo(time) {
        var fid = this.timeToFrameIdx(time);
        this.selectOnly(fid);
    }

    seekToIfOneCellSelected(time) {
        if (!this.isOnlyOneCellSelected()) {
            return;
        }
        this.seekTo(time);
    }

    timeToFrameIdx(time) {
        var idx = Math.floor(this.numFrames * time / this.duration);
        if (idx >= this.numFrames) {
            idx = this.numFrames - 1;
        } else if (idx < 0) {
            idx = 0;
        }
        return idx;
    }

    selectOnly(fid) {
        this.deselectAll();

        var cell = this.cells[fid];
        cell.select();
    }

    selectOnlyBetween(fid1, fid2) {
        var lowerFid = fid1;
        var higherFid = fid2;

        if (lowerFid > higherFid) {
            var tmp = lowerFid;
            lowerFid = higherFid;
            higherFid = tmp;
        }

        for (var cell of this.cells) {
            var fid = cell.fid;
            if (fid >= lowerFid && fid <= higherFid && !cell.isSelected()) {
                cell.select();
            } else if ((fid < lowerFid || fid > higherFid) && cell.isSelected()) {
                cell.deselect();
            }
        }
    }

    getFirstSelectedCell() {
        var selectedCells = this.getSelectedCells();
        if (selectedCells.length == 0) {
            return null;
        }
        return selectedCells[0];
    }

    getSelectedCells() {
        var selected = [];
        for (var cell of this.cells) {
            if (cell.isSelected()) {
                selected.push(cell);
            }
        }
        return selected;
    }

    getCellByDiv(div) {
        var fid = $(div).data("fid");
        return this.cells[fid];
    }

    addSelectedCallback(callback) {
        this.selectedCallbacks.push(callback);
    }

    setCellGroupCreationInvalidCallback(callback) {
        this.cellGroupCreationInvalidCallback = callback;
    }

    deconstruct() {
        $(this.container).empty();
    }

    createGroup(fromFid, numFrames) {
        var cellsToGroup = [];
        // check if can create group
        if (!this.canCreateNewGroup(fromFid, numFrames)) {
            if (this.cellGroupCreationInvalidCallback != null) {
                this.cellGroupCreationInvalidCallback();
            }
            return null;
        }

        for (var i = 0; i < numFrames; i++) {
            var fid = i + fromFid;
            var cell = this.cells[fid];
            cellsToGroup.push(cell);
            if (cell.isInGroup()) {
                return null;
            }
        }

        var group = new CellGroup();
        group.setCells(cellsToGroup);
        return group;
    }

    canCreateNewGroup(fromFid, numFrames) {
        return fromFid >= 0 && (fromFid + numFrames) <= this.numFrames && !this.isCellModifiedIn(fromFid, numFrames);
    }

    deleteSelectedCellControlData() {

        if (!this.isAnyCellSelected()) {
            return;
        }

        var cell = this.getFirstSelectedCell();
        if (!cell.isModified()) {
            return;
        }

        var group = cell.group;
        group.removeControlData();

    }

    shiftAsGroupDist(fid, dist) {
        var cell = this.cells[fid];
        if (!cell.isInGroup()) {
            return;
        }
        var group = cell.group;

        var newStartFid = group.startFid + dist;
        var newEndFid = group.endFid + dist;
        if (newStartFid < 0 || newEndFid >= this.cells.length) {
            return;
        }

        // prevent shifting when there is another modified cell (group) in the middle.
        if (dist < 0 && this.isCellModifiedIn(newStartFid, -dist)) {
            return;
        } else if (dist > 0 && this.isCellModifiedIn(newEndFid - dist + 1, dist)) {
            return;
        }

        var newCellsToGroup = this.cells.slice(newStartFid, newEndFid + 1);
        cell.group.moveTo(newCellsToGroup);

    }

    isCellModifiedIn(startIdx, length) {
        for (var i = 0; i < length; i++) {
            var idx = i + startIdx;
            var cell = this.cells[idx];
            if (cell.isModified()) {
                return true;
            }
        }

        return false;
    }

    isAnyCellSelected() {
        var selected = this.getSelectedCells();
        return selected.length > 0;
    }

    isOnlyOneCellSelected() {
        var selected = this.getSelectedCells();
        return selected.length == 1;
    }

    isAnyCellModified() {
        for (var cell of this.cells) {
            if (cell.isModified()) {
                return true;
            }
        }

        return false;
    }

    copyGroup() {
        var selected = this.getSelectedCells();
        if (selected.length == 0) {
            return;
        }

        this.copiedGroup = selected[0].group;
    }

    pasteGroup() {
        if (this.copiedGroup == null) {
            return;
        }

        var selected = this.getSelectedCells();
        if (selected.length == 0) {
            return;
        }
        var startCell = selected[0];
        var startFid = startCell.fid;
        var numFrames = this.copiedGroup.getNumCells();

        var group = this.createGroup(startFid, numFrames);
        if (group != null) {
            this.copiedGroup.copyControlDataTo(group);
        }

        this.callCellSelectedCallbacks(startCell);
    }

    takeSnapshot() {
        var groups = this.getAllGroups();
        var groupSnapshots = [];
        for (var g of groups) {
            groupSnapshots.push(new CellGroupSnapshot(g));
        }

        return new CellTrackSnapshot(groupSnapshots);
    }

    getAllGroups() {
        var groups = [];
        var group = null;
        for (var cell of this.cells) {
            if (cell.group != group && cell.group != null) {
                groups.push(cell.group);
            }
            group = cell.group;
        }

        return groups;
    }

    loadSnapshot(snapshot) {

        this.clearGroups();
        if (snapshot.isEmpty()) {
            return;
        }

        for (var cgs of snapshot.cellGroupSnapshots) {
            this.loadCellGroupSnapshot(cgs);
        }
        this.callCellSelectedCallbacks(null);
    }

    clearGroups() {
        var groups = this.getAllGroups();
        for (var g of groups) {
            g.remove();
        }
    }

    loadCellGroupSnapshot(cellGroupSnapshot) {
        var startFid = cellGroupSnapshot.startFid;
        var endFid = cellGroupSnapshot.endFid;
        var controlData = cellGroupSnapshot.controlDatas;
        var length = endFid - startFid + 1;
        var group = this.createGroup(startFid, length);
        group.setControlData(controlData);
    }
}

class CellTrackSnapshot {
    constructor(cellGroupSnapshots) {
        this.cellGroupSnapshots = cellGroupSnapshots;
    }

    isEmpty() {
        return this.cellGroupSnapshots == null || this.cellGroupSnapshots.length == 0;
    }
}

class MotionCellTrack extends CellTrack {
    constructor(setting, motionKeypoints) {
        super(setting);
        this.motionKeypoints = motionKeypoints;
        this.cells = this.addCells(this.container, this.numFrames, this.motionKeypoints);
        this.setSelectionCallbacks();
    }

    addCells(container, numCells, keypoints) {
        var interval = this.duration / this.numFrames;
        var cells = [];
        for (var i = 0; i < numCells; i++) {
            var keypoint = keypoints[i];
            var startTime = interval * i;
            var endTime = interval * (i + 1);

            var cell = new Cell(i);

            var data = {
                "start-time": interval * i,
                "end-time": interval * (i + 1),
                "keypoint": keypoint
            };

            cell.setData(data);
            cells.push(cell);
            $(container).append(cell.div);
        }

        return cells;
    }

    syncAvatarWithCursor(avatar) {
        var keypoint = this.getCurrentKeypoint();
        avatar.moveBody(keypoint);
    }

    getCurrentKeypoint() {

        if (!this.isAnyCellSelected()) {
            return null;
        }

        var cell = this.getFirstSelectedCell();
        var keypoint = null;
        if (cell.isModified() && !this.isBypassControlData) {
            keypoint = cell.getControlData("keypoint-modified");
        } else {
            keypoint = cell.getData("keypoint");
        }

        return keypoint;
    }

    getSelectedKeypoints() {
        var selected = this.getSelectedCells();
        if (selected.length == 0) {
            return null;
        }

        var motion = []
        if (selected[0].isModified()) {
            // get current modified group
            var group = selected[0].group;

            for (var cell of group.cells) {
                keypoint = cell.getControlData("keypoint-modified");
                motion.push(keypoint)
            }
        } else {
            // get selected region
            for (var cell of selected) {
                var keypoint = null;
                if (cell.isModified() && !this.isBypassControlData) {
                    keypoint = cell.getControlData("keypoint-modified");
                } else {
                    keypoint = cell.getData("keypoint");
                }
                motion.push(keypoint)
            }
        }

        return motion;
    }

    updateBaseKeypoints(keypoints) {
        var that = this;
        $.each(keypoints, function (i, kp) {
            var cell = that.cells[i];
            cell.updateData('keypoint', kp);
        });
    }

    updateKeypointsToFirstSelectedCell(keypoints) {
        var selected = this.getSelectedCells();
        if (selected.length == 0) {
            return;
        }
        if (selected.length > 1) {
            return;
        }
        var cell = selected[0];
        if (cell.isInGroup() && cell.group.getNumCells() > 1) {
            return;
        }

        var controlValues = [{"keypoint-modified": keypoints}];
        var group = cell.group;
        if (!cell.isInGroup()) {
            group = this.createGroup(cell.fid, 1);
        }
        if (group != null) {
            group.setControlData(controlValues);
        }
    }

    updateMultipleKeypointsAsGroup(motion, insertTime, align_center = false) {
        if (!this.isAnyCellSelected()) {
            return;
        }

        if (typeof insertTime !== "undefined") {

        } else {
            var cell = this.getFirstSelectedCell();
            var insertTime = cell.getData('start-time');
        }

        var insertIdx = this.timeToFrameIdx(insertTime);
        var startIdx = insertIdx;

        if (align_center) {
            if (motion.length > 1) {
                startIdx -= Math.round(motion.length / 2.0);
            }
            startIdx = Math.max(0, startIdx)
            startIdx = Math.min(startIdx, this.cells.length - motion.length)
        }

        var controlValues = [];
        for (var i = 0; i < motion.length; i++) {
            controlValues.push({"keypoint-modified": motion[i]});
        }
        var group = this.createGroup(startIdx, motion.length);
        if (group != null) {
            group.setControlData(controlValues);
        }
    }

    getKeypointsConstraint() {
        var cells = this.cells;
        var result = []
        for (var cell of cells) {
            var keypoint = null;
            var maskbit = null;
            if (cell.isModified()) {
                keypoint = cell.getControlData("keypoint-modified");
                maskbit = 1;
            } else {
                keypoint = cell.getData('keypoint');
                maskbit = 0;
            }
            var copiedArray = deepCopyArray(keypoint);
            copiedArray.push(maskbit)
            result.push(copiedArray);
        }
        return result;
    }

    interpolateTwoPoses() {
        if (!this.isAnyCellSelected()) {
            return;
        }

        var i;
        var anchorIdx = -1
        for (i = 0; i < this.cells.length; i++) {
            if (this.cells[i].isSelected()) {
                anchorIdx = i;
                break;
            }
        }

        var modifiedCellIdx1 = -1
        var modifiedCellIdx2 = -1
        var kps1;
        var kps2;
        for (i = anchorIdx; i >= 0; i--) {
            if (this.cells[i].isModified()) {
                modifiedCellIdx1 = i
                kps1 = this.cells[i].getControlData("keypoint-modified");
            }
        }
        for (i = anchorIdx; i < this.cells.length; i++) {
            if (this.cells[i].isModified()) {
                modifiedCellIdx2 = i
                kps2 = this.cells[i].getControlData("keypoint-modified");
            }
        }

        if (modifiedCellIdx1 == -1 || modifiedCellIdx2 == -1) {
            return;
        }

        // interpolate and set control poses
        var w;
        var controlValues = [];
        for (i = modifiedCellIdx1; i <= modifiedCellIdx2; i++) {
            w = (i - modifiedCellIdx1) / (modifiedCellIdx2 - modifiedCellIdx1);

            var kps = Array.from(kps1);
            for (var j = 0; j < kps1.length; j++) {
                kps[j] = (1 - w) * kps1[j] + w * kps2[j];
            }
            controlData = {"keypoint-modified": kps};
            controlValues.push({"keypoint-modified": kps});
        }

        this.cells[modifiedCellIdx1].removeControlData();
        this.cells[modifiedCellIdx2].removeControlData();

        var length = modifiedCellIdx1 - modifiedCellIdx1 + 1;
        var group = this.createGroup(modifiedCellIdx1, length);
        if (group != null) {
            group.setControlData(controlValues);
        }
        //this.setCellGroupControlData(modifiedCellIdx1, modifiedCellIdx2 - modifiedCellIdx1 + 1, controlValues);

    }
}

class StyleCellTrack extends CellTrack {
    constructor(setting, styleNames) {
        super(setting);
        this.styleNames = styleNames;
        this.cells = this.addCells(this.container, this.numFrames, styleNames);
        this.setSelectionCallbacks();
    }

    addCells(container, numCells, styleNames) {
        var interval = this.duration / this.numFrames;
        var cells = [];
        for (var i = 0; i < numCells; i++) {
            var startTime = interval * i;
            var endTime = interval * (i + 1);

            var cell = new Cell(i);
            var data = {
                "start-time": interval * i,
                "end-time": interval * (i + 1)
            };
            for (var name of styleNames) {
                data[name] = 0; // default value
            }
            cell.setData(data);
            cells.push(cell);
            $(container).append(cell.div);
        }

        return cells;
    }

    updateStyleControlForSelectedCells(data) {
        if (!this.isAnyCellSelected()) {
            return
        }
        var selected = this.getSelectedCells();

        var group = selected[0].group;
        if (group == null) {
            var updateStartFid = selected[0].fid;
            var numCellsToModify = selected.length;
            group = this.createGroup(updateStartFid, numCellsToModify);
        }

        var numCells = group.getNumCells();

        var controlValues = [];
        for (var i = 0; i < numCells; i++) {
            controlValues.push(data);
        }
        group.setControlData(controlValues);
    }

    syncStylePanelWithCursor(stylePanel) {
        if (!this.isAnyCellSelected()) {
            return
        }
        var param = this.getCurrentStyleParam();
        stylePanel.setValues(param);
    }

    getCurrentStyleParam() {
        var cell = this.getFirstSelectedCell();
        return this.getStyleParamOf(cell);
    }

    getStyleParamOf(cell) {
        var param = {};

        for (var name of this.styleNames) {
            if (cell.isModified() && name in cell.controlData) {
                param[name] = cell.controlData[name];
            } else {
                param[name] = cell.data[name];
            }
        }
        return param;
    }

    getStyleConstraints() {
        var cells = this.cells;
        var result = []
        for (var cell of cells) {
            var styleVector = [];
            var param = this.getStyleParamOf(cell);

            for (var name of this.styleNames) {
                styleVector.push(param[name]);
            }

            result.push(styleVector);
        }
        return result;
    }
}
