class Cell {

    constructor(frameId) {
        var $div = $("<div>", {
            "class": "track-frame",
            "data-fid": frameId
        });
        this.fid = frameId;
        this.div = $div;
        this.types = [];
        this.group = null;
    }

    setData(data) {
        this.data = data;
    }

    getData(key) {
        return this.data[key];
    }

    updateData(key, value) {
        this.data[key] = value;
    }

    setControlData(data) {
        this.controlData = data;
        if (!$(this.div).hasClass("modified")) {
            $(this.div).addClass("modified");
        }
    }

    getControlData(key) {
        if (!this.isModified()) {
            return null;
        }
        return this.controlData[key];
    }

    setGroupPartType(type) {
        if (this.types.indexOf(type) == -1) {
            this.types.push(type);
        }
        var $div = $(this.div);
        if (type == "start") {
            $div.addClass("group-start");
        } else if (type == "middle") {
            $div.addClass("group-middle");
        } else if (type == "end") {
            $div.addClass("group-end");
        }
    }

    resetGroupPartType() {
        this.types = [];
        $(this.div).removeClass("group-start");
        $(this.div).removeClass("group-middle");
        $(this.div).removeClass("group-end");
    }

    removeControlData() {
        if (this.isModified()) {
            $(this.div).removeClass("modified");
        }
        this.resetGroupPartType();
        this.controlData = null;
        this.group = null;
    }

    select() {
        $(this.div).addClass("selected");
    }

    deselect() {
        $(this.div).removeClass("selected");
    }

    toggleSelect() {
        if (this.isSelected()) {
            this.deselect();
        } else {
            this.select();
        }
    }

    isModified() {
        return $(this.div).hasClass("modified");
    }

    isSelected() {
        return $(this.div).hasClass("selected");
    }

    isInControlData(key) {
        if (!this.isModified()) {
            return false;
        }
        return key in this.controlData;
    }

    isInGroup() {
        return (this.group != null);
    }

    copyControlDataTo(cell) {
        var copiedControlData = deepCopyDict(this.controlData);
        cell.setControlData(copiedControlData);
    }

}

class CellGroup {
    constructor() {
        this.cells = []
    }

    setCells(cells) {
        this.cells = [];

        for (var cell of cells) {
            this.cells.push(cell);
            cell.group = this;
        }
        this.startFid = this.cells[0].fid;
        this.endFid = this.cells[this.cells.length - 1].fid;
        this.updateCellGroupType();
    }

    getCell(idx) {
        return this.cells[idx];
    }

    hasCell(cell) {
        for (var _cell of cells) {
            if (_cell.fid == cell.fid) {
                return true;
            }
        }

        return false;
    }

    getNumCells() {
        return this.cells.length;
    }

    setControlData(controlData) {
        var numData = controlData.length;
        for (var i = 0; i < numData; i++) {
            var cell = this.cells[i];
            cell.setControlData(controlData[i]);
        }
    }

    removeControlData() {
        for (var cell of this.cells) {
            cell.removeControlData();
        }
    }

    moveTo(newCells) {
        var numCells = newCells.length;
        if (numCells != this.cells.length) {
            return;
        }

        var controlData = [];
        for (var cell of this.cells) {
            controlData.push(cell.controlData);
            cell.removeControlData();
            cell.group = null;
        }

        this.setCells(newCells);
        this.setControlData(controlData);
    }

    updateCellGroupType() {

        // This is just for css.
        var numCells = this.getNumCells();
        if (numCells == 1) {
            var cell = this.cells[0];
            cell.resetGroupPartType();
            cell.setGroupPartType("start");
            cell.setGroupPartType("middle");
            cell.setGroupPartType("end");
        } else {
            var firstCell = this.cells[0];
            firstCell.resetGroupPartType();
            firstCell.setGroupPartType("start");
            firstCell.setGroupPartType("middle");
            var lastCell = this.cells[numCells - 1];
            lastCell.resetGroupPartType();
            lastCell.setGroupPartType("end");
            lastCell.setGroupPartType("middle");

            for (var i = 1; i < numCells - 1; i++) {
                var cell = this.cells[i];
                cell.resetGroupPartType();
                cell.setGroupPartType("middle");
            }
        }
    }

    getMiddleCell() {
        var numCells = this.getNumCells();
        var middleIdx = Math.floor(numCells / 2);
        return this.cells[middleIdx];
    }

    copyControlDataTo(group) {
        var numCells = group.getNumCells();
        if (numCells != this.getNumCells()) {
            return;
        }

        for (var i = 0; i < numCells; i++) {
            this.cells[i].copyControlDataTo(group.cells[i]);
        }
    }

    remove() {
        for (var cell of this.cells) {
            cell.removeControlData();
        }
    }
}

class CellGroupSnapshot {
    // holds only control data (no frame data).
    constructor(group) {
        if (group.cells.length == 0) {
            return;
        }

        this.startFid = group.startFid;
        this.endFid = group.endFid;
        this.controlDatas = [];
        for (var cell of group.cells) {
            this.controlDatas.push(deepCopyDict(cell.controlData));
        }
        this.length = this.controlDatas.length;
    }
}
