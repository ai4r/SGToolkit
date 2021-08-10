class MotionLibrary {
    constructor(_avatar, timeline, open_close_btn_id, apply_btn_id, delete_btn_id) {
        var that = this;
        this.avatar = _avatar
        this.timeline = timeline
        this.curFrame = 0
        this.motionRepeat = 0
        this.openCloseBtn = $(open_close_btn_id);
        this.openCloseBtn.click(function () {
            var $btn = $(this);
            var $content = $(this).parent().next(".content");
            $content.slideToggle(500, () => {
                if ($content.is(":visible")) {
                    $btn.text("Close motion library");
                    that.loadMotionFromDB();
                    window.dispatchEvent(new Event('resize'));  // to adjust avatar resolution
                } else {
                    $btn.text("Open motion library");
                    if (that.timer) {
                        clearInterval(that.timer)
                    }
                }
            });
        });

        $('#motion-library-add button').click(function (e) {
            e.preventDefault();  // cancel form submit
            if ($(this).attr("value") == "add") {
                that.addMotionLibrary();
            } else {  // add using array data
                that.addMotionLibraryManual();
            }
        });

        $(apply_btn_id).click(function () {
            if (that.selectedIndex >= 0) {
                var motion = that.motions[that.selectedIndex].motion
                var motionSpeed = parseInt(document.querySelector('input[name="motion-speed"]:checked').value);
                var flip_lr = $("#btn-flip-left-right").is(':checked')
                that.applyMotion(motion, motionSpeed, flip_lr)
            }
        });

        $("#btn-delete-library").click(function () {
            var modalDialog = $('#deleteLibraryModal')
            modalDialog.modal('hide');
            var delete_id = modalDialog.data('delete-id');
            get("api/delete_motion/" + delete_id, function (data) {
                console.log('motion deleted', data)
                that.loadMotionFromDB()
            });
        });
        /*
        $(delete_btn_id).click(function (){
          if(that.selectedIndex >= 0){
            var delete_id = that.motions[that.selectedIndex].id;
            var modalDialog = $('#deleteLibraryModal');
            modalDialog.data('delete-id', delete_id);
            modalDialog.modal('show');
          }
        });
        */
        this.motions = [];
        this.loadMotionFromDB();
        this.selectedIndex = -1
    }

    loadMotionFromDB() {
        // loading indicator
        var list = $('#motion-library-list')
        list.empty()
        var li = document.createElement('li');
        li.innerHTML = '<i class="fa fa-circle-o-notch fa-spin"></i> Loading...';
        list.append(li);

        // load
        var that = this;
        that.motions = [];
        get("api/motion", function (responseText) {
            var json = JSON.parse(responseText);
            for (var i = 0; i < json.length; i++) {
                that.motions.push(new MotionItem(json[i]._id.$oid, json[i].name, json[i].motion));
            }

            that.displayMotions();
        });
    }

    displayMotions() {
        var that = this;

        // remove existing motion list
        var list = $('#motion-library-list')
        list.empty()

        // create motion list
        that.selectedIndex = -1

        for (var i = 0; i < this.motions.length; i++) {
            var li = document.createElement('li');
            li.innerHTML = this.motions[i].name + " (" + this.motions[i].motion.length + ")";
            li.onclick = function () {
                var itemIdx = $(this).index();
                that.selectedIndex = itemIdx
                that.avatar.gestureKeypoints = that.motions[itemIdx].motion;
                that.curFrame = 0;
                that.motionRepeat = 0;
                if (that.timer) {
                    clearInterval(that.timer)
                }

                that.avatar.restPose();
                that.timer = setInterval(function () {
                    that.play();
                }, 30);

                $('#motion-library-list .list-group-item').removeClass('active');
                this.classList.add('active')
            };
            li.classList.add('list-group-item')
            li.classList.add('col-sm-4')  // multi-column
            list.append(li);
        }
    }

    play() {
        this.curFrame++;
        if (this.curFrame >= this.avatar.gestureKeypoints.length) {
            this.curFrame = 0;
            this.avatar.restPose();
            this.motionRepeat++;

            if (this.motionRepeat >= 5) {
                clearInterval(this.timer)
                this.motionRepeat = 0;
            }
        }

        this.avatar.moveBody(this.avatar.gestureKeypoints[this.curFrame]);
    }

    addMotionLibrary() {
        // get selected region from motion track
        var motionTrack = this.timeline.motionTrack
        if (motionTrack) {
            var selectedMotion = motionTrack.getSelectedKeypoints()

            console.log(selectedMotion)

            var motionName = $("#text-motion-name").val();
            if (motionName.length > 0) {
                var data = {"name": motionName, "motion": selectedMotion};
                post("api/motion", data, (data) => {
                    this.loadMotionFromDB()
                });
            }
        } else {
            bootbox.alert("Please select motion region first!");
        }
    }

    addMotionLibraryManual() {
        var that = this;
        bootbox.prompt({
            title: "Please input raw motion array.",
            inputType: 'textarea',
            callback: function (arrStr) {
                var motionName = $("#text-motion-name").val();
                if (motionName.length > 0 && arrStr != null) {
                    var array = JSON.parse(arrStr);
                    var data = {"name": motionName, "motion": array};
                    post("api/motion", data, (data) => {
                        that.loadMotionFromDB()
                    });
                }
            }
        });
        // bootbox.alert("!");
    }

    applyMotion(motion, speed, flip_lr) {
        var sampled_data = null;
        if (speed == 2 || speed == 3) {
            var tmp = [];
            for (var i = 0; i < motion.length; i += speed) {
                tmp.push(motion[i]);
            }
            sampled_data = tmp
        } else {
            sampled_data = motion
        }

        if (flip_lr) {
            // copy array
            sampled_data = cloneGrid(sampled_data)

            // invert x coordinates
            let nFrames = sampled_data.length
            let dataLength = sampled_data[0].length
            for (var i = 0; i < nFrames; i++) {
                for (var j = 0; j < dataLength; j += 3) {
                    sampled_data[i][j] *= -1;
                }
            }

            // switch left/right joints
            for (var i = 0; i < nFrames; i++) {
                for (var j = 9; j < 18; j++) {
                    var val = sampled_data[i][j];
                    sampled_data[i][j] = sampled_data[i][j + 9];
                    sampled_data[i][j + 9] = val;
                }
            }
        }

        timeline.updateMotionTrackControlData(sampled_data);
    }
}

class MotionItem {
    constructor(id, name, motion) {
        this.id = id
        this.name = name;
        this.motion = motion;
    }
}

function cloneGrid(grid) {
    // function code from https://ozmoroz.com/2020/07/how-to-copy-array/

    // Clone the 1st dimension (column)
    const newGrid = [...grid]
    // Clone each row
    newGrid.forEach((row, rowIndex) => newGrid[rowIndex] = [...row])
    return newGrid
}
