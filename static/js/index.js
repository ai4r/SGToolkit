var timeline = null;
var motionLibrary = null;
var ruleManager = null;

var avatar = null;
var avatarInLibrary = null;
var avatarInterface = null;
var stylePanel = null;
var lastGenerationResult = null;
var uploadedAudioPath = null;

$(window).on('load', function () {

    avatar = setupAvatar("renderCanvas");
    stylePanel = createStylePanel();
    timeline = new Timeline(avatar, stylePanel, "#waveform", "word-canvas", "#motion-ctrl-track", "#style-ctrl-track");

    avatar.scene.onReadyObservable.addOnce(function () {
        avatarInterface = new AvatarInterface(avatar);
        avatarInterface.gizmoDragEndCallback = applyMotionControlToTimeline;
    });
    avatarInLibrary = setupAvatar("renderInLibrary");
    motionLibrary = new MotionLibrary(avatarInLibrary, timeline, "#btn-open-motion-library", "#btn-apply-motion", "#btn-delete-motion")
    ruleManager = new RuleManager(this, "#btn-apply-rule", "#btn-view-rule", "#viewRuleModal")

    // attach button callbacks
    $("#btn-generate").click(generate);
    $("#btn-generate-sure").click(postInputText); // btn in generate modal
    $("#btn-play").click(togglePlay);
    $("#btn-update").click(postInputTextAndConstraints);
    $("#btn-export").click(exportData);
    $("#btn-open-import-dialog").click(function () {
        $("#importJson").modal('show');
    });
    $("#btn-import").click(importData);
    $('#btn-avatar-edit-off').click(function () {
        avatarInterface.turnOffEditMode();
    });
    $('#btn-avatar-edit-position').click(function () {
        avatarInterface.turnOnEditMode(avatar, 'position');
    });
    $('#btn-avatar-edit-rotation').click(function () {
        avatarInterface.turnOnEditMode(avatar, 'rotation');
    });
    $('#btn-help').click(function () {
        $('#help-modal').modal('show');
    });
    $('#btn-undo').click(function () {
        timeline.undoTracks();
        updateUndoReduButtonState();
    });
    $('#btn-redo').click(function () {
        timeline.redoTracks();
        updateUndoReduButtonState();
    })

    // sample text
    $('#sampleTextDropdown .dropdown-item').click(function (e) {
        let text = $(this).text();
        if (text[2] == '.') {
            text = text.substring(4)
        }

        $("#text-input").val(text);

        // set voice btn state
        if ($(this).data("audio")) {
            uploadedAudioPath = $(this).data("audio");
            $("#voice-female").removeAttr('checked');
            $("#voice-female").parent().removeClass('active');
            $("#voice-file").attr('checked', 'checked');
            $("#voice-file").parent().addClass('active');
        } else {
            $("#voice-file").removeAttr('checked');
            $("#voice-file").parent().removeClass('active');
            $("#voice-female").attr('checked', 'checked');
            $("#voice-female").parent().addClass('active');
        }
    })

    // for audio upload form
    FilePond.registerPlugin(FilePondPluginFileValidateType);
    FilePond.setOptions({
        server: {
            url: './',
            process: {
                url: './upload_audio',
                headers: {'X-CSRF-TOKEN': $('input[name="csrf_token"]').val()},
                onload: onAudioUploaded,
            }
        }
    })
    FilePond.parse(document.body);
    $("input[name$='voice']").click(function () {
        var val = $(this).val();
        if (val == 'file') {
            $("#audio-upload-form").show();
        } else {
            $("#audio-upload-form").hide();
        }
    });

    // in modal
    $("#btn-delete-annotation").click(function () {
        $('#deleteModal').modal('hide');
        timeline.deleteSelectedCellControlData();
    });

    $("#btn-motion-track-bypass").on('change', function () {
        var checked = $(this).is(':checked')
        timeline.toggleMotionTrackControlDataBypass(checked);
    });

    $("#btn-scenario-selector").on('change', function () {
        if ($(this).is(':checked')) {
            $("#btn-update").html("Apply Control")
        } else {
            $("#btn-update").html("Interpolate")
        }
    });

    timeline.controlDataModificationInvalidWarningId = "#controlDataWarning";

    // disable mouse wheel event in babylon canvases
    $("#renderCanvas").bind("wheel mousewheel", function (e) {
        e.preventDefault()
    });
    $("#renderInLibrary").bind("wheel mousewheel", function (e) {
        e.preventDefault()
    });

    addStylePanelSliderUpdateCallback();

});

function createStylePanel() {
    var stylePanel = null;
    stylePanel = new StylePannel();
    stylePanel.addStyle("speed", "#speed-style-slider", "#speed-style-val-label");
    //stylePanel.addStyle("accel", "#accel-style-slider", "#accel-style-val-label");
    stylePanel.addStyle("space", "#space-style-slider", "#space-style-val-label");
    stylePanel.addStyle("handedness", "#handedness-style-slider", "#handedness-style-val-label");

    stylePanel.addStylePreset("happy", "#btn-preset-happy", {"speed": 2, "space": 1, "handedness": 0})
    stylePanel.addStylePreset("sad", "#btn-preset-sad", {"speed": -1, "space": 0, "handedness": 0})
    stylePanel.addStylePreset("angry", "#btn-preset-angry", {"speed": 2.5, "space": 2, "handedness": 0})

    return stylePanel;
}

function addStylePanelSliderUpdateCallback() {
    stylePanel.setSliderUpdateCallback(function (data) {
        if (timeline.styleTrack == null) {
            return;
        }
        timeline.updateStyleTrackControlData(data);
    });
}

$(window).keydown(function (e) {
    var code = e.code;
    var target = $(e.target)
    if (!target.is('textarea') && !target.is('input')) {
        if (code == "Delete" || code == "KeyD") {
            $('#deleteModal').modal('show');
        } else if (code == "KeyF") {
            //timeline.fillMotionControl();
        } else if (e.ctrlKey && code == "KeyC") {
            timeline.copySelectedCellGroup();
        } else if (e.ctrlKey && code == "KeyV") {
            timeline.pasteSelectedCellGroup();
        }
    }
});

function setupAvatar(id) {

    var canvas = document.getElementById(id); // Get the canvas element
    return new Avatar(canvas);
}

function afterGeneratedCallback(result) {
    $("#loading-modal").modal('hide');
    console.log("generated", result);
    if (result['msg'] === 'success') {
        lastGenerationResult = result;
        var audioFilename = result['audio-filename'];
        var wordsWithTimestamps = result['words-with-timestamps'];

        var motionKeypoints = result['output-data'];
        let is_manual_mode = !$("#btn-scenario-selector").is(':checked')
        if (is_manual_mode) {
            // set mean pose for all frames
            var nFrames = result['output-data'].length
            for (var i = 0; i < nFrames; i++) {
                motionKeypoints[i] = avatar.meanVec;
            }
        }

        timeline.load(audioFilename, wordsWithTimestamps, motionKeypoints, stylePanel.getStyleNames());

        timeline.setPlayCallback(function () {
            // bypass motion control when play
            var toggleBtn = $("#btn-motion-track-bypass")
            toggleBtn.bootstrapToggle('on');
            $("#btn-play").text("Pause");
        })

        timeline.setPauseCallback(function () {
            var toggleBtn = $("#btn-motion-track-bypass")
            toggleBtn.bootstrapToggle('off');
            $("#btn-play").text("Play");
        })

        ruleManager.loadWords(wordsWithTimestamps)

        if (is_manual_mode) {
            // set first and last pose controls
            // executed after 1 second (wait for track creation)
            setTimeout(function () {
                var restPose = avatar.meanVec;
                timeline.motionTrack.updateMultipleKeypointsAsGroup([restPose], 0);
                timeline.motionTrack.updateMultipleKeypointsAsGroup([restPose], timeline.motionTrack.duration);
            }, 1000);
        }
    } else {
        bootbox.alert("Hmm! Something went wrong. Please reload the page. If it happens again, please contact the maintainer.");
    }

    updateUndoReduButtonState();
}

function afterUpdatedCallback(result) {
    $("#loading-modal").modal('hide');
    if (result['msg'] === 'success') {
        if (!('words-with-timestamps' in result)) {  // reuse the previous word information in the manual mode
            result['words-with-timestamps'] = lastGenerationResult['words-with-timestamps'];
        }
        lastGenerationResult = result;
        var motionKeypoints = result['output-data'];
        timeline.updateMotionKeypoitns(motionKeypoints);
        timeline.setCursorToStart();
    } else {
        bootbox.alert("Hmm! Something went wrong. Please reload the page. If it happens again, please contact the maintainer.");
    }
    updateUndoReduButtonState();
}

function updateUndoReduButtonState() {
    if (timeline.canUndo()) {
        $("#btn-undo").removeClass("disabled");
    } else {
        $("#btn-undo").addClass("disabled");
    }

    if (timeline.canRedo()) {
        $("#btn-redo").removeClass("disabled");
    } else {
        $("#btn-redo").addClass("disabled");
    }
}

function applyMotionControlToTimeline() {
    if (timeline.motionTrack == null) {
        return;
    }
    var keypoints = avatar.modelKeypointsToArray();
    timeline.motionTrack.updateKeypointsToFirstSelectedCell(keypoints);
}

function generate() {
    if (timeline.isAnyCellModified()) {
        $("#generateModal").modal('show');
        return;
    }
    postInputText();
}

function postInputText() {
    $("#generateModal").modal('hide');

    var inputText = $("#text-input").val();

    if (inputText.length <= 0) {
        bootbox.alert('<i class="fa fa-exclamation-triangle fa-2x" aria-hidden="true"></i>  Please input speech text.');
        return;
    }

    var voiceName = document.querySelector('input[name="voice"]:checked').value;
    if (voiceName == 'file') {
        voiceName = uploadedAudioPath
    }
    data = {"text-input": inputText, "voice": voiceName};
    $("#loading-modal").one('shown.bs.modal', function () {
        console.log("callback attached");
        postInput(data, afterGeneratedCallback);
    });
    $("#loading-modal").modal('show');

    timeline.clearHistory();
}

function postInputTextAndConstraints() {
    var inputText = $("#text-input").val();

    if (inputText.length <= 0) {
        bootbox.alert('<i class="fa fa-exclamation-triangle fa-2x" aria-hidden="true"></i>  Please input speech text.');
        return;
    }

    var voiceName = document.querySelector('input[name="voice"]:checked').value;
    if (voiceName == 'file') {
        voiceName = uploadedAudioPath
    }
    data = {"text-input": inputText, "voice": voiceName};

    var keypointConstraints = timeline.getMotionConstraints();
    data['keypoint-constraints'] = keypointConstraints;
    var styleConstraints = timeline.getStyleConstraints();
    data['style-constraints'] = styleConstraints;
    if ($("#btn-scenario-selector").is(':checked')) {
        data['is-manual-scenario'] = 0
    } else {
        data['is-manual-scenario'] = 1
    }

    timeline.saveTracksToHistory();
    $("#loading-modal").one('shown.bs.modal', function () {
        postInput(data, afterUpdatedCallback);
    });
    $("#loading-modal").modal('show');
}

function postInput(data, callback) {
    post("api/input", data, callback);
}

function post(url, data, callback) {
    $.ajax({
        type: "POST",
        contentType: "application/json; charset=utf-8",
        url: url,
        data: JSON.stringify(data),
        dataType: "json",
        success: callback
    });
}

function get(url, callback) {
    $.ajax({
        type: "GET",
        contentType: "application/json; charset=utf-8",
        url: url,
        success: callback
    })
}

function togglePlay() {
    if (timeline.isPlaying()) {
        timeline.pause();
    } else {
        timeline.play();
    }
}

function importData() {
    $("#importJson").modal('hide');

    var files = document.getElementById('importFilePath').files;
    if (files.length <= 0) {
        return false;
    }

    filepath = files.item(0)  // use the first one
    console.log(filepath)
    $("#btn-scenario-selector").bootstrapToggle('on');  // set to auto mode to avoid filling mean poses inside afterGeneratedCallback fn

    var fr = new FileReader();
    fr.onload = function (e) {
        var json = JSON.parse(e.target.result);
        console.log(json)
        afterGeneratedCallback(json)
    }
    fr.readAsText(filepath);
}

function exportData() {
    if (lastGenerationResult) {
        // export filename
        var now = new Date();
        var filename = now.toISOString().slice(0, 10) + '_' + now.getTime()

        // audio (.wav)
        var audioFilename = lastGenerationResult['audio-filename'];
        if (audioFilename) {
            var link = document.createElement("a");
            link.download = filename + '.wav';
            link.href = "media/" + audioFilename + "/" + link.download;
            link.click();
        }

        // input and output data (.json)
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastGenerationResult));
        var link = document.createElement('a');
        link.href = dataStr
        link.download = filename + ".json"
        link.click()
    } else {
        bootbox.alert("Nothing to save. Please synthesize first.");
    }
}

function onAudioUploaded(r) {
    r = $.parseJSON(r);
    let filepath = r.filename[0];
    uploadedAudioPath = filepath
}
