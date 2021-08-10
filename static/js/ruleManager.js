class RuleManager {
    constructor(mainWindow, apply_btn_id, view_btn_id, modal_id) {
        var that = this
        this.mainWindow = mainWindow;
        this.modalId = modal_id;
        this.ruleDialog = null;
        this.rules = {};  // dictionary (key: word, value: list of (ruleid, motion_info))
        this.words = []

        $(apply_btn_id).click(() => {
            this.applyRules();
        });

        $(view_btn_id).click(() => {
            this.ruleDialog = $(this.modalId)
            this.updateRuleList();
            this.updateMotionList();
            this.ruleDialog.modal('show');
        });

        $('#rule-add').submit(function () {
            console.log('clicked add rule')
            $('#btn-add-rule').html('<i class="fa fa-circle-o-notch fa-spin"></i> Loading...');
            $('#btn-add-rule').prop('disabled', true);
            that.addRule();
            return false;
        });

        $("#ruleTable").on('click', '.btnDelete', function () {
            $(this).closest('tr').remove()
            var deleteId = $(this).closest('tr').data('id')
            get("api/delete_rule/" + deleteId, function (data) {
                console.log('rule deleted', data);
                that.getRules(false);
            });
        });

        this.getRules(false);
    }

    applyRules() {
        var words = this.words;
        var timeline = this.mainWindow.timeline

        for (var i = 0; i < words.length; i++) {
            var word = words[i][0].toLowerCase();
            var startTime = words[i][1];
            var endTime = words[i][2];

            console.log(word);

            if (word in this.rules) {
                // select randomly if the rules associated to the same word exist
                var nMotions = this.rules[word].length
                var randomIdx = Math.floor(Math.random() * nMotions)
                var motion = this.rules[word][randomIdx][1].motion
                timeline.motionTrack.updateMultipleKeypointsAsGroup(motion, (endTime + startTime) / 2.0, true)
            }
        }
    }

    updateRuleList() {
        var dialog = this.ruleDialog;
        dialog.find("#ruleTable > tbody").empty();

        var rules = this.rules

        // sort by name
        var rules = Object.keys(rules).sort().reduce(function (Obj, key) {
            Obj[key] = rules[key];
            return Obj;
        }, {});

        // add rows
        for (var key in rules) {
            for (var i = 0; i < rules[key].length; i++) {
                var ruleId = rules[key][i][0];
                var motionName = rules[key][i][1].name;
                var motion = rules[key][i][1].motion;
                dialog.find('#ruleTable > tbody:last-child').append('<tr data-id="' + ruleId + '"><td>' + key + '</td><td>' + motionName + '</td><td>' + 'length: ' + motion.length + '</td><td><button class="btnDelete btn btn-secondary btn-sm">Delete</button></td></tr>');
            }
        }
    }

    updateMotionList() {
        var selector = this.ruleDialog.find("#select-motion");
        selector.empty();
        var motions = this.mainWindow.motionLibrary.motions
        for (var i = 0; i < motions.length; i++) {
            selector.append(new Option(motions[i].name, motions[i].id))
        }
    }

    getRules(updateList = false) {
        get("api/rule", (responseText) => {
            var items = JSON.parse(responseText);
            this.rules = {}
            for (var i = 0; i < items.length; i++) {
                if (!this.rules[items[i].word]) {
                    this.rules[items[i].word] = [];
                }
                var ruleId = items[i]._id.$oid
                this.rules[items[i].word].push([ruleId, items[i].motion_info[0]])
            }

            if (updateList) {
                this.updateRuleList();
            }

            $('#btn-add-rule').html('Add rule');
            $('#btn-add-rule').prop('disabled', false);
        });
    }

    loadWords(wordsWithTimestamps) {
        this.words = wordsWithTimestamps
    }

    addRule() {
        var word = $("#rule-name").val();
        if (word.length > 0) {
            var motionId = $('#select-motion option:selected').val();
            var data = {"word": word, "description": word, "motion": motionId};
            post("api/rule", data, (data) => {
                this.getRules(true);
            });
        }
    }
}
