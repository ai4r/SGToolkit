class SliderAndLabel {
    constructor(sliderDivId, setting, labelDivId) {
        this.sliderDivId = sliderDivId;
        this.slider = new Slider(sliderDivId, setting);
        this.labelDivId = labelDivId;
    }

    updateLabel(value) {
        $(this.labelDivId).text(value);
    }

    update(value) {
        this.slider.setValue(value);
        this.updateLabel(value);
    }

    setSliderUpdateCallback(callback) {
        $(this.sliderDivId).on("slide", callback);
    }

    getValue() {
        return this.slider.getValue();
    }
}

class StylePannel {
    constructor() {
        this.sliders = {}
        this.sliderUpdateCallback = null;
        this.styleNames = []
    }

    addStyle(name, sliderDivId, sliderValueLabelId) {
        var slider = this.createSlider(name, sliderDivId, sliderValueLabelId);
        this.sliders[name] = slider;
        this.styleNames.push(name);
    }

    addStylePreset(name, btnId, styleVal) {
        var that = this
        $(btnId).click(function () {
            var data = styleVal
            that.setValues(data)
            if (that.sliderUpdateCallback != null) {
                var copiedData = {};  // use copied data because it is altered in sliderUpdateCallback fn
                Object.assign(copiedData, data);
                that.sliderUpdateCallback(copiedData);
            }
        });
    }

    createSlider(styleName, divId, valueLabelId) {
        var that = this;

        var setting = {min: -3, max: 3, step: 0.1, value: 0};
        var sliderAndLabel = new SliderAndLabel(divId, setting, valueLabelId);

        sliderAndLabel.setSliderUpdateCallback(function (slideEvt) {
            sliderAndLabel.updateLabel(slideEvt.value);
            if (that.sliderUpdateCallback != null) {
                var data = {};
                for (var name of that.styleNames) {
                    data[name] = that.sliders[name].getValue();
                }
                data[styleName] = slideEvt.value;
                that.sliderUpdateCallback(data);
            }
        });
        return sliderAndLabel;
    }

    setSliderUpdateCallback(callback) {
        this.sliderUpdateCallback = callback;
    }

    getStyleNames() {
        return this.styleNames;
    }

    setValues(data) {
        for (var key in data) {
            if (key in this.sliders) {
                this.sliders[key].update(data[key]);
            }
        }
    }
}
