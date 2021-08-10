function zerosLike(array) {
    zeros = [];
    for (var a of array) {
        zeros.push(0);
    }
    return zeros;
}

function deepCopyArray(array) {
    var newArray = [];
    for (var a of array) {
        if (Array.isArray(a)) {
            newArray.push(deepCopyArray(a));
        } else if (a.constructor == Object) {
            newArray.push(deepCopyDict(a))
        } else {
            newArray.push(a);
        }
    }
    return newArray;
}

function deepCopyDict(data) {
    var newData = {};
    for (var key in data) {
        var d = data[key];
        var copied = null;
        if (Array.isArray(d)) {
            copied = deepCopyArray(d);
        } else if (d.constructor == Object) { // check if dictionary
            copied = deepCopyDict(d);
        } else {
            copied = d;
        }
        newData[key] = d;
    }
    return newData;
}
